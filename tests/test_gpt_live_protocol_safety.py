from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import llm
from livekit.plugins.openai.realtime import GPTLiveModel

from .test_gpt_live_model import _completed, _connect_hook, _function_call_done, _response_event

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("status", [None, "incomplete", "in_progress", "failed"])
async def test_only_completed_function_items_dispatch_once(monkeypatch, status):
    ws = _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    calls = []
    session.on("function_call", calls.append)
    try:
        await session._update_session()
        await session._session_started_fut
        session._handle_event(_response_event("d", {"type": "response.created"}))
        unfinished = _function_call_done("a")
        if status is None:
            unfinished["item"].pop("status")
        else:
            unfinished["item"]["status"] = status
        session._handle_event(_response_event("d", unfinished))
        session._handle_event(_response_event("d", unfinished))
        assert not calls
        complete = _response_event("d", _function_call_done("a"))
        session._handle_event(complete)
        session._handle_event(complete)
        session._handle_event(_response_event("d", _completed("r")))
        await session._append_items(
            [llm.FunctionCallOutput(call_id="a", output="done", is_error=False)]
        )
        # Re-delivery after the response state was removed must not execute the tool again.
        session._handle_event(complete)
        await asyncio.sleep(0.01)
        assert [call.call_id for call in calls] == ["a"]
        assert len([event for event in ws.sent if event["type"] == "response.create"]) == 1
    finally:
        await session.aclose()
        await model.aclose()


async def test_replaced_response_inherits_open_calls_and_ignores_old_terminal_events(monkeypatch):
    ws = _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    try:
        await session._update_session()
        await session._session_started_fut
        session._handle_event(
            _response_event("d", {"type": "response.created", "response": {"id": "r1"}})
        )
        session._handle_event(_response_event("d", _function_call_done("slow")))
        session._handle_event(
            _response_event("d", {"type": "response.created", "response": {"id": "r2"}})
        )
        session._handle_event(_response_event("d", _function_call_done("fast")))
        assert session._delegated_responses["d"].call_ids == {"slow", "fast"}
        session._handle_event(
            _response_event("d", {"type": "response.failed", "response": {"id": "r1"}})
        )
        session._handle_event(_response_event("d", _completed("r1")))
        assert not session._delegated_responses["d"].completed
        session._handle_event(_response_event("d", _completed("r2")))
        session._handle_event(
            _response_event("d", {"type": "response.created", "response": {"id": "r2"}})
        )
        assert session._delegated_responses["d"].completed  # duplicate create cannot reset barrier
        await session._append_items(
            [llm.FunctionCallOutput(call_id="fast", output="two", is_error=False)]
        )
        await asyncio.sleep(0.01)
        assert not [e for e in ws.sent if e["type"] == "response.create"]
        await session._append_items(
            [llm.FunctionCallOutput(call_id="slow", output="one", is_error=False)]
        )
        await asyncio.sleep(0.01)
        assert len([e for e in ws.sent if e["type"] == "response.create"]) == 1
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.parametrize("terminal", ["response.failed", "response.incomplete"])
async def test_failed_blocker_releases_one_ready_continuation(monkeypatch, terminal):
    ws = _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    try:
        await session._update_session()
        await session._session_started_fut
        for d in ("ready", "blocked"):
            session._handle_event(_response_event(d, {"type": "response.created"}))
            session._handle_event(_response_event(d, _function_call_done(d)))
        session._handle_event(_response_event("ready", _completed("r")))
        await session._append_items(
            [llm.FunctionCallOutput(call_id="ready", output="done", is_error=False)]
        )
        await asyncio.sleep(0.01)
        assert not [e for e in ws.sent if e["type"] == "response.create"]
        session._handle_event(_response_event("blocked", {"type": terminal}))
        session._handle_event(_response_event("blocked", {"type": terminal}))
        await asyncio.sleep(0.01)
        assert len([e for e in ws.sent if e["type"] == "response.create"]) == 1
        assert not session._delegated_responses
        assert not session._fnc_call_to_delegation
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.parametrize(
    "invalid",
    [
        8,
        10,
        0,
        -1,
        True,
        False,
        None,
        "12",
        float("nan"),
        float("inf"),
        float("-inf"),
        10**400,
        -(10**400),
    ],
)
async def test_usage_watermark_never_decreases_and_malformed_close_always_acknowledges(
    monkeypatch, invalid
):
    _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    metrics = []
    session.on("metrics_collected", metrics.append)
    try:
        await session._update_session()
        await session._session_started_fut
        for value in (10, invalid, 12, 12):
            event: dict[str, Any] = {"type": "session.usage.updated", "usage": {"seconds": value}}
            session._handle_event(event)
            assert event["usage"]["seconds"] is value  # input payload is not rewritten
        assert [m.session_duration for m in metrics] == [10, 2]
        session._handle_event({"type": "session.closed", "usage": {"seconds": invalid}})
        assert session._session_closed_fut.done()
        assert [m.session_duration for m in metrics] == [10, 2]
    finally:
        await session.aclose()
        await model.aclose()

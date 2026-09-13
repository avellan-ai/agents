from __future__ import annotations

import asyncio

import pytest

from livekit.agents import llm
from livekit.plugins.openai.realtime import GPTLiveModel

from .test_gpt_live_model import _connect_hook, _pcm, _silence, _transcript

pytestmark = pytest.mark.unit


async def test_external_final_preserves_complete_input_and_early_delegation(monkeypatch):
    ws = _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test", input_transcription="external")
    session = model.session()
    events = []
    starts = []
    session.on("input_audio_transcription_completed", events.append)
    session.on("input_speech_started", starts.append)
    try:
        await session._update_session()
        await session._session_started_fut
        assert session.input_transcription == "external"
        assert model.capabilities.user_transcription
        assert "input_transcription" not in ws.sent[0]["session"]
        session.begin_input_transcription(
            item_id="original", turn_started_at=123.5, connection_epoch=0
        )
        session.update_input_transcription(
            item_id="original", transcript="Remember number", is_final=False, connection_epoch=0
        )
        session._handle_event(_transcript("user", "Remember number", 0))
        session._handle_event(
            {
                "type": "session.delegation.created",
                "delegation": {"id": "d1", "target": "responses"},
            }
        )
        assert session._delegated_responses["d1"].input_message_id == "original"
        for _ in range(20):
            session.push_audio(_silence(100))
        session.push_audio(_pcm(0.2))
        await asyncio.sleep(0.01)
        assert session._input_vad is None
        assert any(e["type"] == "session.input_audio.append" for e in ws.sent)
        session._handle_event(_transcript("user", " forty-three", 5000))
        assert len(starts) == 1
        assert not any(e.is_final for e in events)
        session.update_input_transcription(
            item_id="original",
            transcript="Remember number forty-three",
            is_final=True,
            connection_epoch=0,
        )
        # Late native fragments cannot allocate another canonical message or duplicate finality.
        session._handle_event(_transcript("user", ".", 9000))
        final = [e for e in events if e.is_final]
        assert len(final) == 1
        assert (final[0].item_id, final[0].transcript, final[0].turn_started_at) == (
            "original",
            "Remember number forty-three",
            123.5,
        )
        messages = [i for i in session._history.items if isinstance(i, llm.ChatMessage)]
        assert [(m.id, m.text_content, m.created_at) for m in messages] == [
            ("original", "Remember number forty-three", 123.5)
        ]
        session._handle_event(
            {
                "type": "session.delegation.created",
                "delegation": {"id": "d2", "target": "responses"},
            }
        )
        assert session._delegated_responses["d2"].input_message_id == "original"
        with pytest.raises(llm.RealtimeError, match="not active"):
            session.update_input_transcription(
                item_id="original", transcript="duplicate", is_final=True, connection_epoch=0
            )
        with pytest.raises(llm.RealtimeError, match="already been used"):
            session.begin_input_transcription(
                item_id="original", turn_started_at=123.5, connection_epoch=0
            )
    finally:
        await session.aclose()
        await model.aclose()


async def test_transcription_failure_never_promotes_partial_or_native_text(monkeypatch):
    _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test", input_transcription="external")
    session = model.session()
    events = []
    session.on("input_audio_transcription_completed", events.append)
    try:
        await session._update_session()
        await session._session_started_fut
        session.begin_input_transcription(item_id="failed", turn_started_at=123, connection_epoch=0)
        session.update_input_transcription(
            item_id="failed", transcript="I accept", is_final=False, connection_epoch=0
        )
        session.abort_input_transcription(item_id="failed", connection_epoch=0)
        session._handle_event(_transcript("user", "I accept the cost", 0))
        assert not any(e.is_final for e in events)
        assert session._history.get_by_id("failed") is None
        assert session._last_final_input_id is None
        with pytest.raises(llm.RealtimeError):
            session.update_input_transcription(
                item_id="failed", transcript="I accept", is_final=True, connection_epoch=0
            )
        with pytest.raises(llm.RealtimeError):
            session.begin_input_transcription(
                item_id="failed", turn_started_at=123, connection_epoch=0
            )
    finally:
        await session.aclose()
        await model.aclose()


async def test_retired_connection_cannot_finalize_input(monkeypatch):
    _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test", input_transcription="external")
    session = model.session()
    events = []
    session.on("input_audio_transcription_completed", events.append)
    try:
        await session._update_session()
        await session._session_started_fut
        session.begin_input_transcription(
            item_id="retired", turn_started_at=123, connection_epoch=0
        )
        session.update_input_transcription(
            item_id="retired", transcript="partial", is_final=False, connection_epoch=0
        )
        await session._reset_for_reconnect()
        assert not any(e.is_final for e in events)
        assert session._history.get_by_id("retired") is None
        with pytest.raises(llm.RealtimeError, match="retired connection"):
            session.update_input_transcription(
                item_id="retired", transcript="late final", is_final=True, connection_epoch=0
            )
        with pytest.raises(llm.RealtimeError, match="already been used"):
            session.begin_input_transcription(
                item_id="retired", turn_started_at=123, connection_epoch=1
            )
    finally:
        session._handle_event({"type": "session.started"})
        await session.aclose()
        await model.aclose()
    with pytest.raises(llm.RealtimeError):
        session.begin_input_transcription(item_id="closed", turn_started_at=123, connection_epoch=1)


async def test_external_input_is_opt_in(monkeypatch):
    _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    try:
        with pytest.raises(llm.RealtimeError, match="not enabled"):
            session.begin_input_transcription(
                item_id="external", turn_started_at=123, connection_epoch=0
            )
        assert not session._history.items
    finally:
        await session.aclose()
        await model.aclose()


async def test_abort_retires_only_bound_delegations_and_late_tool_outputs(monkeypatch):
    from .test_gpt_live_model import _function_call_done, _response_event

    ws = _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test", input_transcription="external")
    session = model.session()
    calls = []
    session.on("function_call", calls.append)
    try:
        await session._update_session()
        await session._session_started_fut
        # Unrelated backend work must retain its identity and routing.
        session._handle_event(_response_event("unrelated", {"type": "response.created"}))
        session._handle_event(_response_event("unrelated", _function_call_done("keep")))
        session.begin_input_transcription(item_id="failed", turn_started_at=123, connection_epoch=0)
        session._handle_event(
            {
                "type": "session.delegation.created",
                "delegation": {"id": "aborted", "target": "responses"},
            }
        )
        session._handle_event(_response_event("aborted", {"type": "response.created"}))
        session._handle_event(_response_event("aborted", _function_call_done("already-emitted")))
        session.abort_input_transcription(item_id="failed", connection_epoch=0)
        assert "unrelated" in session._delegated_responses
        assert session._fnc_call_to_delegation == {"keep": "unrelated"}
        assert "aborted" not in session._delegated_responses
        assert "aborted" not in session._delegation_input_ids
        before = len(calls)
        session._handle_event(_response_event("aborted", _function_call_done("late-call")))
        session._handle_event(_response_event("aborted", {"type": "response.completed"}))
        # Neither duplicate delegation creation nor a continuation may resurrect it.
        session._handle_event(
            {
                "type": "session.delegation.created",
                "delegation": {"id": "aborted", "target": "responses"},
            }
        )
        session._handle_event(_response_event("aborted", {"type": "response.created"}))
        session._handle_event(_response_event("aborted", _function_call_done("continuation-call")))
        await session._append_items(
            [
                llm.FunctionCallOutput(
                    call_id="already-emitted", output="late result", is_error=False
                )
            ]
        )
        assert len(calls) == before
        assert "aborted" not in session._delegated_responses
        assert not any(
            isinstance(item, (llm.FunctionCall, llm.FunctionCallOutput))
            and item.call_id == "already-emitted"
            for item in session._history.items
        )
        await session._append_items(
            [llm.FunctionCallOutput(call_id="keep", output="valid result", is_error=False)]
        )
        await asyncio.sleep(0.01)
        assert any(e.get("item", {}).get("call_id") == "keep" for e in ws.sent)
        assert not any(e.get("item", {}).get("call_id") == "already-emitted" for e in ws.sent)
        assert not any(e["type"] == "response.create" for e in ws.sent)
    finally:
        await session.aclose()
        await model.aclose()

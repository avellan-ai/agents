from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest
from google.genai import types

from livekit.agents import llm
from livekit.plugins.google.realtime.realtime_api import RealtimeModel, RealtimeSession


def _create_session(model_name: str) -> RealtimeSession:
    model = RealtimeModel(model=model_name, api_key="test-key")
    session = RealtimeSession.__new__(RealtimeSession)
    llm.RealtimeSession.__init__(session, model)
    session._opts = replace(model._opts)
    session._tools = llm.ToolContext.empty()
    session._chat_ctx = llm.ChatContext.empty()
    session._input_resampler = None
    session._current_generation = None
    session._active_session = object()
    session._session_should_close = asyncio.Event()
    session._response_created_futures = {}
    session._pending_generation_fut = None
    session._session_resumption_handle = None
    session._in_user_activity = False
    session._session_lock = asyncio.Lock()
    session._num_retries = 0
    return session


def test_gemini31_generate_reply_uses_realtime_text_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _create_session("gemini-3.1-flash-live-preview")
    sent_events: list[Any] = []
    monkeypatch.setattr(session, "_send_client_event", sent_events.append)

    fut = session.generate_reply(instructions="Greet the visitor.")
    fut.cancel()

    assert len(sent_events) == 1
    event = sent_events[0]
    assert isinstance(event, types.LiveClientRealtimeInput)
    assert event.text is not None
    assert "Greet the visitor." in event.text


def test_gemini25_generate_reply_uses_client_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _create_session("gemini-2.5-flash-native-audio-preview-12-2025")
    sent_events: list[Any] = []
    monkeypatch.setattr(session, "_send_client_event", sent_events.append)

    fut = session.generate_reply(instructions="Greet the visitor.")
    fut.cancel()

    assert len(sent_events) == 1
    assert isinstance(sent_events[0], types.LiveClientContent)


async def test_gemini31_update_instructions_uses_realtime_text_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _create_session("gemini-3.1-flash-live-preview")
    sent_events: list[Any] = []
    monkeypatch.setattr(session, "_send_client_event", sent_events.append)

    await session.update_instructions("The visitor is looking at the pricing page.")

    assert len(sent_events) == 1
    event = sent_events[0]
    assert isinstance(event, types.LiveClientRealtimeInput)
    assert event.text is not None
    assert "Application context update" in event.text
    assert "pricing page" in event.text


async def test_gemini31_update_chat_ctx_uses_realtime_text_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _create_session("gemini-3.1-flash-live-preview")
    sent_events: list[Any] = []
    monkeypatch.setattr(session, "_send_client_event", sent_events.append)
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="assistant", content="The current page is Services.")

    await session.update_chat_ctx(chat_ctx)

    assert len(sent_events) == 1
    event = sent_events[0]
    assert isinstance(event, types.LiveClientRealtimeInput)
    assert event.text is not None
    assert "Application context update" in event.text
    assert "model: The current page is Services." in event.text


def test_gemini31_connect_config_enables_initial_history() -> None:
    session = _create_session("gemini-3.1-flash-live-preview")

    config = session._build_connect_config()

    assert config.history_config is not None
    assert config.history_config.initial_history_in_client_content is True

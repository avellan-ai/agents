from __future__ import annotations

import asyncio
import io
import wave
from typing import Any

import pytest
from hume.empathic_voice.types import (
    AssistantProsody,
    Inference,
    ToolResponseMessage,
    UserInput,
    WebSocketError,
)

from livekit.agents import APIError, llm
from livekit.plugins.hume.realtime.audio import StreamingWavDecoder
from livekit.plugins.hume.realtime.realtime_model import (
    RealtimeModel,
    RealtimeSession,
    _parse_tool_call_arguments,
)


def _build_wav_blob(
    *, sample_rate: int = 16000, channels: int = 1, samples_per_channel: int = 320
) -> tuple[bytes, bytes, bytes]:
    pcm = b"\x01\x00" * samples_per_channel * channels
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wavf:
        wavf.setnchannels(channels)
        wavf.setsampwidth(2)
        wavf.setframerate(sample_rate)
        wavf.writeframes(pcm)

    wav_bytes = wav_buffer.getvalue()
    return wav_bytes[:44], wav_bytes[44:], pcm


def test_streaming_wav_decoder_decodes_header_then_raw_pcm() -> None:
    decoder = StreamingWavDecoder()
    header, body, pcm = _build_wav_blob()

    assert decoder.push(header) == []

    frames = decoder.push(body)
    assert len(frames) == 1
    assert frames[0].sample_rate == 16000
    assert frames[0].num_channels == 1
    assert bytes(frames[0].data) == pcm


def test_streaming_wav_decoder_resets_when_new_wav_header_appears() -> None:
    decoder = StreamingWavDecoder()
    header, body, pcm = _build_wav_blob()

    assert len(decoder.push(header)) == 0
    first_frames = decoder.push(body)
    assert len(first_frames) == 1
    assert bytes(first_frames[0].data) == pcm

    assert len(decoder.push(header)) == 0
    second_frames = decoder.push(body)
    assert len(second_frames) == 1
    assert bytes(second_frames[0].data) == pcm


def test_parse_tool_call_arguments_normalizes_json_or_preserves_raw() -> None:
    assert _parse_tool_call_arguments('{"z":2,"a":1}') == '{"a":1,"z":2}'
    assert _parse_tool_call_arguments("not-json") == "not-json"


async def _idle_main_loop(_: RealtimeSession) -> None:
    await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_generate_reply_flushes_staged_events_after_future_is_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RealtimeSession, "_main_loop", _idle_main_loop)

    session = RealtimeModel(api_key="test-key").session()
    observed: list[tuple[object, int]] = []

    def _record_enqueue(event: Any) -> None:
        observed.append((event, len(session._pending_generation_futs)))

    monkeypatch.setattr(session, "_enqueue_publish", _record_enqueue)

    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hello", id="user_item")
    chat_ctx.insert(
        llm.FunctionCallOutput(
            call_id="call-1",
            name="lookup",
            output='{"ok":true}',
            is_error=False,
        )
    )

    await session.update_chat_ctx(chat_ctx)
    assert len(session._staged_publish_events) == 2

    generation_fut = session.generate_reply()
    assert not generation_fut.done()
    assert len(session._pending_generation_futs) == 1
    assert len(session._staged_publish_events) == 0

    assert len(observed) == 2
    assert all(pending_count == 1 for _, pending_count in observed)
    observed_types = {type(event) for event, _ in observed}
    assert UserInput in observed_types
    assert ToolResponseMessage in observed_types

    await session.aclose()


@pytest.mark.asyncio
async def test_subscribe_event_handles_websocket_error_and_ignores_prosody(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RealtimeSession, "_main_loop", _idle_main_loop)

    session = RealtimeModel(api_key="test-key").session()
    captured: list[tuple[Exception, bool]] = []

    def _record_error(*, error: Exception, recoverable: bool) -> None:
        captured.append((error, recoverable))

    monkeypatch.setattr(session, "_emit_error", _record_error)

    session._handle_subscribe_event(
        WebSocketError(code="invalid_request", message="bad request", slug="invalid_request")
    )
    assert len(captured) == 1
    assert isinstance(captured[0][0], APIError)
    assert captured[0][1] is True

    captured.clear()
    session._handle_subscribe_event(AssistantProsody(models=Inference()))
    assert captured == []

    await session.aclose()

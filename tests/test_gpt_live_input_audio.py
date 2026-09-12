from __future__ import annotations

import asyncio
import base64
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from livekit import rtc
from livekit.agents import APIConnectOptions, utils, vad
from livekit.plugins.openai.realtime import GPTLiveModel, GPTLiveSession, gpt_live_model

from .test_gpt_live_model import _connect_hook, _LifecycleWS, _pcm, _silence, _transcript

pytestmark = pytest.mark.unit


class FakeDetector:
    def __init__(self):
        self.events = utils.aio.Chan[vad.VADEvent]()
        self.frames = []
        self.closed = False

    def push_frame(self, frame):
        self.frames.append(frame)
        self.events.send_nowait(
            vad.VADEvent(
                type=vad.VADEventType.INFERENCE_DONE,
                samples_index=0,
                timestamp=0,
                speech_duration=0,
                silence_duration=0,
                probability=0.9 if any(abs(sample) > 400 for sample in frame.data) else 0.0,
                frames=[frame],
            )
        )

    def __aiter__(self):
        return self.events.__aiter__()

    async def aclose(self):
        self.closed = True
        self.events.close()


def fake_detector(monkeypatch):
    monkeypatch.setattr(
        gpt_live_model.inference, "VAD", lambda **kwargs: SimpleNamespace(stream=FakeDetector)
    )


async def detected(session, *, pending_ms=0):
    async def wait():
        while session._input_vad_ms + pending_ms + 1e-6 < session._input_audio_ms:
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), 2)


@pytest.mark.parametrize("pause_before", [False, True])
async def test_speech_without_transcript_updates_does_not_end_a_turn(monkeypatch, pause_before):
    _connect_hook(monkeypatch)
    fake_detector(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    finals = []
    session.on(
        "input_audio_transcription_completed", lambda e: finals.append(e) if e.is_final else None
    )
    try:
        await session._update_session()
        await session._session_started_fut
        # Already-queued silence must not count after a new transcript arrives.
        for _ in range(20):
            session.push_audio(_silence(100))
        session._handle_event(_transcript("user", "Still talking", 2000))
        if pause_before:
            for _ in range(5):
                session.push_audio(_silence(100))
        for _ in range(15):
            session.push_audio(_pcm(0.2))
        await detected(session)
        assert not finals
        for _ in range(7):
            session.push_audio(_silence(100))
        await detected(session)
        assert not finals
        session.push_audio(_silence(100))
        await detected(session)
        assert [e.transcript for e in finals] == ["Still talking"]
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.virtual_time
async def test_idle_clock_waits_for_start_and_stops_on_close(monkeypatch):
    ws = _connect_hook(monkeypatch, auto_start=False)
    fake_detector(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    try:
        await session._update_session()
        await asyncio.sleep(0.5)
        assert [e["type"] for e in ws.sent] == ["session.start"]
        session._handle_event({"type": "session.started"})
        session.append_commentary("A typed answer can now be spoken")
        await asyncio.sleep(0.8)
        audio = [e for e in ws.sent if e["type"] == "session.input_audio.append"]
        assert len(audio) >= 5
        assert all(base64.b64decode(e["audio"]) == bytes(4800) for e in audio)
        session._handle_event({"type": "session.closed"})
        before = len(ws.sent)
        await asyncio.sleep(0.5)
        assert len(ws.sent) == before
    finally:
        await session.aclose()
        await model.aclose()
    assert session._input_vad is None
    assert session._input_vad_task is None


@pytest.mark.virtual_time
async def test_clock_never_pads_active_or_burst_microphone_audio(monkeypatch):
    ws = _connect_hook(monkeypatch)
    fake_detector(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    try:
        await session._update_session()
        await session._session_started_fut
        for _ in range(10):
            session.push_audio(_pcm(0.2))
            await asyncio.sleep(0.1)
        audio = [e for e in ws.sent if e["type"] == "session.input_audio.append"]
        assert len(audio) == 10
        assert all(any(base64.b64decode(e["audio"])) for e in audio)
        for _ in range(10):
            session.push_audio(_pcm(0.2))
        await asyncio.sleep(0.5)
        assert len([e for e in ws.sent if e["type"] == "session.input_audio.append"]) == 20
        await asyncio.sleep(0.8)
        assert any(
            not any(base64.b64decode(e["audio"]))
            for e in ws.sent
            if e["type"] == "session.input_audio.append"
        )
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.parametrize("sample_rate", [24000, 48000])
async def test_mute_flushes_partial_audio_and_uses_silence_for_detection(monkeypatch, sample_rate):
    ws = _connect_hook(monkeypatch)
    fake_detector(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()

    def frame(ms):
        samples = sample_rate * ms // 1000
        return rtc.AudioFrame(
            np.full(samples, 6000, dtype=np.int16).tobytes(), sample_rate, 1, samples
        )

    try:
        await session._update_session()
        await session._session_started_fut
        session._handle_event(_transcript("user", "Before mute", 0))
        session.push_audio(frame(60))
        session.mute_input()
        stream = session._input_vad
        pre_mute = len(stream.frames)
        for _ in range(10):
            session.push_audio(frame(100))
        await detected(session)
        assert "user" not in session._speech
        assert all(not any(f.data) for f in stream.frames[pre_mute:])
        mute_index = next(
            i for i, e in enumerate(ws.sent) if e["type"] == "session.input_audio.mute"
        )
        assert sum(len(base64.b64decode(e["audio"])) for e in ws.sent[1:mute_index]) == 2880
        session.unmute_input()
        session._handle_event(_transcript("user", "After unmute", 1000))
        for _ in range(15):
            session.push_audio(frame(100))
        await detected(session)
        assert "user" in session._speech
    finally:
        await session.aclose()
        await model.aclose()
    assert stream.closed


@pytest.mark.parametrize("noise_rms", [0.002, 0.02])
async def test_native_vad_finalizes_nonzero_microphone_noise(monkeypatch, noise_rms):
    _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    try:
        await session._update_session()
        await session._session_started_fut
        session._handle_event(_transcript("user", "Finished speaking", 0))
        rng = np.random.default_rng(42)
        for _ in range(20):
            samples = np.clip(rng.normal(0, noise_rms * 32768, 2400), -32768, 32767).astype(
                np.int16
            )
            session.push_audio(rtc.AudioFrame(samples.tobytes(), 24000, 1, 2400))
        await detected(session, pending_ms=32)
        assert "user" not in session._speech
    finally:
        await session.aclose()
        await model.aclose()


async def test_native_vad_keeps_real_recorded_speech_open(monkeypatch):
    _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    with wave.open(
        str(Path(__file__).parent / "test_realtime/weather_question.wav"), "rb"
    ) as audio:
        data = audio.readframes(audio.getnframes())
    frames = [
        rtc.AudioFrame(data[i : i + 4800], 24000, 1, len(data[i : i + 4800]) // 2)
        for i in range(0, len(data), 4800)
    ]
    try:
        await session._update_session()
        await session._session_started_fut
        started = False
        for frame in frames * 3:
            session.push_audio(frame)
            await detected(session, pending_ms=32)
            if session._input_speaking and not started:
                session._handle_event(_transcript("user", "What is the weather?", 0))
                started = True
        assert started and "user" in session._speech
        for _ in range(20):
            session.push_audio(_silence(100))
        await detected(session, pending_ms=32)
        assert "user" not in session._speech
    finally:
        await session.aclose()
        await model.aclose()


async def test_reconnect_settles_vad_and_restores_mute_before_audio(monkeypatch):
    sockets = [_LifecycleWS(), _LifecycleWS()]
    connections = iter(sockets)

    async def connect(self):
        return next(connections)

    monkeypatch.setattr(GPTLiveSession, "_create_ws_conn", connect)
    fake_detector(monkeypatch)
    model = GPTLiveModel(
        api_key="test", conn_options=APIConnectOptions(max_retry=1, retry_interval=0)
    )
    session = model.session()
    try:
        await session._update_session()
        await sockets[0].started.wait()
        await session._session_started_fut
        session.mute_input()
        session.push_audio(_pcm(0.2))
        await detected(session)
        old_stream, old_task = session._input_vad, session._input_vad_task
        await sockets[0].close()
        await asyncio.wait_for(sockets[1].started.wait(), 1)
        await session._session_started_fut
        assert old_stream.closed and old_task.done()
        session.push_audio(_pcm(0.2))
        await detected(session)
        assert [e["type"] for e in sockets[1].sent[:3]] == [
            "session.start",
            "session.input_audio.mute",
            "session.input_audio.append",
        ]
        assert not any(session._input_vad.frames[-1].data)
    finally:
        for ws in sockets:
            ws.emit({"type": "session.closed"})
        await session.aclose()
        await model.aclose()


async def test_microphone_cannot_recreate_detector_during_reconnect_cleanup(monkeypatch):
    _connect_hook(monkeypatch)
    fake_detector(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    closing = asyncio.Event()
    release = asyncio.Event()
    try:
        await session._update_session()
        await session._session_started_fut
        session.push_audio(_pcm(0.2))
        await detected(session)
        session.push_audio(rtc.AudioFrame(bytes(4800 * 2), 48000, 1, 4800))
        old = session._input_vad
        original_close = old.aclose

        async def delayed_close():
            closing.set()
            await release.wait()
            await original_close()

        monkeypatch.setattr(old, "aclose", delayed_close)
        cleanup = asyncio.create_task(session._close_input_vad())
        await closing.wait()
        session.push_audio(_pcm(0.2))
        session.mute_input()
        session.unmute_input()
        assert session._input_vad is None
        release.set()
        await cleanup
        assert old.closed
        session.push_audio(_pcm(0.2))
        await detected(session)
        assert session._input_vad is not old
    finally:
        release.set()
        await session.aclose()
        await model.aclose()


@pytest.mark.virtual_time
async def test_initial_audio_and_mute_keep_queue_order_and_duration(monkeypatch):
    ws = _connect_hook(monkeypatch)
    fake_detector(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    try:
        for _ in range(10):
            session.push_audio(_pcm(0.2))
        session.mute_input()
        # A delayed handshake must not erase the duration of audio waiting to be sent.
        await asyncio.sleep(1.5)
        await session._update_session()
        await session._session_started_fut
        await asyncio.sleep(0.5)
        assert [e["type"] for e in ws.sent] == ["session.start"] + [
            "session.input_audio.append"
        ] * 10 + ["session.input_audio.mute"]
        await asyncio.sleep(0.7)
        assert any(
            not any(base64.b64decode(e["audio"]))
            for e in ws.sent
            if e["type"] == "session.input_audio.append"
        )
    finally:
        await session.aclose()
        await model.aclose()

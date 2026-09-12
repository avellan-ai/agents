"""Interrupted history follows audio playout, independently of caption pacing."""

from __future__ import annotations

import asyncio
import time

import pytest

from livekit.agents.types import TimedString
from livekit.agents.voice.transcription.synchronizer import _SegmentSynchronizerImpl

from .test_transcript_sync_timed import _CollectorTextOutput, _opts, _silent_frames

pytestmark = pytest.mark.unit


async def test_interruption_uses_audio_position_when_captions_run_ahead() -> None:
    captions = _CollectorTextOutput()
    impl = _SegmentSynchronizerImpl(_opts(), next_in_chain=captions)
    try:
        for frame in _silent_frames(0.5):
            impl.push_audio(frame)
        impl.push_text(TimedString("Heard.", start_time=0.0, end_time=0.1))
        impl.push_text(TimedString(" Unplayed.", start_time=0.4, end_time=0.5))
        impl.end_audio_input()
        impl.end_text_input()
        impl.on_playback_started(time.time() - 1.0)
        # The sink's actual playout position can lag caption wall-clock pacing.
        async with asyncio.timeout(2):
            while "".join(captions.words) != "Heard. Unplayed.":
                await asyncio.sleep(0.01)
        impl.mark_playback_finished(playback_position=0.15, interrupted=True)
        assert impl.synchronized_transcript == "Heard."
        await impl.aclose()
        assert impl.synchronized_transcript == "Heard."
    finally:
        await impl.aclose()


async def test_played_timed_text_does_not_depend_on_caption_delivery() -> None:
    class SlowCaptions(_CollectorTextOutput):
        def __init__(self):
            super().__init__()
            self.release = asyncio.Event()

        async def capture_text(self, text: str) -> None:
            await self.release.wait()
            await super().capture_text(text)

    captions = SlowCaptions()
    impl = _SegmentSynchronizerImpl(_opts(), next_in_chain=captions)
    try:
        for frame in _silent_frames(0.5):
            impl.push_audio(frame)
        impl.push_text(TimedString("Heard.", start_time=0.0, end_time=0.1))
        impl.on_playback_started(time.time() - 0.2)
        impl.mark_playback_finished(playback_position=0.15, interrupted=True)
        assert captions.words == []
        assert impl.synchronized_transcript == "Heard."
        impl.push_text(TimedString(" Later.", start_time=0.3, end_time=0.4))
        assert impl.synchronized_transcript == "Heard."
    finally:
        captions.release.set()
        await impl.aclose()


@pytest.mark.parametrize(
    "first, expected",
    [
        (TimedString("Unknown.", start_time=0.0), ""),
        (TimedString("Partial.", start_time=0.0, end_time=0.2), ""),
        (TimedString("Negative.", start_time=-0.1, end_time=-0.05), ""),
        (TimedString("Reversed.", start_time=0.2, end_time=0.1), ""),
        (TimedString("Not finite.", start_time=0.0, end_time=float("nan")), ""),
        (TimedString("Heard.", start_time=0.0, end_time=0.1), "Heard."),
    ],
)
async def test_interrupted_prefix_stops_at_unknown_or_unplayed_span(first, expected) -> None:
    impl = _SegmentSynchronizerImpl(_opts(), next_in_chain=None)
    try:
        for frame in _silent_frames(0.5):
            impl.push_audio(frame)
        impl.push_text(first)
        impl.push_text(TimedString(" Later.", start_time=0.1, end_time=0.15))
        impl.mark_playback_finished(playback_position=0.1, interrupted=True)
        assert impl.synchronized_transcript == expected
    finally:
        await impl.aclose()


async def test_closing_during_caption_delay_does_not_emit_unplayed_word() -> None:
    captions = _CollectorTextOutput()
    impl = _SegmentSynchronizerImpl(_opts(), next_in_chain=captions)
    try:
        for frame in _silent_frames(0.5):
            impl.push_audio(frame)
        impl.push_text(TimedString("Unplayed.", start_time=2.0, end_time=3.0))
        impl.on_playback_started(time.time())
        await asyncio.sleep(0.01)
        assert captions.words == []
        impl.mark_playback_finished(playback_position=0.01, interrupted=True)
        await impl.aclose()
        assert captions.words == []
        assert impl.synchronized_transcript == ""
    finally:
        await impl.aclose()


async def test_exact_audio_endpoint_preserves_completed_span() -> None:
    impl = _SegmentSynchronizerImpl(_opts(), next_in_chain=None)
    try:
        for frame in _silent_frames(0.1):
            impl.push_audio(frame)
        impl.push_text(TimedString("Heard.", start_time=0.0, end_time=0.1))
        impl.mark_playback_finished(playback_position=0.1, interrupted=True)
        assert impl.synchronized_transcript == "Heard."
    finally:
        await impl.aclose()

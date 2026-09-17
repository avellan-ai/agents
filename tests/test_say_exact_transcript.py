"""Fully delivered controlled speech retains exact text despite alignment padding."""

import asyncio

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT, TimedString

from .fake_io import FakeAudioOutput, FakeTextOutput
from .fake_tts import FakeTTS

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


@pytest.mark.parametrize(
    "aligned, expected",
    [
        ("The cost is one rope. ", "The cost is one rope."),
        ("The cost is two ropes. ", "The cost is two ropes. "),
    ],
)
async def test_completed_say_preserves_words_and_removes_only_alignment_padding(aligned, expected):
    class AlignedAgent(Agent):
        async def tts_node(self, text, model_settings):
            assert "".join([chunk async for chunk in text]) == "The cost is one rope."
            frame = rtc.AudioFrame(bytes(2400), 24000, 1, 1200)
            frame.userdata[USERDATA_TIMED_TRANSCRIPT] = [
                TimedString(aligned, start_time=0.0, end_time=0.05)
            ]
            yield frame

    tts = FakeTTS()
    tts._capabilities.aligned_transcript = True
    async with AgentSession(tts=tts, use_tts_aligned_transcript=True) as session:
        session.output.audio = FakeAudioOutput()
        session.output.transcription = FakeTextOutput()
        await session.start(AlignedAgent(instructions="test"))
        handle = session.say("The cost is one rope.")
        await asyncio.wait_for(handle, 3)
        assert not handle.interrupted
        assert [item.text_content for item in handle.chat_items] == [expected]


async def test_interrupted_say_keeps_the_played_prefix_instead_of_controlled_input():
    class InterruptedOutput(FakeAudioOutput):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()

        async def capture_frame(self, frame):
            await super().capture_frame(frame)
            self.started.set()

        def clear_buffer(self):
            if self._pushed_duration:
                self._reset_playout()
                self.on_playback_finished(
                    playback_position=0.1, interrupted=True, synchronized_transcript="The cost "
                )

    async def audio():
        yield rtc.AudioFrame(bytes(48000), 24000, 1, 24000)

    output = InterruptedOutput()
    async with AgentSession() as session:
        session.output.audio = output
        session.output.transcription = FakeTextOutput()
        await session.start(Agent(instructions="test"))
        handle = session.say("The cost is one rope.", audio=audio())
        await asyncio.wait_for(output.started.wait(), 2)
        handle.interrupt(force=True)
        await asyncio.wait_for(handle, 2)
        assert handle.interrupted
        assert [item.text_content for item in handle.chat_items] == ["The cost "]

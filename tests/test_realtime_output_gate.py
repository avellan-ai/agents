"""A native reply's audio and transcript must cross the same pre-playout gate."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, llm, utils

from .fake_io import FakeAudioOutput, FakeTextOutput
from .fake_realtime import FakeRealtimeModel, _audio_frame, fake_capabilities

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class GatedAgent(Agent):
    def __init__(self, outcome="allow"):
        super().__init__(instructions="test")
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.outcome = outcome

    async def realtime_output_node(self, message, model_settings):
        self.entered.set()
        await self.release.wait()
        if self.outcome == "error":
            raise RuntimeError("review unavailable")
        return message if self.outcome == "allow" else None


@asynccontextmanager
async def native_reply(agent):
    model = FakeRealtimeModel(capabilities=fake_capabilities())
    audio_output, text_output = FakeAudioOutput(), FakeTextOutput()
    async with AgentSession(llm=model) as session:
        session.output.audio, session.output.transcription = audio_output, text_output
        await session.start(agent)
        handle = session.generate_reply()
        while not model.active_session._reply_futs:
            await asyncio.sleep(0)
        message_ch = utils.aio.Chan[llm.MessageGeneration]()
        function_ch = utils.aio.Chan[llm.FunctionCall]()
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        modalities = asyncio.Future()
        modalities.set_result(["audio", "text"])
        message_ch.send_nowait(
            llm.MessageGeneration(
                message_id="native-message",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        text_ch.send_nowait("The door is open.")
        audio_ch.send_nowait(_audio_frame(0.5))
        for channel in (text_ch, audio_ch, message_ch, function_ch):
            channel.close()
        model.active_session._reply_futs[0].set_result(
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
                response_id="native-response",
            )
        )
        yield session, handle, audio_output, text_output


async def test_native_audio_and_text_wait_for_the_same_gate():
    agent = GatedAgent()
    async with native_reply(agent) as (_, handle, audio, text):
        await asyncio.wait_for(agent.entered.wait(), 2)
        await asyncio.sleep(0.1)
        assert audio._started_at is None
        assert text._pushed_text == ""
        assert text._messages == []
        agent.release.set()
        await asyncio.wait_for(handle.wait_for_playout(), 2)
        assert any(item.text_content == "The door is open." for item in handle.chat_items)
        assert "The door is open." in text._messages


@pytest.mark.parametrize("outcome", ["reject", "error"])
async def test_rejected_or_failed_gate_never_publishes_native_output(outcome):
    agent = GatedAgent(outcome)
    async with native_reply(agent) as (session, handle, audio, text):
        await asyncio.wait_for(agent.entered.wait(), 2)
        agent.release.set()
        await asyncio.wait_for(handle.wait_for_playout(), 2)
        assert audio._started_at is None
        assert text._pushed_text == ""
        assert not any(item.text_content for item in handle.chat_items)
        assert not any(item.role == "assistant" for item in session.history.messages())


async def test_interruption_during_review_never_releases_the_buffered_reply():
    agent = GatedAgent()
    async with native_reply(agent) as (_, handle, audio, text):
        await asyncio.wait_for(agent.entered.wait(), 2)
        handle.interrupt(force=True)
        agent.release.set()
        await asyncio.wait_for(handle.wait_for_playout(), 2)
        assert audio._started_at is None
        assert text._pushed_text == ""
        assert not any(item.text_content for item in handle.chat_items)


async def test_default_agent_keeps_native_output():
    async with native_reply(Agent(instructions="test")) as (_, handle, _, text):
        await asyncio.wait_for(handle.wait_for_playout(), 2)
        assert "The door is open." in text._messages

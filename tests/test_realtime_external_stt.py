"""A manually gated realtime model without ASR must retain the external final turn."""

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, function_tool, llm, utils

from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .fake_stt import FakeSTT
from .test_realtime_adaptive_interruption import _end_of_turn_info

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


async def test_external_final_is_prepared_and_saved_once_before_native_reply():
    prepared = []
    entered, release = asyncio.Event(), asyncio.Event()

    class PreparedAgent(Agent):
        async def on_user_turn_completed(self, turn_ctx, new_message):
            prepared.append(new_message)
            turn_ctx.add_message(
                id="scene", role="user", content="The canonical scene is revision 43."
            )
            entered.set()
            await release.wait()

    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=False,
            user_transcription=False,
        )
    )
    async with AgentSession(
        llm=model, stt=FakeSTT(), turn_handling={"turn_detection": "manual"}
    ) as session:
        await session.start(PreparedAgent(instructions="test"))
        session._activity.on_end_of_turn(_end_of_turn_info("Do not start work. Recall my reward."))
        await asyncio.wait_for(entered.wait(), 1)
        assert not model.active_session._reply_futs
        release.set()
        while not model.active_session._reply_futs:
            await asyncio.sleep(0)
        messages = [m for m in session.history.messages() if m.role == "user"]
        assert len(messages) == 1
        assert messages[0].id == prepared[0].id
        assert messages[0].text_content == "Do not start work. Recall my reward."
        assert model.active_session.chat_ctx.get_by_id(messages[0].id) is not None
        assert (
            model.active_session.chat_ctx.get_by_id("scene").text_content
            == "The canonical scene is revision 43."
        )
        message_ch = utils.aio.Chan[llm.MessageGeneration]()
        function_ch = utils.aio.Chan[llm.FunctionCall]()
        message_ch.close()
        function_ch.close()
        model.active_session._reply_futs[0].set_result(
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
                response_id="prepared-native-response",
            )
        )


async def test_context_policy_covers_initial_typed_tool_and_explicit_updates():
    class ContextAgent(Agent):
        revision = 0

        async def realtime_context_node(self, chat_ctx):
            chat_ctx.items[:] = [item for item in chat_ctx.items if item.id != "policy"]
            chat_ctx.add_message(
                id="policy", role="user", content=f"Current revision {self.revision}"
            )
            return chat_ctx

        @function_tool
        async def lookup(self) -> str:
            """Retrieve updated campaign facts."""
            self.revision = 3
            return "Retrieved revision three"

    model = FakeRealtimeModel(
        capabilities=fake_capabilities(audio_output=False, auto_tool_reply_generation=False)
    )
    owner = ContextAgent(instructions="test")
    async with AgentSession(llm=model) as session:
        await session.start(owner)
        rt = model.active_session
        assert rt.chat_ctx.get_by_id("policy").text_content == "Current revision 0"
        owner.revision = 1
        await owner.update_chat_ctx(owner.chat_ctx.copy())
        assert rt.chat_ctx.get_by_id("policy").text_content == "Current revision 1"
        owner.revision = 2
        handle = session.generate_reply(
            user_input=llm.ChatMessage(id="original-typed-id", role="user", content=["Look it up."])
        )
        async with asyncio.timeout(3):
            while not rt._reply_futs:
                await asyncio.sleep(0)
        assert rt.chat_ctx.get_by_id("policy").text_content == "Current revision 2"
        assert rt.chat_ctx.get_by_id("original-typed-id") is not None

        def generation(with_tool):
            messages = utils.aio.Chan[llm.MessageGeneration]()
            functions = utils.aio.Chan[llm.FunctionCall]()
            if with_tool:
                functions.send_nowait(
                    llm.FunctionCall(call_id="lookup-1", name="lookup", arguments="{}")
                )
            messages.close()
            functions.close()
            return llm.GenerationCreatedEvent(
                message_stream=messages, function_stream=functions, user_initiated=True
            )

        rt._reply_futs[0].set_result(generation(True))
        async with asyncio.timeout(3):
            while len(rt._reply_futs) < 2:
                await asyncio.sleep(0.01)
        assert rt.chat_ctx.get_by_id("policy").text_content == "Current revision 3"
        rt._reply_futs[1].set_result(generation(False))
        await asyncio.wait_for(handle.wait_for_playout(), 3)
        assert sum(item.id == "original-typed-id" for item in session.history.items) == 1
        assert session.history.get_by_id("policy") is None


async def test_split_external_input_keeps_address_when_pending_reply_is_interrupted():
    prepared = []

    class PreparedAgent(Agent):
        async def on_user_turn_completed(self, turn_ctx, new_message):
            prepared.append(new_message)

    model = FakeRealtimeModel(
        capabilities=fake_capabilities(turn_detection=False, user_transcription=False)
    )
    question = "Explain how to tell a loose joint from a split brace. Advice only."
    async with AgentSession(
        llm=model, stt=FakeSTT(), turn_handling={"turn_detection": "manual"}
    ) as session:
        await session.start(
            PreparedAgent(instructions="Route addressed NPC questions to their agent.")
        )
        session._activity.on_end_of_turn(_end_of_turn_info("Mara,"))
        async with asyncio.timeout(3):
            while len(model.active_session._reply_futs) < 1:
                await asyncio.sleep(0)
        first_speech = session._activity._current_speech
        session._activity.on_end_of_turn(_end_of_turn_info(question))
        async with asyncio.timeout(3):
            while len(model.active_session._reply_futs) < 2:
                await asyncio.sleep(0)
        assert first_speech.interrupted
        expected = [(message.id, message.text_content) for message in prepared]
        assert [text for _, text in expected] == ["Mara,", question]
        for context in (session.history, model.active_session.chat_ctx):
            assert [
                (message.id, message.text_content)
                for message in context.messages()
                if message.role == "user"
            ] == expected
        messages = utils.aio.Chan[llm.MessageGeneration]()
        functions = utils.aio.Chan[llm.FunctionCall]()
        messages.close()
        functions.close()
        model.active_session._reply_futs[-1].set_result(
            llm.GenerationCreatedEvent(
                message_stream=messages,
                function_stream=functions,
                user_initiated=True,
                response_id="after-split-address",
            )
        )

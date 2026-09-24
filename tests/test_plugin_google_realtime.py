from __future__ import annotations

import asyncio
import gc
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import types

from livekit.agents import llm, utils
from livekit.plugins.google.realtime import realtime_api
from livekit.plugins.google.realtime.api_proto import ClientEvents
from livekit.plugins.google.realtime.realtime_api import RealtimeModel, RealtimeSession
from livekit.plugins.google.utils import create_function_response

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("values", [[-1, 0, 1], [-2, 2]])
def test_integer_tool_literals_retain_their_exact_range(values: list[int]) -> None:
    from livekit.plugins.google.utils import _GeminiJsonSchema

    schema = types.Schema.model_validate(
        _GeminiJsonSchema({"type": "integer", "enum": values}).simplify()
    )
    assert schema.enum is None
    allowed = []
    for value in range(-3, 4):
        alternatives = schema.any_of or [schema]
        if any(item.minimum <= value <= item.maximum for item in alternatives):
            allowed.append(value)
    assert allowed == values


def _is_genai_client_teardown(task: asyncio.Task[Any]) -> bool:
    """Whether this task is a genai client's ``aclose()`` left behind by a finalizer.

    Keyed on the coroutine's defining module, not its name alone -- ``aclose``
    is a common method name and this must not touch unrelated tasks. Coroutine
    objects carry no ``__module__``, hence the walk through ``cr_frame``.
    """
    coro = task.get_coro()
    if not (getattr(coro, "__qualname__", "") or "").endswith(".aclose"):
        return False
    frame = getattr(coro, "cr_frame", None)
    module = frame.f_globals.get("__name__", "") if frame else ""
    return module.startswith("google.genai")


@pytest.fixture(autouse=True)
async def _settle_genai_finalizers() -> AsyncIterator[None]:
    """Finish the genai client teardown this test started, before the next one.

    ``AsyncClient.__del__`` schedules ``aclose()`` on whatever event loop is
    running when the collector reaches it, with no check for a client that was
    already closed explicitly -- so even the sessions this module closes
    properly leave a finalizer behind. Settled here, while this test still owns
    the loop, those tasks would otherwise surface as leaked tasks in an
    unrelated test in a later module.
    """
    yield
    gc.collect()
    if pending := [
        task for task in asyncio.all_tasks() if not task.done() and _is_genai_client_teardown(task)
    ]:
        await asyncio.gather(*pending, return_exceptions=True)


# 10ms of silence at the output sample rate (24kHz mono, 16-bit)
_PCM_FRAME = b"\x00\x01" * 240


@asynccontextmanager
async def _make_session(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[RealtimeSession]:
    """A session whose background connect loop is stopped before it hits the network.

    Closed on exit so the genai http clients are released here instead of by
    ``AsyncClient.__del__``, which schedules ``aclose()`` on whatever event loop
    is running when the collector happens to reach them.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    session = RealtimeModel().session()
    # cancel the connect loop before the event loop ever schedules it, so no
    # websocket connection is attempted
    session._msg_ch.close()
    await utils.aio.cancel_and_wait(session._main_atask)
    try:
        yield session
    finally:
        await session.aclose()


@asynccontextmanager
async def _make_configured_session(
    monkeypatch: pytest.MonkeyPatch, **options: object
) -> AsyncIterator[RealtimeSession]:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    session = RealtimeModel(**options).session()  # type: ignore[arg-type]
    session._msg_ch.close()
    await utils.aio.cancel_and_wait(session._main_atask)
    try:
        yield session
    finally:
        await session.aclose()


@pytest.mark.parametrize("model", ["gemini-3.1-flash-live-preview", "gemini-3.8-live"])
async def test_current_models_replace_system_instructions_without_fabricating_speech(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    async with _make_configured_session(monkeypatch, model=model) as session:
        events: list[Any] = []
        session._send_client_event = events.append  # type: ignore[method-assign]
        session._active_session = object()  # type: ignore[assignment]
        try:
            await session.update_instructions("The current scene is the workshop.")
            context = llm.ChatContext.empty()
            context.add_message(role="user", content="The selected object is now the bench.")
            await session.update_chat_ctx(context)
            assert session._session_should_close.is_set()
            assert events == []
            assert session._build_connect_config().system_instruction.parts[0].text == (
                "The current scene is the workshop."
            )
            assert session.chat_ctx.to_dict() == context.to_dict()
        finally:
            session._active_session = None


@pytest.mark.parametrize("model", ["gemini-3.1-flash-live-preview", "gemini-3.8-live"])
async def test_current_models_request_reply_without_a_placeholder_turn(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    async with _make_configured_session(monkeypatch, model=model) as session:
        events: list[Any] = []
        session._send_client_event = events.append  # type: ignore[method-assign]
        reply = session.generate_reply()
        try:
            assert len(events) == 1
            assert events[0].turn_complete is True
            assert not events[0].turns
        finally:
            reply.cancel()
            await asyncio.sleep(0)


async def test_sender_omits_empty_turns_but_preserves_context_only_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_configured_session(monkeypatch, model="gemini-3.8-live") as session:
        sent: list[dict[str, Any]] = []

        async def send_client_content(**kwargs: Any) -> None:
            assert kwargs.get("turns") != [], "Google rejects an explicitly empty turns list"
            sent.append(kwargs)

        transport = SimpleNamespace(send_client_content=send_client_content)
        session._active_session = transport
        session._session_should_close.clear()
        session._msg_ch = utils.aio.Chan()
        session._msg_ch.send_nowait(types.LiveClientContent(turns=[], turn_complete=True))
        turns = [types.Content(role="user", parts=[types.Part(text="Updated scene")])]
        session._msg_ch.send_nowait(types.LiveClientContent(turns=turns, turn_complete=False))
        session._msg_ch.close()
        try:
            await session._send_task(transport)
            assert sent == [
                {"turn_complete": True},
                {"turns": turns, "turn_complete": False},
            ]
        finally:
            session._active_session = None


def _audio_content(**kwargs: object) -> types.LiveServerContent:
    return types.LiveServerContent(
        model_turn=types.Content(
            parts=[types.Part(inline_data=types.Blob(data=_PCM_FRAME, mime_type="audio/pcm"))]
        ),
        **kwargs,  # type: ignore[arg-type]
    )


async def _drain_generation(
    event: llm.GenerationCreatedEvent,
) -> tuple[str, int, list[str]]:
    text = ""
    audio_frames = 0
    async for message in event.message_stream:
        async for chunk in message.text_stream:
            text += chunk
        async for _frame in message.audio_stream:
            audio_frames += 1

    function_calls = [call.name async for call in event.function_stream]
    return text, audio_frames, function_calls


async def test_unspoken_model_text_is_omitted_in_audio_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        session._start_new_generation()
        gen = session._current_generation
        assert gen is not None

        session._handle_server_content(
            types.LiveServerContent(
                model_turn=types.Content(
                    parts=[types.Part(text="call:getWeather{location:Seattle")]
                ),
                output_transcription=types.Transcription(text="Let me check."),
            )
        )

        assert gen.output_text == "Let me check."
        assert gen.text_ch.recv_nowait() == "Let me check."
        assert gen.text_ch.empty()


async def test_model_text_is_forwarded_in_text_modality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_configured_session(monkeypatch, modalities=[types.Modality.TEXT]) as session:
        session._start_new_generation()
        gen = session._current_generation
        assert gen is not None

        session._handle_server_content(
            types.LiveServerContent(
                model_turn=types.Content(parts=[types.Part(text="Hello there.")])
            )
        )

        assert gen.output_text == "Hello there."
        assert gen.text_ch.recv_nowait() == "Hello there."


async def test_model_text_is_forwarded_without_output_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_configured_session(monkeypatch, output_audio_transcription=None) as session:
        session._start_new_generation()
        gen = session._current_generation
        assert gen is not None

        session._handle_server_content(
            types.LiveServerContent(
                model_turn=types.Content(parts=[types.Part(text="Hello there.")])
            )
        )

        assert gen.output_text == "Hello there."
        assert gen.text_ch.recv_nowait() == "Hello there."


async def test_transcript_contains_only_output_transcription_with_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        generations: list[llm.GenerationCreatedEvent] = []
        session.on("generation_created", generations.append)
        session._start_new_generation()

        session._handle_server_content(
            types.LiveServerContent(
                model_turn=types.Content(parts=[types.Part(text="call:assetGenerator{context:")])
            )
        )
        session._handle_server_content(_audio_content())
        session._handle_server_content(
            types.LiveServerContent(output_transcription=types.Transcription(text="Tako je!"))
        )
        session._handle_server_content(types.LiveServerContent(generation_complete=True))
        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        assert len(generations) == 1
        assert await _drain_generation(generations[0]) == ("Tako je!", 1, [])


async def test_tool_call_is_delivered_without_written_call_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_session(monkeypatch) as session:
        generations: list[llm.GenerationCreatedEvent] = []
        session.on("generation_created", generations.append)
        session._start_new_generation()

        session._handle_server_content(
            types.LiveServerContent(
                model_turn=types.Content(parts=[types.Part(text="call:getWeather{location:")])
            )
        )
        session._handle_tool_calls(
            types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(id="fc-1", name="getWeather", args={"location": "Seattle"})
                ]
            )
        )

        assert len(generations) == 1
        assert await _drain_generation(generations[0]) == ("", 0, ["getWeather"])


async def test_transcript_keeps_model_text_in_text_modality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_configured_session(monkeypatch, modalities=[types.Modality.TEXT]) as session:
        generations: list[llm.GenerationCreatedEvent] = []
        session.on("generation_created", generations.append)
        session._start_new_generation()

        session._handle_server_content(
            types.LiveServerContent(
                model_turn=types.Content(parts=[types.Part(text="Hello there.")]),
                turn_complete=True,
            )
        )

        assert len(generations) == 1
        assert await _drain_generation(generations[0]) == ("Hello there.", 0, [])


async def test_transcript_keeps_model_text_without_output_transcription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_configured_session(monkeypatch, output_audio_transcription=None) as session:
        generations: list[llm.GenerationCreatedEvent] = []
        session.on("generation_created", generations.append)
        session._start_new_generation()

        session._handle_server_content(
            types.LiveServerContent(
                model_turn=types.Content(parts=[types.Part(text="Hello there.")]),
                turn_complete=True,
            )
        )

        assert len(generations) == 1
        assert await _drain_generation(generations[0]) == ("Hello there.", 0, [])


async def test_output_streams_close_on_generation_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generation_complete ends the audio/text segment; finalization waits for turn_complete.

    Gemini delays turn_complete until it estimates client-side playback has finished, so
    keying the stream close off turn_complete makes AudioSegmentEnd (and the finalized
    transcript) arrive seconds late (issue #6421). Both streams must close on
    generation_complete, while the generation stays open until turn_complete for input
    transcription and metrics.
    """
    async with _make_session(monkeypatch) as session:
        session._start_new_generation()
        gen = session._current_generation
        assert gen is not None

        session._handle_server_content(
            _audio_content(
                output_transcription=types.Transcription(text="hello"),
                generation_complete=True,
            )
        )

        # audio and text were consumed and both segments ended immediately
        assert gen._first_token_timestamp is not None
        assert gen.output_text == "hello"
        assert gen.audio_ch.closed
        assert gen.text_ch.closed
        # but the generation is still open for trailing input transcription until turn_complete
        assert not gen._done
        assert not gen.message_ch.closed

        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        assert gen._done
        assert gen.message_ch.closed


async def test_late_content_after_generation_complete_is_dropped(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Stray audio/text after generation_complete is dropped (not pushed to a closed stream)."""
    async with _make_session(monkeypatch) as session:
        session._start_new_generation()
        gen = session._current_generation
        assert gen is not None

        session._handle_server_content(_audio_content(generation_complete=True))
        assert gen.audio_ch.closed and gen.text_ch.closed

        with caplog.at_level(logging.WARNING):
            # must not raise ChanClosed, must not append to the transcript, and must warn
            session._handle_server_content(
                _audio_content(output_transcription=types.Transcription(text="late"))
            )

        assert gen.audio_ch.closed and gen.text_ch.closed
        assert gen.output_text == ""
        assert not gen._done
        assert any("after generation completed" in r.message for r in caplog.records)


async def test_input_transcription_uses_generation_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interim and final transcripts stay on the timeline before the reply they prompted."""
    async with _make_session(monkeypatch) as session:
        transcripts: list[llm.InputTranscriptionCompleted] = []
        session.on("input_audio_transcription_completed", transcripts.append)
        session._start_new_generation()
        gen = session._current_generation
        assert gen is not None
        gen._created_timestamp = 1234.5

        session._handle_server_content(
            types.LiveServerContent(input_transcription=types.Transcription(text="hello"))
        )
        session._handle_server_content(types.LiveServerContent(turn_complete=True))

        assert [(event.is_final, event.turn_started_at) for event in transcripts] == [
            (False, 1234.5),
            (True, 1234.5),
        ]


async def test_disabled_input_transcription_does_not_duplicate_external_stt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_configured_session(
        monkeypatch, model="gemini-3.8-live", input_audio_transcription=None
    ) as session:
        transcripts: list[llm.InputTranscriptionCompleted] = []
        session.on("input_audio_transcription_completed", transcripts.append)
        session._chat_ctx.add_message(id="external-final", role="user", content="Read my reward.")
        session._start_new_generation()
        session._handle_server_content(
            types.LiveServerContent(input_transcription=types.Transcription(text="my reward"))
        )
        session._handle_server_content(types.LiveServerContent(turn_complete=True))
        assert not session.capabilities.user_transcription
        assert transcripts == []
        assert [message.id for message in session.chat_ctx.messages()] == ["external-final"]


async def test_session_close_releases_the_genai_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """aclose() must release the genai http clients.

    Otherwise they live until the collector runs ``AsyncClient.__del__``, which
    does ``asyncio.get_running_loop().create_task(self.aclose())`` - creating
    pending tasks on whatever event loop is running at that moment.
    """
    closed = False

    async with _make_session(monkeypatch) as session:
        real_aclose = session._client.aio.aclose

        async def _spy() -> None:
            nonlocal closed
            closed = True
            await real_aclose()

        monkeypatch.setattr(session._client.aio, "aclose", _spy)

    assert closed


def _tool_call(call_id: str = "fc_1", name: str = "lookup") -> types.LiveServerToolCall:
    return types.LiveServerToolCall(
        function_calls=[types.FunctionCall(id=call_id, name=name, args={})]
    )


def _tool_output(
    call_id: str = "fc_1", name: str = "lookup", *, reply_required: bool = True
) -> llm.FunctionCallOutput:
    return llm.FunctionCallOutput(
        call_id=call_id,
        name=name,
        output="42",
        is_error=False,
        reply_required=reply_required,
    )


@asynccontextmanager
async def _make_connected_session(
    monkeypatch: pytest.MonkeyPatch, *, non_blocking_tools: bool = False
) -> AsyncIterator[RealtimeSession]:
    """A session that believes it is connected, so update_chat_ctx actually emits.

    The placeholder is never called: the send task is not running, so client events just
    queue up in `_msg_ch` for the test to inspect. `_make_session` closes that channel to
    stop the connect loop, so it is replaced with an open one first.
    """
    async with _make_session(monkeypatch) as session:
        if non_blocking_tools:
            session._opts.tool_behavior = types.Behavior.NON_BLOCKING
        session._msg_ch = utils.aio.Chan[ClientEvents]()
        session._active_session = object()  # type: ignore[assignment]
        try:
            yield session
        finally:
            # the placeholder has no close(), drop it before aclose() reaches for one
            session._active_session = None


async def _drain_sent(session: RealtimeSession) -> list[object]:
    sent: list[object] = []
    while not session._msg_ch.empty():
        sent.append(session._msg_ch.recv_nowait())
    return sent


@pytest.mark.parametrize(
    "reply_required, scheduling",
    [(False, types.FunctionResponseScheduling.SILENT), (True, None)],
)
async def test_tool_response_scheduling_follows_the_output(
    monkeypatch: pytest.MonkeyPatch,
    reply_required: bool,
    scheduling: types.FunctionResponseScheduling | None,
) -> None:
    """A result owed by an interrupted turn goes out SILENT; a normal one keeps the default.

    Gemini blocks the turn until every call is answered and offers no cancel, so dropping the
    result strands the session (issue #6569). SILENT records it without prompting speech.
    """
    async with _make_connected_session(monkeypatch, non_blocking_tools=True) as session:
        session._start_new_generation()
        session._handle_tool_calls(_tool_call())
        await _drain_sent(session)

        chat_ctx = session.chat_ctx.copy()
        chat_ctx.items.append(_tool_output(reply_required=reply_required))
        await session.update_chat_ctx(chat_ctx)

        sent = await _drain_sent(session)
        responses = [m for m in sent if isinstance(m, types.LiveClientToolResponse)]
        assert len(responses) == 1, f"expected the tool response to be sent, got {sent}"
        assert responses[0].function_responses is not None
        assert responses[0].function_responses[0].id == "fc_1"
        assert responses[0].function_responses[0].scheduling == scheduling


async def test_blocking_tools_send_the_response_and_warn_it_cannot_be_silent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Gemini ignores scheduling on BLOCKING declarations, so the reply cannot be prevented.

    It is sent anyway, since unblocking the turn matters more, and every one is reported.
    """
    async with _make_connected_session(monkeypatch) as session:
        session._start_new_generation()
        session._handle_tool_calls(
            types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(id="fc_1", name="lookup", args={}),
                    types.FunctionCall(id="fc_2", name="search", args={}),
                ]
            )
        )
        await _drain_sent(session)

        with caplog.at_level(logging.WARNING):
            for call_id, name in (("fc_1", "lookup"), ("fc_2", "search")):
                chat_ctx = session.chat_ctx.copy()
                chat_ctx.items.append(_tool_output(call_id, name, reply_required=False))
                await session.update_chat_ctx(chat_ctx)

        responses = [
            m for m in await _drain_sent(session) if isinstance(m, types.LiveClientToolResponse)
        ]
        assert len(responses) == 2
        assert all(r.function_responses[0].scheduling is None for r in responses)  # type: ignore[index]

        warnings = [r for r in caplog.records if "wants no reply" in r.message]
        assert len(warnings) == 2, "every update reports what it could not keep quiet"
        assert [r.functions for r in warnings] == [["lookup"], ["search"]]  # type: ignore[attr-defined]


@pytest.mark.parametrize("vertexai", [False, True])
def test_function_response_scheduling_only_for_gemini_api(vertexai: bool) -> None:
    """Vertex AI rejects `scheduling` (and `id`), so neither is set for it."""
    res = create_function_response(
        _tool_output(),
        vertexai=vertexai,
        tool_response_scheduling=types.FunctionResponseScheduling.SILENT,
    )

    if vertexai:
        assert res.scheduling is None
        assert res.id is None
    else:
        assert res.scheduling == types.FunctionResponseScheduling.SILENT
        assert res.id == "fc_1"


def test_vertex_scheduling_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An explicitly set scheduling is dropped on Vertex AI, so say so instead of ignoring it."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    with caplog.at_level(logging.WARNING):
        RealtimeModel(
            vertexai=True,
            project="p",
            location="us-central1",
            tool_response_scheduling=types.FunctionResponseScheduling.SILENT,
        )

    assert any("tool_response_scheduling is not supported" in r.message for r in caplog.records)


def test_gemini_api_scheduling_does_not_warn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    with caplog.at_level(logging.WARNING):
        RealtimeModel(tool_response_scheduling=types.FunctionResponseScheduling.SILENT)

    assert not any("tool_response_scheduling is not supported" in r.message for r in caplog.records)


class _FakeLiveSession:
    """Stands in for the genai live session and records what the plugin sends."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, object]] = []
        self._closed = asyncio.Event()

    async def send_client_content(self, *, turns: object = None, turn_complete: bool) -> None:
        self.sent.append(("content", turns))

    async def send_tool_response(self, *, function_responses: object) -> None:
        self.sent.append(("tool_response", function_responses))

    async def send_realtime_input(self, **kwargs: object) -> None:
        self.sent.append(("realtime", kwargs))

    async def receive(self) -> AsyncIterator[types.LiveServerMessage]:
        await self._closed.wait()
        return
        yield  # pragma: no cover - makes this an async generator

    async def close(self) -> None:
        self._closed.set()


async def test_external_address_survives_cancel_into_the_next_google_wire_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.genai.live import AsyncLive

    from livekit.agents import Agent, AgentSession

    from .fake_stt import FakeSTT
    from .test_realtime_adaptive_interruption import _end_of_turn_info

    sockets: list[_FakeLiveSession] = []
    contexts: list[list[tuple[str, str | None]]] = []
    configurations: list[Any] = []
    prepared = []

    @asynccontextmanager
    async def connect(self: AsyncLive, **kwargs: Any) -> AsyncIterator[_FakeLiveSession]:
        socket = _FakeLiveSession()
        sockets.append(socket)
        configurations.append(kwargs["config"])
        yield socket

    class PreparedAgent(Agent):
        async def on_user_turn_completed(self, turn_ctx, new_message):
            prepared.append(new_message)

        async def realtime_context_node(self, context):
            contexts.append(
                [(m.id, m.text_content) for m in context.messages() if m.role == "user"]
            )
            return context

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(AsyncLive, "connect", connect)
    model = RealtimeModel(
        model="gemini-3.8-live",
        input_audio_transcription=None,
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
        ),
    )
    question = "Explain how to tell a loose joint from a split brace. Advice only."

    def sent_texts() -> list[str]:
        return [
            part.text
            for socket in sockets
            for kind, turns in socket.sent
            if kind == "content" and turns
            for turn in turns
            for part in turn.parts or []
            if part.text
        ]

    try:
        async with AgentSession(
            llm=model, stt=FakeSTT(), turn_handling={"turn_detection": "manual"}
        ) as session:
            await session.start(
                PreparedAgent(instructions="Route present NPC questions to their agent.")
            )
            session._activity.on_end_of_turn(_end_of_turn_info("Mara,"))
            async with asyncio.timeout(3):
                while "Mara," not in sent_texts():
                    await asyncio.sleep(0)
            first_speech = session._activity._current_speech
            session._activity.on_end_of_turn(_end_of_turn_info(question))
            async with asyncio.timeout(3):
                while question not in sent_texts():
                    await asyncio.sleep(0)
            assert first_speech.interrupted
            expected = [(message.id, message.text_content) for message in prepared]
            assert [text for _, text in expected] == ["Mara,", question]
            assert contexts[-1] == expected
            assert [
                (m.id, m.text_content) for m in session.history.messages() if m.role == "user"
            ] == expected
            assert sent_texts().index("Mara,") < sent_texts().index(question)
            # A fresh connection must restore the address, rather than relying
            # on a previous socket's history after cancellation.
            if len(sockets) > 1:
                last_texts = [
                    part.text
                    for kind, turns in sockets[-1].sent
                    if kind == "content" and turns
                    for turn in turns
                    for part in turn.parts or []
                    if part.text
                ]
                assert "Mara," in last_texts
                assert configurations[-1].session_resumption.handle is None
    finally:
        await model.aclose()


async def test_completed_tool_free_greeting_restores_tools_for_player_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.genai.live import AsyncLive

    from livekit.agents import Agent, AgentSession, function_tool

    from .fake_io import FakeAudioOutput, FakeTextOutput

    called = asyncio.Event()

    class Socket(_FakeLiveSession):
        def __init__(self):
            super().__init__()
            self.incoming = asyncio.Queue()
            self.replies = 0

        async def send_client_content(self, *, turns=None, turn_complete):
            await super().send_client_content(turns=turns, turn_complete=turn_complete)
            if not turn_complete:
                return
            self.replies += 1
            if self.replies == 2:
                self.incoming.put_nowait(
                    types.LiveServerMessage(
                        tool_call=types.LiveServerToolCall(
                            function_calls=[
                                types.FunctionCall(id="player-lookup", name="lookup", args={})
                            ]
                        )
                    )
                )
            else:
                self.incoming.put_nowait(
                    types.LiveServerMessage(
                        server_content=_audio_content(
                            output_transcription=types.Transcription(text="Welcome."),
                            turn_complete=True,
                        )
                    )
                )

        async def receive(self):
            while not self._closed.is_set():
                yield await self.incoming.get()

    socket = Socket()

    @asynccontextmanager
    async def connect(self, **kwargs):
        yield socket

    class ToolAgent(Agent):
        @function_tool
        async def lookup(self) -> str:
            """Retrieve the current facts."""
            called.set()
            return "Current facts."

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(AsyncLive, "connect", connect)
    model = RealtimeModel(model="gemini-3.8-live")
    try:
        async with AgentSession(llm=model) as session:
            session.output.audio = FakeAudioOutput()
            session.output.transcription = FakeTextOutput()
            await session.start(ToolAgent(instructions="Use lookup for player questions."))
            greeting = session.generate_reply(user_input="Greet briefly.", tool_choice="none")
            await asyncio.wait_for(greeting, 3)
            await asyncio.wait_for(session.wait_for_idle(), 3)
            assert session._activity._rt_session._opts.tool_choice is None
            session.generate_reply(user_input="Retrieve the facts.")
            await asyncio.wait_for(called.wait(), 3)
    finally:
        await model.aclose()


@asynccontextmanager
async def _connected_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    handle: str | None,
    known: llm.ChatContext | None = None,
    sent_after_handle: llm.ChatContext | None = None,
    pending: llm.ChatContext | None = None,
    caller_handle: bool = False,
    model: str | None = None,
    unsent: llm.ChatContext | None = None,
) -> AsyncIterator[tuple[RealtimeSession, _FakeLiveSession]]:
    """Connect once onto a fake socket.

    `known` is the state the handle stands for, `sent_after_handle` what the previous
    socket synced after the handle arrived, `pending` the update that arrives before the
    connect loop runs. `caller_handle` passes the handle through `RealtimeModel` instead,
    so its baseline is unknown. `unsent` lists items of `sent_after_handle` that were
    queued but never sent before the socket dropped.
    """
    from google.genai.live import AsyncLive

    fake = _FakeLiveSession()

    @asynccontextmanager
    async def _connect(self: AsyncLive, **kwargs: object) -> AsyncIterator[_FakeLiveSession]:
        yield fake

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(AsyncLive, "connect", _connect)
    model_kwargs = {"model": model} if model else {}
    if caller_handle:
        session = RealtimeModel(
            session_resumption=types.SessionResumptionConfig(handle=handle), **model_kwargs
        ).session()
    else:
        session = RealtimeModel(**model_kwargs).session()
        session._session_resumption_handle = handle
    if known is not None:
        session._resumption_chat_ctx = known
        session._chat_ctx = sent_after_handle if sent_after_handle is not None else known
    if unsent is not None:
        session._unsent_item_ids = {item.id for item in unsent.items}
    if pending is not None:
        await session.update_chat_ctx(pending)
    try:
        while session._active_session is None:
            await asyncio.sleep(0.01)
        # let the send task drain the queued events onto the fake socket
        await asyncio.sleep(0.05)
        yield session, fake
    finally:
        await session.aclose()


def _texts(sent: list[tuple[str, object]]) -> list[list[str]]:
    return [
        [p.text for c in turns for p in c.parts]  # type: ignore[attr-defined]
        for kind, turns in sent
        if kind == "content"
    ]


async def test_fresh_session_replays_chat_ctx(monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content="hello")
    ctx.add_message(role="assistant", content="hi")

    async with _connected_session(monkeypatch, handle=None, pending=ctx) as (session, fake):
        assert _texts(fake.sent) == [["hello", "hi"]]
        assert session._pending_chat_ctx is None


async def test_instruction_change_replaces_connected_system_and_preserves_completed_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.genai.live import AsyncLive

    sockets: list[_FakeLiveSession] = []
    configurations: list[Any] = []
    completions: list[bool] = []

    class Socket(_FakeLiveSession):
        async def send_client_content(self, *, turns=None, turn_complete):
            completions.append(turn_complete)
            await super().send_client_content(turns=turns, turn_complete=turn_complete)

    @asynccontextmanager
    async def connect(self, **kwargs):
        socket = Socket()
        sockets.append(socket)
        configurations.append(kwargs["config"])
        yield socket

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(AsyncLive, "connect", connect)
    model = RealtimeModel(model="gemini-3.8-live", instructions="Greet only. No tools.")
    session = model.session()
    context = llm.ChatContext.empty()
    context.add_message(id="greeting", role="assistant", content="Welcome.")
    await session.update_chat_ctx(context)
    try:
        async with asyncio.timeout(2):
            while not sockets or not sockets[0].sent:
                await asyncio.sleep(0)
            session._session_resumption_handle = "would-restore-greeting-only-system"
            await session.update_instructions("Use tools to save the player's chosen stakes.")
            assert session._session_should_close.is_set()
            latest = session.chat_ctx.copy()
            latest.items.extend(
                [
                    llm.FunctionCall(
                        id="call", call_id="saved-stakes", name="prepare", arguments="{}"
                    ),
                    llm.FunctionCallOutput(
                        id="result",
                        call_id="saved-stakes",
                        name="prepare",
                        output="Saved terms A.",
                        is_error=False,
                    ),
                ]
            )
            latest.add_message(id="choice", role="user", content="Change the approach.")
            await session.update_chat_ctx(latest)
            while len(sockets) != 2 or not sockets[1].sent:
                await asyncio.sleep(0)
        assert configurations[0].system_instruction.parts[0].text == "Greet only. No tools."
        assert configurations[1].system_instruction.parts[0].text == (
            "Use tools to save the player's chosen stakes."
        )
        assert configurations[1].session_resumption.handle is None
        assert sockets[0]._closed.is_set()
        restored = _texts(sockets[1].sent)[0]
        assert restored[0] == "Welcome."
        assert '"output": "Saved terms A."' in restored[1]
        assert "already completed" in restored[1]
        assert restored[2] == "Change the approach."
        assert all(kind == "content" for kind, _ in sockets[1].sent)
        assert not any(completions)
        assert session.chat_ctx.get_by_id("result") is not None
        await session.update_instructions("Use tools to save the player's chosen stakes.")
        assert not session._session_should_close.is_set()
    finally:
        await session.aclose()
        await model.aclose()


async def test_context_restart_during_connect_keeps_the_new_reply_on_the_new_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.genai.live import AsyncLive

    connecting, release = asyncio.Event(), asyncio.Event()
    sockets = []

    class Socket(_FakeLiveSession):
        def __init__(self):
            super().__init__()
            self.completions = 0
            self.incoming = asyncio.Queue()

        async def send_client_content(self, *, turns=None, turn_complete):
            await super().send_client_content(turns=turns, turn_complete=turn_complete)
            if turn_complete:
                self.completions += 1
                await self.incoming.put(
                    types.LiveServerMessage(
                        server_content=types.LiveServerContent(
                            model_turn=types.Content(
                                role="model", parts=[types.Part(text="Ready.")]
                            ),
                            generation_complete=True,
                            turn_complete=True,
                        )
                    )
                )

        async def receive(self):
            while not self._closed.is_set():
                yield await self.incoming.get()

    @asynccontextmanager
    async def connect(self, **kwargs):
        socket = Socket()
        sockets.append(socket)
        assert len(sockets) <= 2
        if len(sockets) == 1:
            connecting.set()
            await release.wait()
        yield socket

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(AsyncLive, "connect", connect)
    model = RealtimeModel(model="gemini-3.8-live")
    session = model.session()
    try:
        await asyncio.wait_for(connecting.wait(), 1)
        await session.update_instructions("Current campaign and NPC roster.")
        context = llm.ChatContext.empty()
        context.add_message(id="original-player-input", role="user", content="Recall my reward.")
        await session.update_chat_ctx(context)
        reply = session.generate_reply()
        release.set()
        await asyncio.wait_for(reply, 2)
        assert len(sockets) == 2
        assert [socket.completions for socket in sockets] == [0, 1]
        assert _texts(sockets[1].sent[:1]) == [["Recall my reward."]]
        assert [message.id for message in session.chat_ctx.messages()].count(
            "original-player-input"
        ) == 1
    finally:
        release.set()
        await session.aclose()
        await model.aclose()


async def test_generate_reply_uses_the_configured_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    from livekit.agents import APIConnectOptions

    async with _make_configured_session(
        monkeypatch, model="gemini-3.8-live", conn_options=APIConnectOptions(timeout=0.02)
    ) as session:
        with pytest.raises(llm.RealtimeError, match="timed out"):
            await asyncio.wait_for(session.generate_reply(), 0.2)


async def test_timed_out_reply_retires_socket_without_late_tools_or_replaying_saved_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.genai.live import AsyncLive

    from livekit.agents import APIConnectOptions

    class Socket(_FakeLiveSession):
        def __init__(self):
            super().__init__()
            self.incoming = asyncio.Queue()
            self.completions = []

        async def send_client_content(self, *, turns=None, turn_complete):
            self.completions.append(turn_complete)
            await super().send_client_content(turns=turns, turn_complete=turn_complete)

        async def receive(self):
            while True:
                message = await self.incoming.get()
                if message is None:
                    return
                yield message

        async def close(self):
            await super().close()
            self.incoming.put_nowait(None)

    sockets = []
    configs = []

    @asynccontextmanager
    async def connect(self, **kwargs):
        socket = Socket()
        sockets.append(socket)
        configs.append(kwargs["config"])
        yield socket

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(AsyncLive, "connect", connect)
    model = RealtimeModel(
        model="gemini-3.8-live", conn_options=APIConnectOptions(timeout=0.05, max_retry=0)
    )
    session = model.session()
    generations = []
    session.on("generation_created", generations.append)
    context = llm.ChatContext.empty()
    context.add_message(id="consent", role="user", content="Proceed once.")
    context.items.extend(
        [
            llm.FunctionCall(id="call", call_id="saved-roll", name="resolve_check", arguments="{}"),
            llm.FunctionCallOutput(
                id="receipt",
                call_id="saved-roll",
                name="resolve_check",
                output='{"rawDie":5,"status":"rolled"}',
                is_error=False,
            ),
        ]
    )
    await session.update_chat_ctx(context)
    try:
        async with asyncio.timeout(2):
            while not sockets or not sockets[0].sent:
                await asyncio.sleep(0)
            session._session_resumption_handle = "old-provider-response"
            with pytest.raises(llm.RealtimeError, match="timed out"):
                await session.generate_reply()
            sockets[0].incoming.put_nowait(
                types.LiveServerMessage(
                    tool_call=types.LiveServerToolCall(
                        function_calls=[
                            types.FunctionCall(id="late-call", name="record_outcome", args={})
                        ],
                    )
                )
            )
            while len(sockets) < 2 or not sockets[1].sent:
                await asyncio.sleep(0)
            assert sockets[0]._closed.is_set()
            assert configs[1].session_resumption.handle is None
            assert not generations
            assert session.chat_ctx.get_by_id("receipt") is not None
            assert not any(
                isinstance(item, llm.FunctionCall) and item.call_id == "late-call"
                for item in session.chat_ctx.items
            )
            assert not any(sockets[1].completions)
            assert all(kind == "content" for kind, _ in sockets[1].sent)
            reply = session.generate_reply()
            sockets[1].incoming.put_nowait(
                types.LiveServerMessage(
                    server_content=types.LiveServerContent(
                        model_turn=types.Content(
                            role="model", parts=[types.Part(text="The saved die is five.")]
                        ),
                        generation_complete=True,
                        turn_complete=True,
                    )
                )
            )
            assert (await reply).user_initiated is True
            assert len(generations) == 1
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.parametrize("change", ["remove", "replace"])
async def test_replacing_google_history_discards_the_old_resumption_handle(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    async with _make_configured_session(monkeypatch, model="gemini-3.8-live") as session:
        known = llm.ChatContext.empty()
        known.add_message(id="scene", role="user", content="Old canonical image facts")
        updated = llm.ChatContext.empty()
        if change == "replace":
            updated.add_message(id="scene", role="user", content="New canonical image facts")
        session._chat_ctx = known
        session._session_resumption_handle = "would-restore-forbidden-history"
        session._resumption_chat_ctx = known.copy()
        session._active_session = object()  # type: ignore[assignment]
        try:
            await session.update_chat_ctx(updated)
            assert session._session_should_close.is_set()
            assert session._session_resumption_handle is None
            assert session._resumption_chat_ctx is None
            assert session.chat_ctx.to_dict() == updated.to_dict()
        finally:
            session._active_session = None


async def test_interrupted_google_reply_restarts_with_only_the_heard_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_configured_session(monkeypatch, model="gemini-3.8-live") as session:
        context = llm.ChatContext.empty()
        context.add_message(id="reply", role="assistant", content="Heard. Never played.")
        session._chat_ctx = context
        session._session_resumption_handle = "old-full-answer"
        session.truncate(
            message_id="reply",
            modalities=["audio", "text"],
            audio_end_ms=1000,
            audio_transcript="Heard.",
        )
        assert session._session_resumption_handle is None
        assert session.chat_ctx.get_by_id("reply").text_content == "Heard."
        assert session._session_should_close.is_set()


async def test_same_canonical_image_with_new_local_cache_id_does_not_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_configured_session(monkeypatch, model="gemini-3.8-live") as session:
        original, refreshed = llm.ChatContext.empty(), llm.ChatContext.empty()
        for ctx in (original, refreshed):
            ctx.add_message(
                id="scene",
                role="user",
                content=[
                    "Scene revision 2",
                    llm.ImageContent(image="https://fixture.invalid/current.png"),
                ],
            )
        session._chat_ctx = original
        session._session_should_close.clear()
        session._active_session = object()  # type: ignore[assignment]
        try:
            await session.update_chat_ctx(refreshed)
            assert not session._session_should_close.is_set()
            changed = llm.ChatContext.empty()
            changed.add_message(
                id="scene",
                role="user",
                content=[
                    "Scene revision 3",
                    llm.ImageContent(image="https://fixture.invalid/new.png"),
                ],
            )
            await session.update_chat_ctx(changed)
            assert session._session_should_close.is_set()
        finally:
            session._active_session = None


async def test_response_instructions_are_not_fabricated_model_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_configured_session(monkeypatch, model="gemini-3.8-live") as session:
        events: list[Any] = []
        session._send_client_event = events.append  # type: ignore[method-assign]
        pending = session.generate_reply(instructions="Answer the latest player briefly.")
        try:
            content = next(event for event in events if isinstance(event, types.LiveClientContent))
            assert content.turn_complete is True
            assert content.turns and content.turns[0].role == "user"
            assert "not dialogue" in content.turns[0].parts[0].text
            assert "Answer the latest player briefly." in content.turns[0].parts[0].text
            assert list(session.chat_ctx.messages()) == []
        finally:
            pending.cancel()


async def test_unheard_google_reply_is_not_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_configured_session(monkeypatch, model="gemini-3.8-live") as session:
        session._start_new_generation()
        generation = session._current_generation
        generation.output_text = "Never heard."
        session.truncate(
            message_id=generation.response_id,
            modalities=["audio"],
            audio_end_ms=0,
            audio_transcript="",
        )
        assert session.chat_ctx.get_by_id(generation.response_id) is None
        assert generation._done


async def test_idle_interrupt_does_not_create_an_empty_manual_audio_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_configured_session(
        monkeypatch,
        model="gemini-3.8-live",
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
        ),
    ) as session:
        session.interrupt()
        assert not session._in_user_activity
        session._start_new_generation()
        session.interrupt()
        assert not session._in_user_activity


async def test_typed_interrupt_does_not_replay_empty_microphone_activity_on_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_configured_session(
        monkeypatch,
        model="gemini-3.8-live",
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
        ),
    ) as session:
        events: list[Any] = []
        session._send_client_event = events.append  # type: ignore[method-assign]
        session._start_new_generation()
        reply = session._current_generation
        session.interrupt()
        assert len(events) == 1 and events[0].activity_start is not None
        events.clear()
        session.truncate(
            message_id=reply.response_id,
            modalities=["audio"],
            audio_end_ms=2000,
            audio_transcript="",
        )
        context = session.chat_ctx.copy()
        context.add_message(id="typed-after-interruption", role="user", content="What did I earn?")
        await session.update_chat_ctx(context)
        future = session.generate_reply()
        try:
            assert not any(isinstance(event, types.LiveClientRealtimeInput) for event in events)
            assert any(
                isinstance(event, types.LiveClientContent) and event.turn_complete
                for event in events
            )
            assert session.chat_ctx.get_by_id("typed-after-interruption") is not None
        finally:
            future.cancel()


async def test_truncated_history_reconnects_without_replaying_tools_or_old_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.genai.live import AsyncLive

    sockets: list[_FakeLiveSession] = []
    configurations: list[Any] = []

    @asynccontextmanager
    async def connect(self: AsyncLive, **kwargs: Any) -> AsyncIterator[_FakeLiveSession]:
        socket = _FakeLiveSession()
        sockets.append(socket)
        configurations.append(kwargs["config"])
        yield socket

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(AsyncLive, "connect", connect)
    session = RealtimeModel(model="gemini-3.8-live").session()
    context = llm.ChatContext.empty()
    context.add_message(id="question", role="user", content="What did I earn?")
    context.add_message(id="reply", role="assistant", content="A meal. Unheard extra reward.")
    context.items.extend(
        [
            llm.FunctionCall(id="call", call_id="lookup-1", name="lookup_reward", arguments="{}"),
            llm.FunctionCallOutput(
                id="result",
                call_id="lookup-1",
                name="lookup_reward",
                output="A meal.",
                is_error=False,
            ),
        ]
    )
    await session.update_chat_ctx(context)
    try:
        async with asyncio.timeout(2):
            while len(sockets) != 1 or not sockets[0].sent:
                await asyncio.sleep(0)
            session._session_resumption_handle = "must-not-resume-unheard-content"
            session.truncate(
                message_id="reply",
                modalities=["audio", "text"],
                audio_end_ms=900,
                audio_transcript="A meal.",
            )
            # A finalized turn arriving during reconnect must survive the replacement.
            latest = session.chat_ctx.copy()
            latest.add_message(id="next", role="user", content="And a place to sleep?")
            await session.update_chat_ctx(latest)
            while len(sockets) != 2 or not sockets[1].sent:
                await asyncio.sleep(0)
        assert sockets[0]._closed.is_set()
        assert configurations[1].session_resumption.handle is None
        restored = _texts(sockets[1].sent)[0]
        assert restored[:2] == ["What did I earn?", "A meal."]
        assert '"output": "A meal."' in restored[2]
        assert "already completed" in restored[2]
        assert restored[3] == "And a place to sleep?"
        assert all(kind == "content" for kind, _ in sockets[1].sent)
        assert session.chat_ctx.get_by_id("result") is not None
        assert session.chat_ctx.get_by_id("next") is not None
        assert session._pending_generation_fut is None
    finally:
        await session.aclose()


async def test_history_reset_preserves_in_progress_manual_audio_once_on_new_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.genai.live import AsyncLive

    from livekit import rtc
    from livekit.plugins.google.realtime.realtime_api import INPUT_AUDIO_SAMPLE_RATE

    sockets: list[_FakeLiveSession] = []
    configurations: list[Any] = []

    @asynccontextmanager
    async def connect(self: AsyncLive, **kwargs: Any) -> AsyncIterator[_FakeLiveSession]:
        socket = _FakeLiveSession()
        sockets.append(socket)
        configurations.append(kwargs["config"])
        yield socket

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(AsyncLive, "connect", connect)
    session = RealtimeModel(
        model="gemini-3.8-live",
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
        ),
    ).session()

    def audio(milliseconds: int, byte: int) -> bytes:
        data = bytes([byte]) * (INPUT_AUDIO_SAMPLE_RATE * milliseconds // 1000 * 2)
        session.push_audio(rtc.AudioFrame(data, INPUT_AUDIO_SAMPLE_RATE, 1, len(data) // 2))
        return data

    try:
        async with asyncio.timeout(2):
            while not sockets or session._active_session is None:
                await asyncio.sleep(0)
            session.start_user_activity()
            prefix = audio(100, 1)
            partial = audio(20, 2)
            while len(sockets[0].sent) < 3:
                await asyncio.sleep(0)
            session._session_resumption_handle = "old-audio-handle"
            session._reset_chat_ctx(session.chat_ctx)
            suffix = audio(45, 3)
            generation = session.generate_reply()
            while len(sockets) < 2 or len(sockets[1].sent) < 7:
                await asyncio.sleep(0)
        events = [event for kind, event in sockets[1].sent if kind == "realtime"]
        assert "activity_start" in events[0]
        assert "activity_end" in events[-1]
        assert sum("activity_start" in event for event in events) == 1
        assert sum("activity_end" in event for event in events) == 1
        assert (
            b"".join(event["audio"].data for event in events if "audio" in event)
            == prefix + partial + suffix
        )
        assert configurations[1].session_resumption.handle is None
        assert session._active_input_audio == []
        generation.cancel()
    finally:
        await session.aclose()


async def test_result_of_a_call_from_retired_connection_is_context_not_tool_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _make_configured_session(
        monkeypatch,
        model="gemini-3.8-live",
        tool_behavior=types.Behavior.NON_BLOCKING,
        tool_response_scheduling=types.FunctionResponseScheduling.SILENT,
    ) as session:
        events: list[Any] = []
        session._send_client_event = events.append  # type: ignore[method-assign]
        session._start_new_generation()
        session._handle_tool_calls(
            types.LiveServerToolCall(
                function_calls=[
                    types.FunctionCall(id="reward-lookup", name="lookup_reward", args={})
                ]
            )
        )
        context = session.chat_ctx.copy()
        session._reset_chat_ctx(context)
        session._session_should_close.clear()
        session._active_session = object()  # type: ignore[assignment]
        try:
            context.items.append(
                llm.FunctionCallOutput(
                    id="reward-result",
                    call_id="reward-lookup",
                    name="lookup_reward",
                    output="A meal and a loft bunk.",
                    is_error=False,
                )
            )
            await session.update_chat_ctx(context)
            assert not session.capabilities.auto_tool_reply_generation
            assert len(events) == 1
            assert isinstance(events[0], types.LiveClientContent)
            assert events[0].turn_complete is False
            assert "A meal and a loft bunk." in events[0].turns[0].parts[0].text
            assert "already completed" in events[0].turns[0].parts[0].text
        finally:
            session._active_session = None


async def test_agent_session_finishes_tool_reply_after_native_speech_rejection_and_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    from google.genai.live import AsyncLive

    from livekit.agents import Agent, AgentSession, function_tool

    from .fake_io import FakeAudioOutput, FakeTextOutput

    sockets = []
    second_connected = asyncio.Event()

    class Socket(_FakeLiveSession):
        def __init__(self, number):
            super().__init__()
            self.number = number
            self.incoming = asyncio.Queue()

        async def send_client_content(self, *, turns=None, turn_complete=True):
            self.sent.append(("content", turns or []))
            if not turn_complete:
                return
            text = "Ten gold coins." if self.number == 1 else "A meal and a loft bunk."
            self.incoming.put_nowait(
                types.LiveServerMessage(
                    server_content=_audio_content(
                        output_transcription=types.Transcription(text=text),
                        turn_complete=self.number != 1,
                    )
                )
            )
            if self.number == 1:
                self.incoming.put_nowait(
                    types.LiveServerMessage(
                        tool_call=types.LiveServerToolCall(
                            function_calls=[
                                types.FunctionCall(id="fixed-lookup", name="lookup_reward", args={})
                            ]
                        )
                    )
                )

        async def send_tool_response(self, *, function_responses):
            raise AssertionError("The replacement socket never issued that old call")

        async def receive(self):
            while not self._closed.is_set():
                yield await self.incoming.get()

    @asynccontextmanager
    async def connect(self, **kwargs):
        socket = Socket(len(sockets) + 1)
        sockets.append(socket)
        assert len(sockets) <= 2
        if len(sockets) == 2:
            second_connected.set()
        yield socket

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(AsyncLive, "connect", connect)
    model = RealtimeModel(
        model="gemini-3.8-live",
        tool_behavior=types.Behavior.NON_BLOCKING,
        tool_response_scheduling=types.FunctionResponseScheduling.SILENT,
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
        ),
    )
    calls = []

    class CheckedAgent(Agent):
        @function_tool()
        async def lookup_reward(self) -> str:
            """Look up the already-earned reward, without changing it."""
            calls.append("lookup_reward")
            await second_connected.wait()
            return "A meal and a loft bunk."

        async def realtime_output_node(self, message, model_settings):
            text = "".join([str(chunk) async for chunk in message.text_stream])
            frames = [frame async for frame in message.audio_stream]
            if text == "Ten gold coins.":
                session._activity._rt_session.truncate(
                    message_id=message.message_id,
                    modalities=["audio", "text"],
                    audio_end_ms=0,
                    audio_transcript="",
                )
                return None

            async def audio():
                for frame in frames:
                    yield frame

            async def words():
                yield text

            return replace(message, audio_stream=audio(), text_stream=words())

    try:
        async with AgentSession(llm=model, turn_handling={"turn_detection": "manual"}) as session:
            text_output = FakeTextOutput()
            session.output.audio, session.output.transcription = FakeAudioOutput(), text_output
            await session.start(CheckedAgent(instructions="Use the fixed reward lookup."))
            handle = session.generate_reply(user_input="What did I earn?")
            await asyncio.wait_for(handle, 3)
            await asyncio.wait_for(session.wait_for_idle(), 3)
            assert calls == ["lookup_reward"]
            assert len(sockets) == 2
            assert "Ten gold coins." not in text_output._messages
            assert "A meal and a loft bunk." in text_output._messages
            assert any(
                "A meal and a loft bunk." in text
                for texts in _texts(sockets[1].sent)
                for text in texts
            )
            assert not any(
                message.text_content == "Ten gold coins." for message in session.history.messages()
            )
    finally:
        await model.aclose()


async def test_resumed_session_skips_chat_ctx_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """The server restored the conversation from the handle; replaying it would duplicate it."""
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="user", content="hello")

    async with _connected_session(monkeypatch, handle="resume-1", known=ctx, pending=ctx) as (
        session,
        fake,
    ):
        assert fake.sent == []
        assert session._pending_chat_ctx is None
        assert [m.text_content for m in session.chat_ctx.messages()] == ["hello"]


async def test_resumed_session_sends_only_the_disconnected_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Items appended during the restart are new to the resumed session; the rest is not."""
    known = llm.ChatContext.empty()
    known.add_message(role="user", content="hello")
    updated = known.copy()
    updated.add_message(role="user", content="one more thing")

    async with _connected_session(monkeypatch, handle="resume-1", known=known, pending=updated) as (
        session,
        fake,
    ):
        assert _texts(fake.sent) == [["one more thing"]]
        assert [m.text_content for m in session.chat_ctx.messages()] == [
            "hello",
            "one more thing",
        ]


async def test_resumed_session_delivers_the_tool_result_from_the_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resumed session still holds the call open, so the result produced meanwhile answers it."""
    known = llm.ChatContext.empty()
    known.add_message(role="user", content="book it")
    known.items.append(llm.FunctionCall(call_id="call-1", name="book", arguments="{}"))
    updated = known.copy()
    updated.items.append(
        llm.FunctionCallOutput(call_id="call-1", name="book", output="done", is_error=False)
    )

    async with _connected_session(monkeypatch, handle="resume-1", known=known, pending=updated) as (
        _,
        fake,
    ):
        assert [kind for kind, _ in fake.sent] == ["tool_response"]
        responses = fake.sent[0][1]
        assert [r.id for r in responses] == ["call-1"]  # type: ignore[attr-defined]


async def test_resumed_session_resends_what_the_handle_missed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A message synced after the last handle is not in the server snapshot; resend it."""
    known = llm.ChatContext.empty()
    known.add_message(role="user", content="hello")
    later = known.copy()
    later.add_message(role="user", content="sent before the socket dropped")

    async with _connected_session(
        monkeypatch, handle="resume-1", known=known, sent_after_handle=later
    ) as (session, fake):
        assert _texts(fake.sent) == [["sent before the socket dropped"]]
        assert session._pending_chat_ctx is None


async def test_session_handle_holds_what_the_old_socket_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini 3.8's handle restores everything consumed, even after it arrived; resending
    that would duplicate it in the resumed session (JA-1519, measured live)."""
    known = llm.ChatContext.empty()
    known.add_message(role="user", content="hello")
    later = known.copy()
    later.add_message(role="user", content="sent before the socket dropped")

    async with _connected_session(
        monkeypatch,
        handle="resume-1",
        known=known,
        sent_after_handle=later,
        model="gemini-3.8-live",
    ) as (session, fake):
        assert fake.sent == []
        assert [m.text_content for m in session.chat_ctx.messages()] == [
            "hello",
            "sent before the socket dropped",
        ]


async def test_session_handle_resends_only_what_never_left(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Items queued but not sent before a restart are new to the resumed 3.8 session."""
    known = llm.ChatContext.empty()
    known.add_message(role="user", content="hello")
    later = known.copy()
    later.add_message(role="user", content="sent before the socket dropped")
    never_sent = llm.ChatContext.empty()
    never_sent.add_message(role="user", content="queued when it dropped")
    later.items.extend(never_sent.items)
    updated = later.copy()
    updated.add_message(role="user", content="arrived during the restart")

    async with _connected_session(
        monkeypatch,
        handle="resume-1",
        known=known,
        sent_after_handle=later,
        unsent=never_sent,
        pending=updated,
        model="gemini-3.8-live",
    ) as (_, fake):
        assert _texts(fake.sent) == [["queued when it dropped", "arrived during the restart"]]


async def test_caller_provided_handle_adopts_the_history_without_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With a handle from the constructor the baseline is unknown; the history is the server's."""
    history = llm.ChatContext.empty()
    history.add_message(role="user", content="from the previous process")
    history.add_message(role="assistant", content="noted")

    async with _connected_session(
        monkeypatch, handle="resume-1", pending=history, caller_handle=True
    ) as (session, fake):
        assert fake.sent == []
        assert [m.text_content for m in session.chat_ctx.messages()] == [
            "from the previous process",
            "noted",
        ]


@llm.function_tool
async def _restart_tool() -> str:
    """Any new tool makes update_tools restart the socket."""
    return ""


async def test_handle_does_not_claim_a_queued_but_unsent_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A handle that lands while a diff is still queued must not cover it; a restart re-sends it."""
    from google.genai.live import AsyncLive

    known = llm.ChatContext.empty()
    known.add_message(role="user", content="hello")
    updated = known.copy()
    updated.add_message(role="user", content="queued behind the handle")

    class _GatedSession(_FakeLiveSession):
        """First socket: the connect-time handle arrives while the send is still blocked."""

        def __init__(self) -> None:
            super().__init__()
            self.gate = asyncio.Event()

        async def send_client_content(self, *, turns: object, turn_complete: bool) -> None:
            await self.gate.wait()
            await super().send_client_content(turns=turns, turn_complete=turn_complete)

        async def receive(self) -> AsyncIterator[types.LiveServerMessage]:
            yield types.LiveServerMessage(
                session_resumption_update=types.LiveServerSessionResumptionUpdate(
                    new_handle="resume-2", resumable=True
                )
            )
            await self._closed.wait()

    sockets: list[_FakeLiveSession] = [_GatedSession(), _FakeLiveSession()]
    opened: list[_FakeLiveSession] = []

    @asynccontextmanager
    async def _connect(self: AsyncLive, **kwargs: object) -> AsyncIterator[_FakeLiveSession]:
        fake = sockets[len(opened)]
        opened.append(fake)
        yield fake

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(AsyncLive, "connect", _connect)
    session = RealtimeModel().session()
    session._session_resumption_handle = "resume-1"
    session._resumption_chat_ctx = known
    session._chat_ctx = known
    await session.update_chat_ctx(updated)
    try:
        while session._session_resumption_handle != "resume-2":
            await asyncio.sleep(0.01)
        # the diff is queued but its send is blocked, so the new handle must not cover it
        assert [m.text_content for m in session._resumption_chat_ctx.messages()] == ["hello"]

        # restart before the send completes: the channel drain drops the queued diff
        await session.update_tools([_restart_tool])
        while len(opened) < 2:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        assert _texts(opened[1].sent) == [["queued behind the handle"]]
    finally:
        await session.aclose()


async def test_failed_send_with_a_queued_update_replays_each_item_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An error restart drops the queue; the resume diff must be the only thing that re-sends."""
    from google.genai.live import AsyncLive

    known = llm.ChatContext.empty()
    known.add_message(role="user", content="hello")
    first = known.copy()
    first.add_message(role="user", content="first update")

    class _FailingSession(_FakeLiveSession):
        """Blocks the first send until released, then fails it."""

        def __init__(self) -> None:
            super().__init__()
            self.blocked = asyncio.Event()
            self.release = asyncio.Event()

        async def send_client_content(self, *, turns: object, turn_complete: bool) -> None:
            self.blocked.set()
            await self.release.wait()
            raise RuntimeError("socket gone")

    sockets: list[_FakeLiveSession] = [_FailingSession(), _FakeLiveSession()]
    opened: list[_FakeLiveSession] = []

    @asynccontextmanager
    async def _connect(self: AsyncLive, **kwargs: object) -> AsyncIterator[_FakeLiveSession]:
        fake = sockets[len(opened)]
        opened.append(fake)
        yield fake

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(AsyncLive, "connect", _connect)
    session = RealtimeModel().session()
    session._session_resumption_handle = "resume-1"
    session._resumption_chat_ctx = known
    session._chat_ctx = known
    await session.update_chat_ctx(first)
    try:
        failing = sockets[0]
        assert isinstance(failing, _FailingSession)
        await asyncio.wait_for(failing.blocked.wait(), timeout=2)
        # a second update queues behind the blocked send
        second = first.copy()
        second.add_message(role="user", content="second update")
        await session.update_chat_ctx(second)
        failing.release.set()

        while len(opened) < 2:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        assert opened[0].sent == []
        assert _texts(opened[1].sent) == [["first update", "second update"]]
        assert session._unsent_item_ids == set()
    finally:
        await session.aclose()


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini-3.8-live", types.Behavior.NON_BLOCKING),
        ("gemini-3.8-live-extended-thinking", types.Behavior.NON_BLOCKING),
        ("gemini-3.1-flash-live-preview", None),
        ("gemini-2.5-flash-native-audio-preview-12-2025", None),
    ],
)
def test_tool_behavior_default_follows_the_model(
    monkeypatch: pytest.MonkeyPatch, model: str, expected: types.Behavior | None
) -> None:
    """Models that are async by default must be declared async, or SILENT is never claimed."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    behavior = RealtimeModel(model=model)._opts.tool_behavior
    assert (behavior if utils.is_given(behavior) else None) == expected


def test_explicit_tool_behavior_wins_over_the_model_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    model = RealtimeModel(model="gemini-3.8-live", tool_behavior=types.Behavior.BLOCKING)
    assert model._opts.tool_behavior == types.Behavior.BLOCKING


# -- connection rotation ------------------------------------------------------------------


class _ScriptedSocket(_FakeLiveSession):
    """A live socket whose server messages (or a failure to raise) the test feeds in."""

    def __init__(self) -> None:
        super().__init__()
        self.incoming: asyncio.Queue[types.LiveServerMessage | Exception] = asyncio.Queue()

    def serve(self, message: types.LiveServerMessage | Exception) -> None:
        self.incoming.put_nowait(message)

    async def receive(self) -> AsyncIterator[types.LiveServerMessage]:
        while True:
            message = await self.incoming.get()
            if isinstance(message, Exception):
                raise message
            yield message


class _Server:
    """Hands out scripted sockets; an exception in `connects` fails that connect attempt."""

    def __init__(self, *connects: _ScriptedSocket | Exception) -> None:
        self.connects = list(connects)
        self.configs: list[types.LiveConnectConfig] = []
        self.opened: list[_ScriptedSocket] = []

    @property
    def handles(self) -> list[str | None]:
        return [c.session_resumption.handle for c in self.configs if c.session_resumption]

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from google.genai.live import AsyncLive

        @asynccontextmanager
        async def connect(live: AsyncLive, **kwargs: Any) -> AsyncIterator[_ScriptedSocket]:
            self.configs.append(kwargs["config"])
            outcome = self.connects.pop(0) if self.connects else _ScriptedSocket()
            if isinstance(outcome, Exception):
                raise outcome
            self.opened.append(outcome)
            yield outcome

        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
        monkeypatch.setattr(AsyncLive, "connect", connect)


def _handle_update(handle: str) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        session_resumption_update=types.LiveServerSessionResumptionUpdate(
            new_handle=handle, resumable=True
        )
    )


def _speech(**kwargs: object) -> types.LiveServerMessage:
    return types.LiveServerMessage(server_content=_audio_content(**kwargs))


def _internal_error() -> Exception:
    from google.genai import errors

    return errors.APIError(1011, {"message": "Internal error"})


async def _eventually(condition: Any, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


@asynccontextmanager
async def _rotating_session(
    server: _Server, monkeypatch: pytest.MonkeyPatch, **options: Any
) -> AsyncIterator[tuple[RealtimeSession, list[llm.RealtimeModelError]]]:
    from livekit.agents import APIConnectOptions

    server.install(monkeypatch)
    # Delphi's current setting: no retries configured at all
    options.setdefault("conn_options", APIConnectOptions(max_retry=0, timeout=5))
    session = RealtimeModel(model="gemini-3.8-live", **options).session()
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)
    try:
        yield session, errors
    finally:
        await session.aclose()


async def test_go_away_lets_the_reply_finish_then_resumes_on_a_new_connection(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    first = _ScriptedSocket()
    server = _Server(first)
    async with _rotating_session(server, monkeypatch) as (session, errors):
        generations: list[llm.GenerationCreatedEvent] = []
        session.on("generation_created", generations.append)
        await _eventually(lambda: server.opened)
        first.serve(_handle_update("handle-1"))
        first.serve(_speech())
        await _eventually(lambda: generations)
        drained = asyncio.create_task(_drain_generation(generations[0]))

        with caplog.at_level(logging.INFO, logger="livekit.plugins.google"):
            first.serve(types.LiveServerMessage(go_away=types.LiveServerGoAway(time_left="30s")))
            await asyncio.sleep(0.3)
            assert len(server.opened) == 1, "GoAway must not cut the reply in progress"

            first.serve(_speech(generation_complete=True))
            first.serve(
                types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))
            )
            await _eventually(lambda: len(server.opened) == 2)
            await _eventually(
                lambda: any(r.getMessage() == "Gemini connection rotated" for r in caplog.records)
            )

        _, audio_frames, _ = await asyncio.wait_for(drained, 1)
        assert audio_frames == 2
        assert server.handles == [None, "handle-1"]
        assert errors == []
        rotated = [r for r in caplog.records if r.getMessage() == "Gemini connection rotated"]
        assert [(r.reason, r.resumed) for r in rotated] == [("go_away", True)]  # type: ignore[attr-defined]


async def test_go_away_rotates_by_its_deadline_even_when_the_reply_never_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _ScriptedSocket()
    server = _Server(first)
    async with _rotating_session(server, monkeypatch) as (session, errors):
        await _eventually(lambda: server.opened)
        first.serve(_handle_update("handle-1"))
        first.serve(_speech())
        first.serve(types.LiveServerMessage(go_away=types.LiveServerGoAway(time_left="1.7s")))
        # 1.7s minus the 1.5s safety margin
        await _eventually(lambda: len(server.opened) == 2, timeout=1.0)
        assert server.handles == [None, "handle-1"]
        assert errors == []


async def test_proactive_rotation_waits_for_a_turn_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _ScriptedSocket()
    server = _Server(first)
    async with _rotating_session(server, monkeypatch, max_connection_age=0.2) as (
        session,
        errors,
    ):
        await _eventually(lambda: server.opened)
        first.serve(_handle_update("handle-1"))
        first.serve(_speech())
        await asyncio.sleep(0.5)
        assert len(server.opened) == 1, "an aged connection must not rotate mid-reply"

        first.serve(_speech(generation_complete=True, turn_complete=True))
        await _eventually(lambda: len(server.opened) == 2)
        assert server.handles == [None, "handle-1"]
        assert errors == []


async def test_pending_reply_request_waits_out_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generate_reply before the server answers is a turn in progress, not a boundary."""
    first = _ScriptedSocket()
    server = _Server(first)
    async with _rotating_session(server, monkeypatch, max_connection_age=0.2) as (
        session,
        errors,
    ):
        await _eventually(lambda: server.opened)
        reply = session.generate_reply()
        await asyncio.sleep(0.5)
        assert len(server.opened) == 1

        first.serve(_speech(generation_complete=True, turn_complete=True))
        await asyncio.wait_for(reply, 1)
        await _eventually(lambda: len(server.opened) == 2)
        assert errors == []


async def test_internal_error_on_a_live_connection_reconnects_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _ScriptedSocket()
    server = _Server(first)
    async with _rotating_session(server, monkeypatch) as (session, errors):
        await _eventually(lambda: server.opened)
        first.serve(_handle_update("handle-1"))
        first.serve(_internal_error())
        await _eventually(lambda: len(server.opened) == 2)
        await asyncio.sleep(0.1)
        assert server.handles == [None, "handle-1"]
        assert errors == []


async def test_reply_pending_when_the_socket_drops_is_requested_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _ScriptedSocket(), _ScriptedSocket()
    server = _Server(first, second)
    async with _rotating_session(server, monkeypatch) as (session, errors):
        await _eventually(lambda: server.opened)
        first.serve(_handle_update("handle-1"))
        reply = session.generate_reply(instructions="Describe the gate.")
        await _eventually(lambda: first.sent)
        first.serve(_internal_error())

        await _eventually(lambda: second.sent)
        assert _texts(second.sent) == [
            ["Application response instructions (not dialogue):\nDescribe the gate."]
        ]
        second.serve(_speech(generation_complete=True, turn_complete=True))
        created = await asyncio.wait_for(reply, 1)
        assert created.user_initiated
        assert errors == []


async def test_rejected_resumption_handle_reconnects_fresh_with_the_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.genai import errors as genai_errors

    """A handle refused twice is dropped; the fresh connection replays the history."""
    monkeypatch.setattr(realtime_api, "RESUME_SETTLE_DELAY", 0.05)
    history = llm.ChatContext.empty()
    history.add_message(role="user", content="I open the gate.")
    history.add_message(role="assistant", content="It creaks open.")
    fresh = _ScriptedSocket()
    refused = genai_errors.APIError(1008, {"message": "invalid handle"})
    server = _Server(refused, refused, fresh)
    async with _rotating_session(
        server,
        monkeypatch,
        session_resumption=types.SessionResumptionConfig(handle="expired"),
    ) as (session, errors):
        await session.update_chat_ctx(history)
        await _eventually(lambda: fresh.sent)
        assert server.handles == ["expired", "expired", None]
        assert _texts(fresh.sent) == [["I open the gate.", "It creaks open."]]
        assert not [e for e in errors if not e.recoverable]


async def test_a_handle_refused_once_is_retried_before_replaying_history(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Gemini refuses a handle (1011) while it settles the last streamed audio; the same
    handle resumes a moment later (JA-1519, measured live on gemini-3.8-live)."""
    monkeypatch.setattr(realtime_api, "RESUME_SETTLE_DELAY", 0.05)
    first, resumed = _ScriptedSocket(), _ScriptedSocket()
    server = _Server(first, _internal_error(), resumed)
    async with _rotating_session(server, monkeypatch, max_connection_age=0.2) as (
        session,
        errors,
    ):
        with caplog.at_level(logging.INFO, logger="livekit.plugins.google"):
            await _eventually(lambda: server.opened)
            first.serve(_handle_update("handle-1"))
            await _eventually(lambda: len(server.opened) == 2)
            await _eventually(
                lambda: any(r.getMessage() == "Gemini connection rotated" for r in caplog.records)
            )
        assert server.handles == [None, "handle-1", "handle-1"]
        assert resumed.sent == []
        rotated = [r for r in caplog.records if r.getMessage() == "Gemini connection rotated"]
        assert [(r.reason, r.resumed) for r in rotated] == [("max_age", True)]  # type: ignore[attr-defined]
        assert errors == []


async def test_resume_waits_for_streamed_audio_to_settle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resume right after the last streamed frame is refused, so the connect waits."""
    from livekit import rtc
    from livekit.plugins.google.realtime.realtime_api import INPUT_AUDIO_SAMPLE_RATE

    monkeypatch.setattr(realtime_api, "RESUME_SETTLE_DELAY", 0.3)
    first = _ScriptedSocket()
    server = _Server(first)
    connected_at: list[float] = []
    original_install = server.install

    def install(mp: pytest.MonkeyPatch) -> None:
        original_install(mp)
        from google.genai.live import AsyncLive

        inner = AsyncLive.connect

        @asynccontextmanager
        async def timed(live: Any, **kwargs: Any) -> AsyncIterator[Any]:
            connected_at.append(asyncio.get_running_loop().time())
            async with inner(live, **kwargs) as socket:
                yield socket

        mp.setattr(AsyncLive, "connect", timed)

    server.install = install  # type: ignore[method-assign]
    async with _rotating_session(server, monkeypatch) as (session, errors):
        await _eventually(lambda: server.opened)
        first.serve(_handle_update("handle-1"))
        data = b"\x00" * (INPUT_AUDIO_SAMPLE_RATE // 20 * 2)
        session.push_audio(rtc.AudioFrame(data, INPUT_AUDIO_SAMPLE_RATE, 1, len(data) // 2))
        await _eventually(lambda: first.sent)
        sent_at = asyncio.get_running_loop().time()
        first.serve(_internal_error())
        await _eventually(lambda: len(server.opened) == 2)
        assert server.handles == [None, "handle-1"]
        assert connected_at[1] - sent_at >= 0.25
        assert errors == []


async def test_a_second_failure_after_the_reconnect_budget_is_still_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reconnect is bounded: a live drop followed by a failed reconnect ends the session."""
    first = _ScriptedSocket()
    server = _Server(first, _internal_error())
    async with _rotating_session(server, monkeypatch) as (session, errors):
        await _eventually(lambda: server.opened)
        first.serve(_internal_error())
        await _eventually(lambda: errors)
        assert [e.recoverable for e in errors] == [False]
        assert len(server.configs) == 2

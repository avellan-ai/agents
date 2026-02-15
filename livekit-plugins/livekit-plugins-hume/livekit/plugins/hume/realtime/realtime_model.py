# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
import weakref
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from hume.client import AsyncHumeClient
from hume.empathic_voice.chat.socket_client import AsyncChatSocketClient
from hume.empathic_voice.types import (
    AssistantEnd,
    AssistantInput,
    AssistantMessage,
    AssistantProsody,
    AudioInput,
    AudioOutput,
    ChatMetadata,
    ConnectSessionSettings,
    Context,
    PauseAssistantMessage,
    ResumeAssistantMessage,
    SessionSettings,
    SubscribeEvent,
    Tool as HumeTool,
    ToolCallMessage,
    ToolErrorMessage,
    ToolResponseMessage,
    UserInput,
    UserInterruption,
    UserMessage,
    WebSocketError,
)
from livekit import rtc
from livekit.agents import APIError, llm, utils
from livekit.agents.llm.utils import compute_chat_ctx_diff
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from ..log import logger
from .audio import StreamingWavDecoder

INPUT_SAMPLE_RATE = 16000
NUM_CHANNELS = 1
PENDING_GENERATION_TIMEOUT_SECONDS = 5.0

_PublishEvent = (
    AudioInput
    | SessionSettings
    | UserInput
    | AssistantInput
    | ToolResponseMessage
    | ToolErrorMessage
    | PauseAssistantMessage
    | ResumeAssistantMessage
)


@dataclass
class _RealtimeOptions:
    model: str
    api_key: str | None
    access_token: str | None
    base_url: str | None
    config_id: str | None
    config_version: int | None
    allow_connection: bool | None
    resumed_chat_group_id: str | None
    verbose_transcription: bool | None
    session_settings: ConnectSessionSettings | None
    conn_options: APIConnectOptions


@dataclass
class _MessageGenerationState:
    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    modalities: asyncio.Future[list[Literal["text", "audio"]]]
    text_seen: bool = False
    audio_seen: bool = False

    def set_text_seen(self) -> None:
        self.text_seen = True
        if not self.modalities.done():
            if self.audio_seen:
                self.modalities.set_result(["audio", "text"])
            else:
                self.modalities.set_result(["text"])

    def set_audio_seen(self) -> None:
        self.audio_seen = True
        if not self.modalities.done():
            if self.text_seen:
                self.modalities.set_result(["audio", "text"])
            else:
                self.modalities.set_result(["audio"])


@dataclass
class _ResponseGeneration:
    generation_key: str
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    message_state: _MessageGenerationState
    created_timestamp: float = field(default_factory=time.time)
    first_token_timestamp: float | None = None
    user_initiated: bool = False
    done: bool = False


class RealtimeModel(llm.RealtimeModel):
    def __init__(
        self,
        *,
        model: str = "evi-4-mini",
        api_key: str | None = None,
        access_token: str | None = None,
        base_url: str | None = None,
        config_id: str | None = None,
        config_version: int | None = None,
        allow_connection: bool | None = None,
        resumed_chat_group_id: str | None = None,
        verbose_transcription: bool | None = None,
        session_settings: ConnectSessionSettings | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=True,
            )
        )

        resolved_api_key = api_key or os.environ.get("HUME_API_KEY")
        if not resolved_api_key and not access_token:
            raise ValueError(
                "Hume credentials are required. Provide api_key/HUME_API_KEY or access_token."
            )

        self._opts = _RealtimeOptions(
            model=model,
            api_key=resolved_api_key,
            access_token=access_token,
            base_url=base_url,
            config_id=config_id,
            config_version=config_version,
            allow_connection=allow_connection,
            resumed_chat_group_id=resumed_chat_group_id,
            verbose_transcription=verbose_transcription,
            session_settings=session_settings,
            conn_options=conn_options,
        )
        self._sessions = weakref.WeakSet[RealtimeSession]()
        self._label = f"hume-{model}"

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Hume"

    def session(self) -> RealtimeSession:
        sess = RealtimeSession(realtime_model=self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        await asyncio.gather(*(sess.aclose() for sess in list(self._sessions)), return_exceptions=True)


class RealtimeSession(
    llm.RealtimeSession[Literal["hume_server_event_received", "hume_client_event_queued"]]
):
    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model = realtime_model
        self._opts = realtime_model._opts

        self._chat_ctx = llm.ChatContext.empty()
        self._tools = llm.ToolContext.empty()
        self._tool_response_required: dict[str, bool] = {}

        self._pending_generation_futs = deque[asyncio.Future[llm.GenerationCreatedEvent]]()
        self._pending_generation_timeout_handles: dict[
            asyncio.Future[llm.GenerationCreatedEvent], asyncio.TimerHandle
        ] = {}

        self._staged_publish_events: list[_PublishEvent] = []
        self._msg_ch: utils.aio.Chan[_PublishEvent] = utils.aio.Chan()

        self._client = AsyncHumeClient(api_key=self._opts.api_key, base_url=self._opts.base_url)
        self._chat_socket: AsyncChatSocketClient | None = None
        self._connect_attempts = 0

        self._generation_seq = 0
        self._active_generation_key: str | None = None
        self._active_generation: _ResponseGeneration | None = None
        self._audio_decoders: dict[str, StreamingWavDecoder] = {}

        self._current_user_turn_item_id: str | None = None
        self._input_resampler: rtc.AudioResampler | None = None
        input_rate = (
            self._opts.session_settings.audio.sample_rate
            if self._opts.session_settings and self._opts.session_settings.audio
            and self._opts.session_settings.audio.sample_rate
            else INPUT_SAMPLE_RATE
        )
        self._input_sample_rate = input_rate
        self._input_num_channels = (
            self._opts.session_settings.audio.channels
            if self._opts.session_settings and self._opts.session_settings.audio
            and self._opts.session_settings.audio.channels
            else NUM_CHANNELS
        )
        self._input_bstream = utils.audio.AudioByteStream(
            sample_rate=self._input_sample_rate,
            num_channels=self._input_num_channels,
            samples_per_channel=max(1, self._input_sample_rate // 50),
        )

        self._chat_id: str | None = None
        self._chat_group_id: str | None = self._opts.resumed_chat_group_id

        self._closed = False
        self._main_task = asyncio.create_task(self._main_loop(), name="HumeRealtimeSession.main")

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools

    async def update_instructions(self, instructions: str) -> None:
        self._enqueue_publish(SessionSettings(system_prompt=instructions))

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        diff = compute_chat_ctx_diff(self._chat_ctx, chat_ctx)
        for _ in diff.to_remove:
            logger.warning("Hume realtime: removal from remote chat context is not supported.")

        id_to_item = {item.id: item for item in chat_ctx.items}
        for _, item_id in [*diff.to_create, *diff.to_update]:
            if (item := id_to_item.get(item_id)) is None:
                continue
            self._stage_chat_item(item)

        self._chat_ctx = chat_ctx.copy()

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        self._tools.update_tools(tools)
        parsed_tools = self._tools.parse_function_tools("openai.responses")
        hume_tools: list[HumeTool] = []
        for tool in parsed_tools:
            if tool.get("type") != "function":
                continue

            name = cast(str, tool.get("name") or "")
            if not name:
                continue

            description = cast(str | None, tool.get("description"))
            parameters = _to_stringified_json_schema(tool.get("parameters"))
            hume_tools.append(
                HumeTool(
                    type="function",
                    name=name,
                    description=description,
                    parameters=parameters,
                )
            )

        self._enqueue_publish(SessionSettings(tools=hume_tools))

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        if is_given(tool_choice):
            logger.warning(
                "Hume realtime does not currently support tool_choice updates; ignoring.",
                extra={"tool_choice": tool_choice},
            )

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        for resampled_frame in self._resample_audio(frame):
            for chunk in self._input_bstream.push(bytes(resampled_frame.data)):
                payload = base64.b64encode(bytes(chunk.data)).decode("ascii")
                self._enqueue_publish(AudioInput(data=payload))

    def push_video(self, frame: rtc.VideoFrame) -> None:
        logger.warning("video is not supported by Hume realtime.")

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        fut = asyncio.Future[llm.GenerationCreatedEvent]()
        self._pending_generation_futs.append(fut)

        def _on_timeout() -> None:
            if not fut.done():
                fut.set_exception(
                    llm.RealtimeError("generate_reply timed out waiting for generation_created event.")
                )
                with contextlib.suppress(ValueError):
                    self._pending_generation_futs.remove(fut)

        handle = asyncio.get_event_loop().call_later(PENDING_GENERATION_TIMEOUT_SECONDS, _on_timeout)
        self._pending_generation_timeout_handles[fut] = handle

        def _clear_timeout(_: asyncio.Future[llm.GenerationCreatedEvent]) -> None:
            timeout_handle = self._pending_generation_timeout_handles.pop(fut, None)
            if timeout_handle:
                timeout_handle.cancel()

        fut.add_done_callback(_clear_timeout)

        if is_given(instructions):
            self._enqueue_publish(SessionSettings(system_prompt=instructions))

        if self._staged_publish_events:
            for event in self._staged_publish_events:
                self._enqueue_publish(event)
            self._staged_publish_events.clear()

        return fut

    def commit_audio(self) -> None:
        logger.warning("commit_audio is not supported by Hume realtime.")

    def clear_audio(self) -> None:
        self._input_bstream.clear()
        logger.warning("clear_audio is not supported by Hume realtime.")

    def interrupt(self) -> None:
        logger.info(
            "Hume realtime interruption is handled locally in LiveKit; no provider cancel sent by default."
        )

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        logger.warning("truncate is not supported by Hume realtime.")

    async def aclose(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._msg_ch.close()
        await utils.aio.cancel_and_wait(self._main_task)
        self._fail_pending_generation_futures(
            llm.RealtimeError("Session closed before generation was created.")
        )
        self._close_generation(self._active_generation)

    def _enqueue_publish(self, event: _PublishEvent) -> None:
        if self._msg_ch.closed:
            return
        self.emit("hume_client_event_queued", event)
        self._msg_ch.send_nowait(event)

    def _stage_chat_item(self, item: llm.ChatItem) -> None:
        if item.type == "message":
            if item.role == "user":
                text = _message_text(item)
                if text:
                    self._staged_publish_events.append(UserInput(text=text))
            elif item.role in ("system", "developer"):
                text = _message_text(item)
                if text:
                    self._enqueue_publish(
                        SessionSettings(system_prompt=text, context=Context(text=text, type="temporary"))
                    )
            return

        if item.type == "function_call_output":
            if self._tool_response_required.get(item.call_id, True) is False:
                return

            if item.is_error:
                self._staged_publish_events.append(
                    ToolErrorMessage(
                        tool_call_id=item.call_id,
                        error=item.output,
                        content=item.output,
                        level="warn",
                        tool_type="function",
                    )
                )
            else:
                self._staged_publish_events.append(
                    ToolResponseMessage(
                        tool_call_id=item.call_id,
                        content=item.output,
                        tool_name=item.name,
                        tool_type="function",
                    )
                )

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler and frame.sample_rate != self._input_resampler._input_rate:
            self._input_resampler = None

        if self._input_resampler is None and frame.sample_rate != self._input_sample_rate:
            self._input_resampler = rtc.AudioResampler(
                frame.sample_rate,
                self._input_sample_rate,
                quality=rtc.AudioResamplerQuality.MEDIUM,
            )

        if self._input_resampler:
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    async def _main_loop(self) -> None:
        retries = 0
        while not self._msg_ch.closed:
            try:
                async with self._client.empathic_voice.chat.connect(
                    access_token=self._opts.access_token,
                    allow_connection=self._opts.allow_connection,
                    config_id=self._opts.config_id,
                    config_version=self._opts.config_version,
                    resumed_chat_group_id=self._chat_group_id,
                    verbose_transcription=self._opts.verbose_transcription,
                    api_key=None,
                    session_settings=self._opts.session_settings,
                ) as socket:
                    if self._connect_attempts > 0:
                        self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())
                        self._fail_pending_generation_futures(
                            llm.RealtimeError("Session reconnected before generation was created.")
                        )

                    self._connect_attempts += 1
                    retries = 0
                    self._chat_socket = socket

                    send_task = asyncio.create_task(self._send_loop(socket), name="HumeRealtime.send")
                    recv_task = asyncio.create_task(self._recv_loop(socket), name="HumeRealtime.recv")

                    done, pending = await asyncio.wait(
                        [send_task, recv_task], return_when=asyncio.FIRST_EXCEPTION
                    )
                    for task in pending:
                        task.cancel()
                    await utils.aio.cancel_and_wait(*pending)

                    for task in done:
                        exc = task.exception()
                        if exc:
                            raise exc

            except asyncio.CancelledError:
                raise
            except Exception as e:
                recoverable = retries < self._opts.conn_options.max_retry and not self._msg_ch.closed
                self._emit_error(error=e, recoverable=recoverable)
                self._fail_pending_generation_futures(
                    llm.RealtimeError("Connection reset before generation was created.")
                )

                if not recoverable:
                    break

                await asyncio.sleep(self._opts.conn_options._interval_for_retry(retries))
                retries += 1
            finally:
                self._chat_socket = None

    async def _send_loop(self, socket: AsyncChatSocketClient) -> None:
        async for event in self._msg_ch:
            await socket.send_publish(event)

    async def _recv_loop(self, socket: AsyncChatSocketClient) -> None:
        async for event in socket:
            self.emit("hume_server_event_received", event)
            self._handle_subscribe_event(event)

    def _handle_subscribe_event(self, event: SubscribeEvent) -> None:
        if isinstance(event, ChatMetadata):
            self._chat_id = event.chat_id
            self._chat_group_id = event.chat_group_id
            return

        if isinstance(event, UserInterruption):
            self._current_user_turn_item_id = None
            self.emit("input_speech_started", llm.InputSpeechStartedEvent())
            return

        if isinstance(event, UserMessage):
            self._handle_user_message(event)
            return

        if isinstance(event, AssistantMessage):
            self._handle_assistant_message(event)
            return

        if isinstance(event, AudioOutput):
            self._handle_audio_output(event)
            return

        if isinstance(event, AssistantEnd):
            self._handle_assistant_end()
            return

        if isinstance(event, ToolCallMessage):
            self._handle_tool_call(event)
            return

        if isinstance(event, WebSocketError):
            self._emit_error(
                error=APIError(message=event.message, body={"code": event.code, "slug": event.slug}),
                recoverable=True,
            )
            return

        if isinstance(event, AssistantProsody):
            # Prosody doesn't participate in turn orchestration.
            return

        # Tool responses/errors and session settings events are informational from server side.
        if isinstance(event, (ToolResponseMessage, ToolErrorMessage, SessionSettings)):
            return

        logger.warning("Unhandled Hume realtime event.", extra={"event_type": type(event).__name__})

    def _handle_user_message(self, event: UserMessage) -> None:
        transcript = event.message.content or ""
        item_id = self._current_user_turn_item_id
        if item_id is None:
            item_id = utils.shortuuid("item_")
            self._current_user_turn_item_id = item_id

        if event.interim:
            self.emit("input_audio_transcription_completed", _transcription(item_id, transcript, False))
            return

        self.emit("input_speech_stopped", llm.InputSpeechStoppedEvent(user_transcription_enabled=True))
        self.emit("input_audio_transcription_completed", _transcription(item_id, transcript, True))
        self._current_user_turn_item_id = None

    def _handle_assistant_message(self, event: AssistantMessage) -> None:
        generation = self._open_generation(_assistant_generation_key(event))
        text = event.message.content or ""
        if text:
            generation.message_state.set_text_seen()
            generation.message_state.text_ch.send_nowait(text)
            if generation.first_token_timestamp is None:
                generation.first_token_timestamp = time.time()

    def _handle_audio_output(self, event: AudioOutput) -> None:
        generation = self._open_generation(_audio_generation_key(event))
        decoder = self._audio_decoders.setdefault(
            generation.generation_key, StreamingWavDecoder()
        )
        chunk_bytes = base64.b64decode(event.data)
        frames = decoder.push(chunk_bytes)
        if frames:
            generation.message_state.set_audio_seen()
            if generation.first_token_timestamp is None:
                generation.first_token_timestamp = time.time()
            for frame in frames:
                generation.message_state.audio_ch.send_nowait(frame)

    def _handle_assistant_end(self) -> None:
        generation = self._active_generation
        self._close_generation(generation)

    def _handle_tool_call(self, event: ToolCallMessage) -> None:
        generation = self._open_generation(f"tool:{event.tool_call_id}")
        self._tool_response_required[event.tool_call_id] = event.response_required

        arguments = _parse_tool_call_arguments(event.parameters)
        fnc_call = llm.FunctionCall(
            call_id=event.tool_call_id,
            name=event.name,
            arguments=arguments,
            extra={
                "hume": {
                    "response_required": event.response_required,
                    "tool_type": event.tool_type,
                }
            },
        )
        generation.function_ch.send_nowait(fnc_call)

    def _open_generation(self, generation_key: str) -> _ResponseGeneration:
        if (
            self._active_generation is not None
            and not self._active_generation.done
            and self._active_generation.generation_key != generation_key
        ):
            self._close_generation(self._active_generation)

        if (
            self._active_generation is not None
            and not self._active_generation.done
            and self._active_generation.generation_key == generation_key
        ):
            return self._active_generation

        self._generation_seq += 1
        message_id = f"hume_msg_{self._generation_seq}"
        text_ch: utils.aio.Chan[str] = utils.aio.Chan()
        audio_ch: utils.aio.Chan[rtc.AudioFrame] = utils.aio.Chan()
        modalities = asyncio.Future[list[Literal["text", "audio"]]]()
        message_state = _MessageGenerationState(
            message_id=message_id,
            text_ch=text_ch,
            audio_ch=audio_ch,
            modalities=modalities,
        )

        generation = _ResponseGeneration(
            generation_key=generation_key,
            message_ch=utils.aio.Chan(),
            function_ch=utils.aio.Chan(),
            message_state=message_state,
        )
        generation.message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=message_id,
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )

        user_initiated = False
        pending_fut: asyncio.Future[llm.GenerationCreatedEvent] | None = None
        if self._pending_generation_futs:
            pending_fut = self._pending_generation_futs.popleft()
            user_initiated = True

        generation.user_initiated = user_initiated
        ev = llm.GenerationCreatedEvent(
            message_stream=generation.message_ch,
            function_stream=generation.function_ch,
            user_initiated=user_initiated,
            response_id=generation_key,
        )
        if pending_fut and not pending_fut.done():
            pending_fut.set_result(ev)

        self.emit("generation_created", ev)

        self._active_generation_key = generation_key
        self._active_generation = generation
        return generation

    def _close_generation(self, generation: _ResponseGeneration | None) -> None:
        if generation is None or generation.done:
            return

        generation.done = True
        if not generation.message_state.modalities.done():
            if generation.message_state.audio_seen:
                generation.message_state.modalities.set_result(["audio"])
            elif generation.message_state.text_seen:
                generation.message_state.modalities.set_result(["text"])
            else:
                generation.message_state.modalities.set_result(["text"])

        generation.message_state.text_ch.close()
        generation.message_state.audio_ch.close()
        generation.function_ch.close()
        generation.message_ch.close()

        duration = max(0.0, time.time() - generation.created_timestamp)
        ttft = (
            generation.first_token_timestamp - generation.created_timestamp
            if generation.first_token_timestamp is not None
            else -1.0
        )
        metrics = RealtimeModelMetrics(
            timestamp=generation.created_timestamp,
            request_id=generation.generation_key,
            ttft=ttft,
            duration=duration,
            cancelled=False,
            label=self._realtime_model.label,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            tokens_per_second=0,
            input_token_details=RealtimeModelMetrics.InputTokenDetails(
                audio_tokens=0,
                text_tokens=0,
                image_tokens=0,
                cached_tokens=0,
                cached_tokens_details=RealtimeModelMetrics.CachedTokenDetails(
                    audio_tokens=0,
                    text_tokens=0,
                    image_tokens=0,
                ),
            ),
            output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                text_tokens=0,
                audio_tokens=0,
                image_tokens=0,
            ),
            metadata=Metadata(
                model_name=self._realtime_model.model,
                model_provider=self._realtime_model.provider,
            ),
        )
        self.emit("metrics_collected", metrics)

        self._audio_decoders.pop(generation.generation_key, None)
        if self._active_generation is generation:
            self._active_generation = None
            self._active_generation_key = None

    def _emit_error(self, *, error: Exception, recoverable: bool) -> None:
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._realtime_model.label,
                error=error,
                recoverable=recoverable,
            ),
        )

    def _fail_pending_generation_futures(self, exc: Exception) -> None:
        while self._pending_generation_futs:
            fut = self._pending_generation_futs.popleft()
            if not fut.done():
                fut.set_exception(exc)

        for handle in self._pending_generation_timeout_handles.values():
            handle.cancel()
        self._pending_generation_timeout_handles.clear()


def _message_text(msg: llm.ChatMessage) -> str:
    parts = [part for part in msg.content if isinstance(part, str)]
    return "\n".join(parts).strip()


def _to_stringified_json_schema(parameters: Any) -> str:
    if isinstance(parameters, str):
        return parameters
    if parameters is None:
        return json.dumps({"type": "object", "properties": {}, "required": []})
    return json.dumps(parameters)


def _parse_tool_call_arguments(raw: str) -> str:
    # Hume currently exposes `parameters` as a string field. For compatibility, we normalize
    # any valid JSON payload into compact JSON and otherwise forward the raw string.
    with contextlib.suppress(Exception):
        parsed = json.loads(raw)
        return json.dumps(parsed, separators=(",", ":"), sort_keys=True)
    return raw


def _transcription(item_id: str, transcript: str, is_final: bool) -> llm.InputTranscriptionCompleted:
    return llm.InputTranscriptionCompleted(item_id=item_id, transcript=transcript, is_final=is_final)


def _assistant_generation_key(event: AssistantMessage) -> str:
    if event.id:
        return f"assistant:{event.id}"
    return f"assistant:synthetic:{utils.shortuuid()}"


def _audio_generation_key(event: AudioOutput) -> str:
    return f"audio:{event.id}" if event.id else f"audio:synthetic:{utils.shortuuid()}"

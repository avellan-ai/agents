# LiveKit Agents + Hume EVI 4-mini Integration Plan (No CLM)

Last updated: 2026-02-15  
Branch: `feature/hume-evi4-mini-realtime`  
Scope decision: start with Hume-managed models only (no custom language model).

## 1) Scope Snapshot

### Goals
- Add a first-class Hume EVI speech-to-speech realtime integration to `livekit-plugins-hume`.
- Support EVI `4-mini` and expose runtime/session/config features needed for production voice agents.
- Preserve existing Hume TTS plugin behavior.
- Provide examples, tests, and rollout guidance.

### Non-goals (for this iteration)
- No CLM (Custom Language Model) endpoint integration.
- No OpenRouter/Baseten/Fireworks proxy work.
- No changes to LiveKit core abstractions unless unavoidable for provider compatibility.

## 2) Definition of Done

- `livekit-plugins-hume` includes a working `llm.RealtimeModel` implementation backed by Hume EVI chat websocket.
- Agent can run full duplex S2S with:
  - user audio input streaming,
  - assistant audio output playback,
  - transcription events,
  - interruption handling,
  - tool calls and tool responses.
- Connection feature coverage includes:
  - `config_id`, `config_version`,
  - `resumed_chat_group_id`,
  - `verbose_transcription`,
  - `allow_connection` for control-plane compatibility,
  - connect-time `session_settings` fields supported by handshake query params:
    `audio`, `context`, `custom_session_id`, `language_model_api_key`,
    `system_prompt`, `variables`, `voice_id`.
- Runtime session feature coverage includes:
  - websocket `SessionSettings` updates for `system_prompt`, `context`, `variables`, `voice_id`,
  - websocket `SessionSettings` updates for `tools` (user-defined function tools).
  - `builtin_tools` support in V1 via configuration (`config_id`/`config_version`);
    runtime mutation deferred.
- Realtime interface contract coverage includes:
  - `generate_reply()` future completion with `user_initiated=True` for text-triggered turns,
  - passive server-generated turns (`user_initiated=False`) for audio-driven replies,
  - `push_video()` safe no-op behavior (audio-only provider),
  - `metrics_collected` event emission.
- Unit tests pass for event mapping, audio encode/decode path, and tool lifecycle.
- Example agent added for local validation.
- Plugin README updated with setup and feature matrix.

## 3) Target Architecture

```text
LiveKit AgentSession
  -> livekit.plugins.hume.realtime.RealtimeModel
    -> AsyncHumeClient.empathic_voice.chat.connect(...)
      -> Hume EVI WebSocket
        <- AudioOutput / AssistantMessage / ToolCall / UserInterruption / ...
```

### Control/Management Plane (same plugin package)

```text
App bootstrap / backend utilities
  -> Hume REST clients (configs, prompts, tools, chat_groups, chats, control_plane)
  -> create/patch/list configs and operational hooks
```

## 4) Feature Coverage Plan

### Runtime Features (chat websocket)

| Feature | Planned | Notes |
|---|---|---|
| Duplex audio streaming | Yes | `AudioInput` publish + `AudioOutput` subscribe |
| Turn-taking / interruption | Yes | map `user_interruption` and speech boundary events |
| Interim/final transcription | Yes | enable `verbose_transcription` for interim `user_message` |
| Assistant text stream | Yes | map `assistant_message` chunks into LiveKit message generation |
| Tool calling (function tools) | Yes | map `tool_call` to LiveKit `FunctionCall` |
| Built-in tools (`web_search`, `hang_up`) | Yes | V1 via configuration (`config_id`/`config_version`), not runtime session mutation |
| Pause/resume assistant | Yes | support publish events for pause/resume control |
| Dynamic variables | Yes | via `session_settings.variables` |
| Runtime prompt override | Yes | via `session_settings.system_prompt` |
| Runtime context injection | Yes | via `session_settings.context` |
| Voice switching mid-session | Yes | via `session_settings.voice_id` |
| Chat resumability | Yes | use `resumed_chat_group_id` on connect/reconnect |
| Chat metadata tracking | Yes | capture `chat_id` and `chat_group_id` from metadata event |

### Config/Operations Features (REST/control-plane)

| Feature | Planned | Notes |
|---|---|---|
| Config CRUD/versioning | Yes | `empathic_voice.configs` |
| Prompt/tool management | Yes | `prompts`, `tools` clients |
| Webhooks in config | Yes | supported via config payload |
| Chat and chat-group retrieval | Yes | `chats`, `chat_groups` clients |
| In-progress chat external control | Yes | `control_plane` (requires `allow_connection=true`) |
| Event history retrieval | Yes | read path only (no schema changes) |

## 5) Detailed Implementation Phases

### Phase 0: Contract Freeze and Packaging

### Tasks
- Lock baseline against local SDK at `../../HumeAI/hume-python-sdk` (currently `0.13.8`).
- Define plugin runtime dependency:
  - add `hume` python SDK dependency in `livekit-plugins-hume/pyproject.toml`
    with an initial bounded range (for example `>=0.13.8,<0.14`), then widen only after CI.
- Confirm minimum Python compatibility and websocket stack compatibility with repo constraints.
- Define default auth policy:
  - support `api_key` and `access_token`,
  - recommend short-lived `access_token` for production to reduce credential blast radius
    (both methods are passed in websocket query params in current SDK).
  - never log websocket URLs or query strings; redact auth-bearing params in all connection/error
    logs and telemetry attributes.

### Exit criteria
- `uv sync` resolves successfully.
- Hume SDK imports cleanly from plugin package test shell.

### Phase 1: Realtime Module Skeleton

### Tasks
- Add module tree:
  - `livekit-plugins/livekit-plugins-hume/livekit/plugins/hume/realtime/__init__.py`
  - `livekit-plugins/livekit-plugins-hume/livekit/plugins/hume/realtime/realtime_model.py`
  - optional helpers (`events.py`, `audio.py`) if needed for separation.
- Export realtime symbols in:
  - `livekit-plugins/livekit-plugins-hume/livekit/plugins/hume/__init__.py`
- Define options dataclass for:
  - auth (`api_key`, optional `access_token` path),
  - `config_id`, `config_version`,
  - `evi_version` use via config management flow (not websocket parameter),
  - `verbose_transcription`, `allow_connection`, `resumed_chat_group_id`,
  - initial `session_settings`.
- Define explicit method contracts for LiveKit `RealtimeSession` API:
  - `generate_reply()` semantics for Hume (active trigger vs passive wait),
  - `update_options(tool_choice=...)` mapping or safe no-op if unsupported,
  - `push_video()` audio-only no-op,
  - `interrupt()` local-only or provider cancel mapping, explicitly idempotent,
  - `commit_audio()` / `clear_audio()` / `truncate()` no-op or mapped behavior.
- Set realtime capabilities to match provider behavior.

### Initial capability target
- `turn_detection=True`
- `user_transcription=True`
- `audio_output=True`
- `manual_function_calls=True`
- `auto_tool_reply_generation=False` with staged tool/user updates flushed only after
  `generate_reply()` future arming.
- `message_truncation=False` (until proven safe/supported)

### Exit criteria
- `RealtimeModel.session()` returns a valid `RealtimeSession`.
- Plugin imports without side effects.

### Phase 2: Connection Lifecycle and Event Loop

### Tasks
- Wire async websocket connection via:
  - `AsyncHumeClient(...).empathic_voice.chat.connect(...)`.
- Implement session worker tasks:
  - outgoing publish queue,
  - incoming subscribe stream parser,
  - reconnect/backoff task handling.
- Surface LiveKit error events with recoverable vs non-recoverable classification.

### Event mapping plan
- `UserInterruption` -> emit `input_speech_started`.
- `UserMessage`:
  - first observed user speech signal in a turn (interim or final) should emit
    `input_speech_started` if not already emitted for that turn.
  - assign a deterministic per-turn synthetic `item_id` on first interim/final transcript and
    reuse it for all transcript events in that user turn.
  - `interim=true` -> emit `input_audio_transcription_completed` with `is_final=false`.
  - `interim=false` -> emit `input_speech_stopped`, then
    `input_audio_transcription_completed` with `is_final=true`.
  - if `verbose_transcription` is disabled, finalize only path still emits
    `input_speech_stopped` on each final `UserMessage`.
- `AssistantMessage` + `AudioOutput` + `AssistantEnd`:
  - correlate into one `GenerationCreatedEvent` with precedence:
    `assistant_message.id` -> `audio_output.id` -> session-scoped synthetic turn key.
  - treat `AssistantEnd` as the authoritative generation boundary event.
- `WebSocketError` -> emit `error` with recoverable classification based on code/slug.
- `AssistantProsody` -> ignore for turn orchestration and message generation;
  optional debug-only telemetry path.
- parser handling for unknown subscribe variants -> explicit warning + safe skip (no crash).

### Generation contract plan
- Maintain an internal queue of pending `generate_reply()` futures.
- For text-triggered and tool-result-triggered turns, `update_chat_ctx()` must not eagerly publish
  provider events that can trigger generation. Stage outbound `UserInput`/`ToolResponseMessage`/
  `ToolErrorMessage` and flush only after `generate_reply()` arms its pending future.
- When `generate_reply()` is invoked, resolve the oldest pending future on the next
  correlated assistant generation and mark `user_initiated=True`.
- Assistant generations arriving without pending requests are emitted as passive
  server turns with `user_initiated=False`.
- If an assistant event races with staged flush, classify using arming-window fencing so
  generate-reply-triggered turns are not misclassified as passive.
- Apply a bounded timeout to each pending future (target: 5s); timeout raises `RealtimeError`.
- On `aclose()` and reconnect transitions, fail all pending futures deterministically to avoid
  deadlocks.

### Exit criteria
- Local mocked websocket tests show stable start/stop and clean `aclose()`.

### Phase 3: Audio Pipeline

### Tasks
- Implement `push_audio(frame)`:
  - normalize/convert PCM payload,
  - align with connect-time `session_settings.audio` (`linear16`, sample rate, channels),
  - base64 encode to Hume `AudioInput`.
- Implement output decode:
  - base64 decode Hume `AudioOutput` WAV payload,
  - parse WAV header once at stream start, then treat subsequent chunk bytes as raw PCM payload
    for frame emission (no per-chunk `wave.open()` parsing),
  - convert to `rtc.AudioFrame`,
  - stream through generation audio channel.
- Add adaptive buffering/chunking policy tuned for low latency
  (target default 20ms server-side chunk cadence).

### Constraints to enforce
- mono audio path with explicit sample-rate handling.
- bounded queue sizes to prevent runaway memory on backpressure.
- explicit resampling policy:
  - input resample to configured session input rate,
  - output decode from Hume WAV 48kHz before frame emission.

### Exit criteria
- Round-trip audio path verified in tests with deterministic fixtures.

### Phase 4: Tooling and Conversation State

### Tasks
- Map LiveKit tool schema to Hume session/config tool format.
- Handle incoming `tool_call`:
  - create LiveKit `FunctionCall`,
  - preserve `tool_call_id`,
  - parse `ToolCallMessage.parameters` via a fixture-locked format contract derived from a real
    websocket trace (arguments JSON vs schema-like payload),
  - route tool outputs back via `ToolResponseMessage` / `ToolErrorMessage`.
- Handle `response_required=false` tool calls as informational without deadlocking reply pipeline.
- Keep local conversation mirror for diff-safe updates where required.
- Accept and pass through server-originated `tool_response` / `tool_error` events for built-in tools
  to keep conversation state and telemetry consistent.

### Exit criteria
- Tool call success/error paths validated with unit tests.

### Phase 5: Session Settings + Dynamic Updates

### Tasks
- Implement:
  - `update_instructions()` -> `session_settings.system_prompt`,
  - `update_chat_ctx()` -> context injection strategy,
  - `update_tools()` -> user-defined tool set update (V1),
  - `update_options()` -> selected runtime options.
- Support pause/resume controls when agent policy needs response suppression.
- Defer runtime `builtin_tools` mutation in V1; provision built-ins via config versions.
- Document unsupported/no-op methods with explicit warnings:
  - `commit_audio`, `clear_audio`, `truncate` behavior if not provider-native.
  - `interrupt` uses explicit provider cancel only if proven safe; otherwise local-only best-effort.
  - no-op methods must return immediately and be idempotent.
- Ensure `update_chat_ctx()` uses deterministic diffing to avoid replaying old items on each update.

### Exit criteria
- Dynamic updates can be applied mid-session without reconnect.

### Phase 6: Resilience, Resume, and Limits

### Tasks
- Implement reconnect with jittered backoff.
- Persist latest `chat_group_id` and reconnect using `resumed_chat_group_id`.
- Emit `session_reconnected` after a successful reconnect and state resync.
- Respect provider session limits with proactive rotation option.
- Capture metrics:
  - connect latency,
  - first token latency,
  - generation duration,
  - audio output throughput,
  - reconnect counts.
- Emit LiveKit `metrics_collected` events in the same shape expected by existing telemetry hooks.

### Exit criteria
- Forced disconnect simulation resumes session context successfully.

### Phase 7: Config and Operations Helpers

### Tasks
- Add helper docs/examples for:
  - create/update config version (`evi_version="4-mini"`),
  - assign voice/prompt/tools/builtin tools/timeouts/nudges/webhooks,
  - list chats/chat groups,
  - optional control-plane usage with `allow_connection=true`.
- Keep helpers thin; avoid embedding large management framework in plugin runtime path.

### Exit criteria
- Reproducible bootstrap flow from empty account to running voice agent.

### Phase 8: Tests, Example, and Docs

### Tasks
- Add tests for:
  - capability flags and option translation,
  - websocket event mapping,
  - explicit handling for `WebSocketError` and `AssistantProsody`,
  - interleaved/out-of-order `AssistantMessage` and `AudioOutput` event sequences,
  - audio encode/decode conversions,
  - chunked WAV decode behavior (header once, PCM continuation),
  - tool lifecycle and IDs,
  - `ToolCallMessage.parameters` parsing contract from captured websocket fixture,
  - `generate_reply()` pending-future resolution semantics,
  - `update_chat_ctx()` -> `generate_reply()` ordering race (no passive misclassification),
  - `generate_reply()` timeout and close/reconnect cancellation semantics,
  - reconnect/resume semantics,
  - deterministic transcript `item_id` assignment across interim/final events,
  - unsupported realtime methods (`commit_audio`, `clear_audio`, `truncate`) no-op contract,
  - `interrupt()` idempotent/best-effort contract under active and inactive generations,
  - `push_video()` no-op contract.
- Add compatibility regression tests for existing Hume TTS import and basic synth path.
- Add example:
  - `examples/voice_agents/hume_evi_realtime.py`
- Update docs:
  - `livekit-plugins/livekit-plugins-hume/README.md`
  - include env vars, config bootstrap, and feature support table.

### Exit criteria
- Targeted tests pass.
- Example runs end-to-end with valid keys/config.

## 6) File-Level Worklist

- [ ] `livekit-plugins/livekit-plugins-hume/pyproject.toml`
- [ ] `livekit-plugins/livekit-plugins-hume/livekit/plugins/hume/__init__.py`
- [ ] `livekit-plugins/livekit-plugins-hume/livekit/plugins/hume/realtime/__init__.py`
- [ ] `livekit-plugins/livekit-plugins-hume/livekit/plugins/hume/realtime/realtime_model.py`
- [ ] `livekit-plugins/livekit-plugins-hume/livekit/plugins/hume/realtime/events.py` (if needed)
- [ ] `livekit-plugins/livekit-plugins-hume/livekit/plugins/hume/realtime/audio.py` (if needed)
- [ ] `livekit-plugins/livekit-plugins-hume/README.md`
- [ ] `examples/voice_agents/hume_evi_realtime.py`
- [ ] `tests/test_hume_realtime_config.py`
- [ ] `tests/test_hume_realtime_events.py`
- [ ] `tests/test_hume_realtime_audio.py`
- [ ] `tests/test_hume_realtime_tools.py`
- [ ] `tests/test_hume_realtime_noop_methods.py`
- [ ] `tests/test_hume_realtime_generate_reply.py`
- [ ] `tests/test_hume_realtime_event_ordering.py`
- [ ] `tests/test_hume_realtime_interrupt.py`
- [ ] `tests/test_hume_tts_regression.py`

## 7) Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| SDK/API shape drift | Medium | pin SDK version initially, then widen after CI validation |
| Unsupported realtime method mismatch (`truncate`, `commit_audio`) | Medium | explicit no-op semantics + warning logs |
| Audio format mismatch across providers | High | strict conversion helpers + fixture tests |
| Assistant text/audio chunk correlation ambiguity | High | explicit ID-based turn correlation + synthetic key fallback |
| Session timeout / long call limits | Medium | reconnect/rotation policy + metrics alerts |
| Tool-call deadlocks on missing response | High | enforce response timeout and error fallback path |
| Backpressure under poor network | Medium | bounded channels, drop/flush policy, telemetry |
| API key/token leakage via URL query parameters in logs | High | enforce URL/query redaction in logs/telemetry and prefer short-lived access tokens |

## 8) Rollout Plan

1. Land behind new plugin module without changing existing TTS behavior.
2. Validate with internal demo agent and synthetic load.
3. Publish plugin prerelease (`livekit-plugins-hume` prerelease tag).
4. Run limited production canary with observability enabled.
5. Promote to stable after latency/error thresholds are met.

## 9) Defaults Locked For Implementation

- Default voice in first example: use the voice already configured in Hume config; no hard-coded
  voice ID in sample code.
- `verbose_transcription` in sample code: default `true` for better interruption UX.
- Default quickstart `builtin_tools`: enable both `web_search` and `hang_up`; document clearly
  that `hang_up` can end the websocket/session and should be disabled for unattended smoke tests.
- Auth posture in docs/examples: `access_token`-first with `api_key` fallback for local dev;
  position tokens as short-lived risk reduction, not query-param hiding.
- `interrupt()` behavior (V1): local interruption is authoritative and idempotent; provider-side
  cancel is optional/best-effort only when clearly safe.
- `generate_reply()` reliability: each pending future gets a bounded timeout (target 5s); pending
  futures are failed on `aclose()` and reconnect transitions to prevent deadlocks.
- Runtime `builtin_tools` updates are deferred for V1; connect-time `builtin_tools` remains
  supported.
- `InputTranscriptionCompleted.item_id`: use deterministic synthetic IDs per finalized user turn
  (Hume `UserMessage` has no stable item ID).
- Assistant turn correlation priority: `assistant_message.id` -> `audio_output.id` -> session-scoped
  synthetic turn key.

## 10) References

- Hume EVI overview: https://dev.hume.ai/docs/speech-to-speech-evi/overview
- Hume EVI configuration: https://dev.hume.ai/docs/speech-to-speech-evi/configuration
- Hume EVI language model config: https://dev.hume.ai/docs/speech-to-speech-evi/configuration/language-model
- Hume EVI tool use: https://dev.hume.ai/docs/speech-to-speech-evi/features/tool-use
- Hume EVI custom language model guide (out of current scope): https://dev.hume.ai/docs/speech-to-speech-evi/guides/custom-language-model
- Local SDK root: `../../HumeAI/hume-python-sdk`
- LiveKit realtime abstraction: `livekit-agents/livekit/agents/llm/realtime.py`
- Existing Hume plugin package: `livekit-plugins/livekit-plugins-hume`

# GPT-Live

Two agents for the OpenAI GPT-Live full-duplex voice model: the ordinary one, and the variant where this process does the reasoning.

For setup instructions and more details, see the [main examples README](../../README.md).

```bash
python gpt_live_agent.py console
python client_delegation.py console
```

## Voices

Both examples use `voice="marin"`. `GPTLiveVoices` also offers `aster`, `beacon`, `cinder`, `stone`, and `vesper`. Other supported names and custom voice objects still pass through to the API.

## Context acknowledgments

`append_instructions`, `append_thinking`, and `append_commentary` queue context and return without waiting for an acknowledgment. Their `session.*.appended` events arrive at the estimated context-injection end. They do not mean speech has finished. The plugin does not gate later commands on these events or apply an acknowledgment timeout. The adapter uses output audio to determine when speech ends.

`session.closed.reason` accepts `close_requested`, `expired`, `content`, `remote_hangup`, and `connection_lost`. The plugin logs the reason and collects the final usage for each close event.

## Delegation

GPT-Live listens and speaks at the same time, but it does no reasoning and runs no tools of its own. When the conversation needs either, it delegates. Where that work goes is fixed when the session opens and cannot change afterwards.

| file | `delegation=` | who does the work |
| --- | --- | --- |
| `gpt_live_agent.py` | `"responses"` (the default) | a backend Responses model |
| `client_delegation.py` | `"client"` | your own process |

## Responses delegation — `gpt_live_agent.py`

The backend model calls the agent's `@function_tool` the usual way, so this reads like any other voice agent. `responses_options` selects that model and gives it its own instructions, separate from the voice persona.

The example also seeds a prior conversation as startup history and adds `WebSearch()`, a hosted tool the backend runs with no client round trip.

## Client delegation — `client_delegation.py`

The model hands the work over as a `delegation_created` event and waits. Nothing on the wire can reach a framework tool, so the agent carries no tools at all — passing any raises `RealtimeError` when the session starts, rather than leaving an agent whose tools silently never run.

The tools live on an ordinary `llm.LLM` that `run_delegation` drives itself, and the answer goes back through `append_commentary(answer, delegation_id=...)`, which the voice model says in its own words, capped at 500 tokens.

The event arrives before the caller's turn reaches the chat context, so the words that triggered it ride on it as `pending_transcript`; everything before that is already history. `pending_message_id` and `pending_message_created_at` identify that same user message across interim and final transcripts. Upsert by this ID rather than creating a second user turn for the partial text. Both optional fields are `None` when there is no current user speech.

`delegation_created` is emitted from the plugin's read loop, so the handler starts a task and returns instead of blocking it.

### A simple example, and what it cannot do

Each delegation answers on its own, in its own task, knowing only its own request. So a later delegation cannot replace an earlier one: ask to book Monday, change your mind to Tuesday a moment later, and both run and both answer.

Superseding needs one expert that holds the whole conversation and sees the correction. The framework will support it.

## Explicit backend image and typed input (local fork)

The Live audio frontend cannot accept images. Passing an image in its `ChatContext` now raises `RealtimeError` before storing or discarding anything. In client delegation, send the image and exact typed text to your own vision backend. In managed Responses delegation, select a vision-capable backend and use the separate typed input API after `session.started`:

```python
from openai.types.responses.response_input_item_param import ResponseInputItemParam

item: ResponseInputItemParam = {
    "type": "message",
    "role": "user",
    "content": [
        {"type": "input_text", "text": "I selected object 0043 in this current image."},
        {"type": "input_image", "image_url": current_image_url, "detail": "high"},
    ],
}
receipts = live_session.queue_backend_input([item])
run_receipt = live_session.run_backend()
```

The complete input batch is validated before sending any item. HTTP(S) image URLs, base64 image data URIs, and file IDs are supported. No image download or provider request occurs during validation. Queueing creates `response.item.create` events; it does not issue `response.create`. A later automatic voice delegation can also consume queued backend context. Queueing and explicit running both reject known active backend work, including the interval between delegation creation and its first response event. Wait for completion and pending tool results before trying again. A correction does not cancel or supersede an existing task; application code owns that decision.

Each returned `GPTLiveCommandReceipt` has an event ID, connection epoch, event type, status, and optional error code. The same receipt object updates as transport events arrive. `queued` means local enqueue; `sent` means the WebSocket send completed. **Backend item/create commands have no standalone correlated success acknowledgment.** Their receipts remain `sent` while subsequent nested Responses lifecycle events describe backend progress; these events cannot prove that a particular image was accepted. The frontend `append_*` methods also return receipts, which become `acknowledged` only on their matching `session.*.appended` event. None of these statuses proves speech playback.

A correlated provider error marks the receipt `error`; backend errors prevent further manual runs until the context is rebuilt in a new session. A known error also suppresses a run still waiting in the local send queue. An error arriving after a run was sent cannot undo that request. Keep application submission records and do not automatically retry uncertain paid work.

Receipts that remain queued or sent become `connection_lost` when that connection ends. `connection_epoch` increments on reconnect. Unsent commands from the previous connection, including tool results and commentary, are dropped; backend inputs are never automatically replayed. Rebuild only current authorized context after the new `session.started`. An acknowledged append remains historical acceptance evidence, not proof its context survived reconnect. All receipts are retained for late errors until connection teardown; configure a finite session duration for long-running applications.

Hermetic regression checks for this fork:

```bash
.venv/bin/python -m pytest tests/test_gpt_live_model.py tests/test_gpt_live_backend_input.py tests/test_expr_markup.py --unit -q
```

These validate wire serialization, lifecycle and preserved expression behavior. They do not establish provider entitlement, visual grounding, audio quality, actual duplex interaction, or played speech.

## Input clock and reliable turn history (local fork)

An idle connection now receives 100 ms silence frames after startup, keeping Live's input clock running for typed requests and missing microphones. Padding pauses while supplied microphone audio still covers that time and ends with the WebSocket attempt. This advances the required audio clock for typed requests; actual spoken output still requires provider and runtime acceptance. It does not turn this audio model into a text-only simulation model.

User transcript finalization uses the bundled local speech detector, rather than counting every microphone frame as silence. Ongoing speech can therefore continue while transcript delivery pauses. Only consecutive non-speech input after the latest transcript advances the silence interval. Detector results for audio queued before that transcript are excluded. The native detector runs locally, and reconnect/close settles its tasks. Mute transitions flush partial input first, make local detection observe silence, and preserve desired mute state before new-connection audio.

Backend function items require `status="completed"` and dispatch at most once per call ID on a connection. New response events preserve outstanding calls, duplicate response creation cannot reset a completed result barrier, and failed/incomplete blockers release other ready work. One conversation-level continuation is sent after the known required outputs are ready; a refused or uncertain continuation is not automatically retried. Voice-usage accounting keeps an increasing finite watermark and rejects malformed duration values before event parsing, so an invalid final usage field cannot prevent close acknowledgment.

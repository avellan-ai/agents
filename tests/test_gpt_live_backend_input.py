from __future__ import annotations

import asyncio

import pytest

from livekit.agents import llm
from livekit.plugins.openai.realtime import GPTLiveModel

from .test_gpt_live_model import _completed, _connect_hook, _function_call_done, _response_event

pytestmark = pytest.mark.unit


async def test_managed_tool_keeps_original_input_identity_across_continuation(monkeypatch):
    _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    calls = []
    session.on("function_call", calls.append)
    try:
        await session._update_session()
        await asyncio.wait_for(session.wait_started(), timeout=1)
        assert session.backend_input_ready
        session.run_backend(input_message_id="turn-A")
        assert not session.backend_input_ready
        session._handle_event(
            {"type": "session.delegation.created", "delegation": {"id": "d", "target": "responses"}}
        )
        session._handle_event(
            _response_event("d", {"type": "response.created", "response": {"id": "r1"}})
        )
        session._last_final_input_id = "turn-B"
        session._handle_event(_response_event("d", _function_call_done("call-A")))
        assert calls[0].extra == {"gpt_live": {"input_message_id": "turn-A"}}
        session._handle_event(_response_event("d", _completed("r1")))
        await session._append_items(
            [llm.FunctionCallOutput(call_id="call-A", output="queued", is_error=False)]
        )
        assert not session._delegated_responses
        session._handle_event(
            _response_event("d", {"type": "response.created", "response": {"id": "r2"}})
        )
        session._handle_event(_response_event("d", _function_call_done("call-A2")))
        assert calls[1].extra == {"gpt_live": {"input_message_id": "turn-A"}}
    finally:
        await session.aclose()
        await model.aclose()


@pytest.mark.parametrize(
    "url", ["https://example.invalid/current.png", "data:image/png;base64,aGVsbG8="]
)
async def test_image_and_exact_text_survive_backend_wire_without_running(monkeypatch, url):
    ws = _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    try:
        await session._update_session()
        await asyncio.sleep(0.01)
        item = {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Object 0042; no, correct that to 0043."},
                {"type": "input_image", "image_url": url, "detail": "high"},
            ],
        }
        receipts = session.queue_backend_input([item])
        assert receipts[0].status == "queued"
        item["content"][0]["text"] = "mutated after submission"
        await asyncio.sleep(0.01)
        assert receipts[0].status == "sent"
        assert [e["type"] for e in ws.sent] == ["session.start", "response.item.create"]
        assert ws.sent[-1]["item"]["content"] == [
            {"type": "input_text", "text": "Object 0042; no, correct that to 0043."},
            {"type": "input_image", "image_url": url, "detail": "high"},
        ]
        assert "input" not in ws.sent[0]["session"]
        run = session.run_backend()
        with pytest.raises(llm.RealtimeError, match="busy"):
            session.run_backend()
        await asyncio.sleep(0.01)
        assert ws.sent[-1] == {"type": "response.create", "event_id": run.event_id}
        session._handle_event(_response_event("d", {"type": "response.created"}))
        assert run.status == "sent", "uncorrelated lifecycle must not fabricate acknowledgment"
        session._handle_event(_response_event("d", _completed("r")))
        assert receipts[0].status == "sent", "backend items have no success acknowledgment"
        session.queue_backend_input([{"role": "user", "content": "next"}])
    finally:
        await session.aclose()
        await model.aclose()


async def test_backend_batch_validation_is_atomic_and_mode_guarded(monkeypatch):
    ws = _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    try:
        with pytest.raises(llm.RealtimeError, match="session.started"):
            session.queue_backend_input([{"role": "user", "content": "text"}])
        await session._update_session()
        await asyncio.sleep(0.01)
        with pytest.raises(ValueError, match="input_text and input_image"):
            session.queue_backend_input(
                [
                    {"role": "user", "content": "valid"},
                    {"role": "user", "content": [{"type": "input_video", "url": "bad"}]},
                ]
            )
        with pytest.raises(ValueError, match="messages only"):
            session.queue_backend_input(
                [{"type": "function_call_output", "call_id": "fake", "output": "bad"}]
            )
        await asyncio.sleep(0.01)
        assert len(ws.sent) == 1
        session._opts.delegation = "client"
        with pytest.raises(llm.RealtimeError, match="responses delegation"):
            session.run_backend()
        with pytest.raises(llm.RealtimeError, match="responses delegation"):
            session.queue_backend_input([{"role": "user", "content": "valid"}])
    finally:
        await session.aclose()
        await model.aclose()
    with pytest.raises(llm.RealtimeError, match="closed"):
        session.append_thinking("late result")


async def test_images_in_chat_context_fail_before_mutation(monkeypatch):
    _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test", delegation="client")
    session = model.session()
    try:
        message = llm.ChatMessage(
            role="user",
            content=["look", llm.ImageContent(image="https://example.invalid/still.png")],
        )
        with pytest.raises(llm.RealtimeError, match="client vision backend"):
            await session._append_items([message])
        assert not session._history.items
    finally:
        await session.aclose()
        await model.aclose()


async def test_pending_tools_guard_input_and_continue_once_in_order(monkeypatch):
    ws = _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    try:
        await session._update_session()
        await asyncio.sleep(0.01)
        session._handle_event(_response_event("d", {"type": "response.created"}))
        session._handle_event(_response_event("d", _function_call_done("a")))
        session._handle_event(_response_event("d", _function_call_done("b")))
        session._handle_event(_response_event("d", _completed("r")))
        with pytest.raises(llm.RealtimeError, match="pending tool results"):
            session.queue_backend_input([{"role": "user", "content": "correction"}])
        await session._append_items(
            [llm.FunctionCallOutput(call_id="a", output="one", is_error=False)]
        )
        with pytest.raises(llm.RealtimeError, match="busy"):
            session.run_backend()
        await session._append_items(
            [llm.FunctionCallOutput(call_id="b", output="two", is_error=False)]
        )
        with pytest.raises(llm.RealtimeError, match="busy"):
            session.run_backend()  # continuation queued but response.created not received
        await asyncio.sleep(0.01)
        assert [e["type"] for e in ws.sent[1:]] == [
            "response.item.create",
            "response.item.create",
            "response.create",
        ]
        session._handle_event(_response_event("d", {"type": "response.created"}))
        session._handle_event(_response_event("d", _completed("r2")))
        session.queue_backend_input([{"role": "user", "content": "correction"}])
    finally:
        await session.aclose()
        await model.aclose()


async def test_correlated_append_ack_and_backend_error_are_distinct(monkeypatch):
    _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    try:
        await session._update_session()
        await asyncio.sleep(0.01)
        append = session.append_thinking("verified visible facts")
        backend = session.queue_backend_input([{"role": "user", "content": "0042"}])[0]
        await asyncio.sleep(0.01)
        ack = {"type": "session.thinking.appended", "client_event_id": append.event_id}
        session._handle_event(ack)
        session._handle_event(ack)
        assert append.status == "acknowledged"
        assert backend.status == "sent"
        session._handle_event(
            {
                "type": "error",
                "error": {
                    "client_event_id": backend.event_id,
                    "code": "invalid_image",
                    "message": "private detail",
                },
            }
        )
        assert backend.status == "error"
        assert backend.error_code == "invalid_image"
        assert "private detail" not in repr(backend)
        with pytest.raises(llm.RealtimeError, match="new session"):
            session.run_backend()
    finally:
        await session.aclose()
        await model.aclose()


async def test_reconnect_invalidates_receipts_and_drops_unsent_work(monkeypatch):
    ws = _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    try:
        await session._update_session()
        await asyncio.sleep(0.01)
        item = session.queue_backend_input([{"role": "user", "content": "do once"}])[0]
        run = session.run_backend()
        append = session.append_commentary("old connection result")
        await session._reset_for_reconnect()
        assert session.connection_epoch == 1
        assert all(r.status == "connection_lost" for r in (item, run, append))
        session._handle_event(
            {"type": "session.commentary.appended", "client_event_id": append.event_id}
        )
        assert append.status == "connection_lost"
        assert session._msg_ch.empty()
        await asyncio.sleep(0.01)
        assert len(ws.sent) == 1
    finally:
        session._handle_event({"type": "session.started"})
        await session.aclose()
        await model.aclose()


@pytest.mark.parametrize(
    "image",
    [
        {"type": "input_image"},
        {"type": "input_image", "image_url": "file:///private.png"},
        {
            "type": "input_image",
            "image_url": "https://example.invalid/still.png",
            "file_id": "file_1",
        },
        {"type": "input_image", "image_url": "data:image/png;base64,"},
        {"type": "input_image", "image_url": "data:image/png;base64,not base64!"},
    ],
)
async def test_invalid_image_batch_never_partially_sends(monkeypatch, image):
    ws = _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    try:
        await session._update_session()
        await asyncio.sleep(0.01)
        with pytest.raises(ValueError):
            session.queue_backend_input(
                [
                    {"role": "user", "content": "valid"},
                    {"role": "user", "content": [image]},
                ]
            )
        await asyncio.sleep(0.01)
        assert len(ws.sent) == 1
    finally:
        await session.aclose()
        await model.aclose()


async def test_startup_image_context_rejected_before_start(monkeypatch):
    ws = _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    try:
        ctx = llm.ChatContext.empty()
        ctx.add_message(
            role="user", content=[llm.ImageContent(image="https://example.invalid/still.png")]
        )
        with pytest.raises(llm.RealtimeError, match="does not accept images"):
            await session._update_session(chat_ctx=ctx)
        await asyncio.sleep(0.01)
        assert not session._history.items
        assert not ws.sent
    finally:
        await session.aclose()
        await model.aclose()


async def test_automatic_delegation_blocks_manual_input_before_response_created(monkeypatch):
    _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    try:
        await session._update_session()
        await asyncio.sleep(0.01)
        session._handle_event(
            {"type": "session.delegation.created", "delegation": {"id": "d", "target": "responses"}}
        )
        with pytest.raises(llm.RealtimeError, match="busy"):
            session.run_backend()
        with pytest.raises(llm.RealtimeError, match="busy"):
            session.queue_backend_input([{"role": "user", "content": "correction"}])
        session._handle_event(_response_event("d", {"type": "response.created"}))
        session._handle_event(_response_event("d", _completed("r")))
        session.queue_backend_input([{"role": "user", "content": "correction"}])
    finally:
        await session.aclose()
        await model.aclose()


async def test_append_receipt_cannot_ack_a_backend_item(monkeypatch):
    _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    try:
        await session._update_session()
        await asyncio.sleep(0.01)
        receipt = session.queue_backend_input(
            [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "file_id": "file_test", "detail": "auto"}],
                }
            ]
        )[0]
        await asyncio.sleep(0.01)
        session._handle_event(
            {"type": "session.thinking.appended", "client_event_id": receipt.event_id}
        )
        assert receipt.status == "sent"
    finally:
        await session.aclose()
        await model.aclose()
    assert receipt.status == "connection_lost"


async def test_send_failure_never_replays_uncertain_backend_work(monkeypatch):
    import aiohttp

    from livekit.agents import APIConnectOptions
    from livekit.plugins.openai.realtime import GPTLiveSession

    from .test_gpt_live_model import _LifecycleWS

    class FailingWS(_LifecycleWS):
        async def send_str(self, data):
            import json

            if json.loads(data)["type"] == "response.item.create":
                raise aiohttp.ClientConnectionError("uncertain send")
            await super().send_str(data)

    sockets = [FailingWS(), _LifecycleWS()]
    connections = iter(sockets)

    async def connect(self):
        return next(connections)

    monkeypatch.setattr(GPTLiveSession, "_create_ws_conn", connect)
    model = GPTLiveModel(
        api_key="test", conn_options=APIConnectOptions(max_retry=1, retry_interval=0)
    )
    session = model.session()
    try:
        await session._update_session()
        await asyncio.wait_for(sockets[0].started.wait(), 1)
        await session._session_started_fut
        receipt = session.queue_backend_input([{"role": "user", "content": "once"}])[0]
        run = session.run_backend()
        await asyncio.wait_for(sockets[1].started.wait(), 1)
        assert session.connection_epoch == 1
        assert receipt.status == run.status == "connection_lost"
        assert [e["type"] for e in sockets[1].sent] == ["session.start"]
    finally:
        for ws in sockets:
            ws.emit({"type": "session.closed"})
        await session.aclose()
        await model.aclose()


async def test_a_known_input_error_prevents_already_queued_run(monkeypatch):
    ws = _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    try:
        await session._update_session()
        await asyncio.sleep(0.01)
        item = session.queue_backend_input([{"role": "user", "content": "invalid context"}])[0]
        run = session.run_backend()
        session._handle_event(
            {"type": "error", "error": {"client_event_id": item.event_id, "code": "invalid_input"}}
        )
        await asyncio.sleep(0.01)
        assert run.status == "error"
        assert run.error_code == "backend_context_failed"
        assert all(e["type"] != "response.create" for e in ws.sent)
    finally:
        await session.aclose()
        await model.aclose()


async def test_continuation_waits_for_every_known_delegation_batch(monkeypatch):
    ws = _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test")
    session = model.session()
    try:
        await session._update_session()
        await asyncio.sleep(0.01)
        for d in ("a", "b"):
            session._handle_event(_response_event(d, {"type": "response.created"}))
            session._handle_event(_response_event(d, _function_call_done(d)))
            session._handle_event(_response_event(d, _completed(d)))
        await session._append_items(
            [llm.FunctionCallOutput(call_id="a", output="one", is_error=False)]
        )
        await asyncio.sleep(0.01)
        assert all(e["type"] != "response.create" for e in ws.sent)
        await session._append_items(
            [llm.FunctionCallOutput(call_id="b", output="two", is_error=False)]
        )
        await asyncio.sleep(0.01)
        assert [e["type"] for e in ws.sent[1:]] == [
            "response.item.create",
            "response.item.create",
            "response.create",
        ]
    finally:
        await session.aclose()
        await model.aclose()


async def test_client_delegation_preserves_pending_user_message_identity(monkeypatch):
    from .test_gpt_live_model import _transcript

    _connect_hook(monkeypatch)
    model = GPTLiveModel(api_key="test", delegation="client")
    session = model.session()
    delegations = []
    transcripts = []
    session.on("delegation_created", delegations.append)
    session.on("input_audio_transcription_completed", transcripts.append)
    try:
        await session._update_session()
        await asyncio.sleep(0.01)
        session._handle_event(_transcript("user", "Look at", 100))
        session._handle_event(
            {"type": "session.delegation.created", "delegation": {"id": "d", "target": "client"}}
        )
        pending = delegations[0]
        assert pending.pending_transcript == "Look at"
        assert pending.pending_message_id == transcripts[0].item_id
        assert pending.pending_message_created_at == transcripts[0].turn_started_at
        session._handle_event(_transcript("user", " the door", 300))
        session._end_speech("user")
        assert transcripts[-1].is_final
        assert transcripts[-1].transcript == "Look at the door"
        assert transcripts[-1].item_id == pending.pending_message_id
        assert transcripts[-1].turn_started_at == pending.pending_message_created_at
        session._handle_event(
            {"type": "session.delegation.created", "delegation": {"id": "d2", "target": "client"}}
        )
        assert delegations[-1].pending_transcript == ""
        assert delegations[-1].pending_message_id is None
        assert delegations[-1].pending_message_created_at is None
    finally:
        await session.aclose()
        await model.aclose()

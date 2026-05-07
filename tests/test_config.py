from dataclasses import replace
from typing import Any, cast

import pytest
from openai.types.shared import Reasoning

from livekit.plugins.openai.realtime import RealtimeModel
from livekit.plugins.openai.realtime.realtime_model import RealtimeSession, process_base_url


def _session_event_payload(event: Any) -> dict[str, Any]:
    assert not isinstance(event, dict)
    return cast(
        dict[str, Any],
        event.model_dump(by_alias=True, exclude_unset=True, exclude_defaults=False),
    )


def _create_realtime_session(model: RealtimeModel) -> RealtimeSession:
    session = RealtimeSession.__new__(RealtimeSession)
    session._opts = replace(model._opts)
    session._instructions = None
    return session


def _create_session_update_payload(model: RealtimeModel) -> dict[str, Any]:
    session = _create_realtime_session(model)
    event = RealtimeSession._create_session_update_event(session)
    return _session_event_payload(event)


def test_process_base_url() -> None:
    assert (
        process_base_url("https://api.openai.com/v1", "gpt-4")
        == "wss://api.openai.com/v1/realtime?model=gpt-4"
    )
    assert (
        process_base_url("http://example.com", "gpt-4") == "ws://example.com/realtime?model=gpt-4"
    )
    preserved_url = process_base_url(
        "wss://livekit.ai/voice/v1/chat/voice?client=oai&enable_noise_suppression=true",
        "gpt-4",
    )
    assert (
        preserved_url
        == "wss://livekit.ai/voice/v1/chat/voice?client=oai&enable_noise_suppression=true&model=gpt-4"
    )
    assert (
        process_base_url(
            "https://test.azure.com/openai",
            "gpt-4",
        )
        == "wss://test.azure.com/openai/realtime?model=gpt-4"
    )

    assert (
        process_base_url(
            "https://test.azure.com/openai",
            "gpt-4",
            is_azure=True,
            azure_deployment="my-deployment",
            api_version="2025-04-12",
        )
        == "wss://test.azure.com/openai/realtime?api-version=2025-04-12&deployment=my-deployment"
    )
    assert (
        process_base_url(
            "https://test.azure.com/custom/path",
            "gpt-4",
            api_version="2025-04-12",
        )
        == "wss://test.azure.com/custom/path?model=gpt-4"
    )


def test_realtime_model_supports_gpt_realtime_2() -> None:
    model = RealtimeModel(model="gpt-realtime-2", api_key="test-key")

    payload = _create_session_update_payload(model)

    assert payload["session"]["model"] == "gpt-realtime-2"
    assert "reasoning" not in payload["session"]


def test_realtime_model_includes_reasoning() -> None:
    model = RealtimeModel(
        model="gpt-realtime-2",
        api_key="test-key",
        reasoning=Reasoning(effort="high"),
    )

    payload = _create_session_update_payload(model)

    assert payload["session"]["reasoning"] == {"effort": "high"}


def test_realtime_model_update_options_includes_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RealtimeModel(model="gpt-realtime-2", api_key="test-key")
    session = _create_realtime_session(model)
    sent_events: list[Any] = []
    monkeypatch.setattr(session, "send_event", sent_events.append)

    session.update_options(reasoning=Reasoning(effort="low"))

    assert len(sent_events) == 1
    payload = _session_event_payload(sent_events[0])
    assert payload["session"]["reasoning"] == {"effort": "low"}

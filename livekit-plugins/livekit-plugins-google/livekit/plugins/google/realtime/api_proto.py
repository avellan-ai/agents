from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from google.genai import types

# Gemini API deprecations: https://ai.google.dev/gemini-api/docs/deprecations
# Gemini API release notes with preview deprecations: https://ai.google.dev/gemini-api/docs/changelog
# live models: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api
# VertexAI retirement: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions#retired-models
# Additional references:
# 1. https://github.com/kazunori279/adk-streaming-test/blob/main/test_report.md
LiveAPIModels = Literal[
    # VertexAI models
    "gemini-live-2.5-flash-native-audio",  # GA https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash-live-api#live-2.5-flash
    # Gemini API models
    "gemini-3.1-flash-live-preview",
    "gemini-2.5-flash-native-audio-preview-12-2025",  # https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash-live
]

RESTRICTED_CLIENT_CONTENT_MODELS: frozenset[str] = frozenset(
    {
        # Gemini 3.1 only accepts send_client_content for initial history seeding.
        # Mid-session text/context updates must use send_realtime_input(text=...).
        # https://ai.google.dev/gemini-api/docs/live-api/capabilities
        "gemini-3.1-flash-live-preview",
    }
)

Voice = Literal[
    "Achernar",
    "Achird",
    "Algenib",
    "Algieba",
    "Alnilam",
    "Aoede",
    "Autonoe",
    "Callirrhoe",
    "Charon",
    "Despina",
    "Enceladus",
    "Erinome",
    "Fenrir",
    "Gacrux",
    "Iapetus",
    "Kore",
    "Laomedeia",
    "Leda",
    "Orus",
    "Pulcherrima",
    "Puck",
    "Rasalgethi",
    "Sadachbia",
    "Sadaltager",
    "Schedar",
    "Sulafat",
    "Umbriel",
    "Vindemiatrix",
    "Zephyr",
    "Zubenelgenubi",
]


ClientEvents = (
    types.ContentListUnion
    | types.ContentListUnionDict
    | types.LiveClientContentOrDict
    | types.LiveClientRealtimeInput
    | types.LiveClientRealtimeInputOrDict
    | types.LiveClientToolResponseOrDict
    | types.FunctionResponseOrDict
    | Sequence[types.FunctionResponseOrDict]
)

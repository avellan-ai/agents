# Hume AI plugin for LiveKit Agents

Support for text-to-speech with [Hume](https://www.hume.ai/).

Also includes an early RealtimeModel integration for Hume EVI speech-to-speech.

See [https://docs.livekit.io/agents/integrations/tts/hume/](https://docs.livekit.io/agents/integrations/tts/hume/) for more information.

## Installation

```bash
pip install livekit-plugins-hume
```

You will need an API Key from Hume, it can be set as an environment variable: `HUME_API_KEY`. You can get it from [here](https://platform.hume.ai/settings/keys)

## Realtime (EVI)

```python
from livekit.plugins.hume import RealtimeModel

rt_model = RealtimeModel(
    model="evi-4-mini",
    api_key="...",
)
```

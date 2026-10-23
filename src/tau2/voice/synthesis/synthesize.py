"""Synthesis through the explicitly bound local runtime."""

from tau2.data_model.audio import PCM_SAMPLE_RATE, AudioData
from tau2.data_model.voice import ProviderConfig, RuntimeTTSConfig
from tau2.inference import get_inference_backends
from tau2.voice.utils.audio_preprocessing import convert_to_mono, resample_audio


def synthesize_voice(
    text: str, provider: str, provider_config: ProviderConfig
) -> AudioData:
    if provider != "runtime":
        raise ValueError("A local runtime cannot call a legacy synthesis provider")
    backends = get_inference_backends()
    if backends is None:
        raise RuntimeError("Runtime synthesis requires injected inference backends")
    if not isinstance(provider_config, RuntimeTTSConfig):
        raise TypeError("Runtime synthesis requires RuntimeTTSConfig")
    audio = backends.synthesis.synthesize(text, provider_config)
    if not audio.format.is_pcm16:
        raise ValueError(f"TTS must output PCM_S16LE, got {audio.format.encoding}")
    return resample_audio(convert_to_mono(audio), PCM_SAMPLE_RATE)

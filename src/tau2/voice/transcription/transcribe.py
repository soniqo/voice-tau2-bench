"""Transcription through the explicitly bound local runtime."""

from tau2.data_model.audio import AudioData
from tau2.data_model.voice import (
    RuntimeTranscriptionConfig,
    TranscriptionConfig,
    TranscriptionResult,
)
from tau2.inference import get_inference_backends


def transcribe_audio(
    audio_data: AudioData, config: TranscriptionConfig | RuntimeTranscriptionConfig
) -> TranscriptionResult:
    if not isinstance(config, RuntimeTranscriptionConfig):
        raise ValueError("A local runtime cannot call a legacy transcription provider")
    backends = get_inference_backends()
    if backends is None:
        raise RuntimeError("Runtime transcription requires injected inference backends")
    return backends.transcription.transcribe(audio_data, config)

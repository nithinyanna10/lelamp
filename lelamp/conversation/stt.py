from __future__ import annotations

from pydantic import BaseModel


class SttResult(BaseModel):
    text: str
    confidence: float
    language: str


async def transcribe(audio: bytes, sample_rate: int = 16000) -> SttResult:
    """whisper.cpp with CoreML preferred; faster-whisper fallback."""
    raise NotImplementedError

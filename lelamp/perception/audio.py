from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict


class SpeechEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    timestamp: float
    is_speech: bool
    confidence: float
    audio_chunk: bytes | None = None


async def audio_task(
    out_queue: asyncio.Queue[SpeechEvent],
    sample_rate: int = 16000,
) -> None:
    raise NotImplementedError


def run_vad(chunk: bytes, sample_rate: int) -> float:
    raise NotImplementedError

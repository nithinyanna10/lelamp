from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic import BaseModel, ConfigDict


class TtsChunk(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio: bytes
    is_final: bool


async def synthesize(text_stream: AsyncIterator[str]) -> AsyncIterator[TtsChunk]:
    """Cartesia streaming, fallback to macOS `say`."""
    raise NotImplementedError
    yield  # pragma: no cover - keeps this an async generator for type checkers

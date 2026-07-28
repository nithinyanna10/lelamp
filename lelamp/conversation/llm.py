from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

MODEL = "claude-sonnet-5"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "query_memory",
        "description": "Search remembered objects by description, time range, and/or spatial hint.",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "time_range": {
                    "type": "object",
                    "properties": {
                        "start_ts": {"type": "number"},
                        "end_ts": {"type": "number"},
                    },
                },
                "spatial_hint": {"type": "string"},
            },
            "required": ["description"],
        },
    },
    {
        "name": "describe_current_scene",
        "description": "Describe what the lamp currently sees.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "point_at",
        "description": "Point the lamp at a remembered object by id.",
        "input_schema": {
            "type": "object",
            "properties": {"object_id": {"type": "integer"}},
            "required": ["object_id"],
        },
    },
    {
        "name": "remember",
        "description": "Store a fact for later recall.",
        "input_schema": {
            "type": "object",
            "properties": {"fact": {"type": "string"}},
            "required": ["fact"],
        },
    },
]


class ConversationTurn(BaseModel):
    role: str
    text: str


class TextDelta(BaseModel):
    text: str


class ToolCallEvent(BaseModel):
    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str


async def stream_response(
    history: list[ConversationTurn],
) -> AsyncIterator[TextDelta | ToolCallEvent]:
    raise NotImplementedError
    yield  # pragma: no cover - keeps this an async generator for type checkers

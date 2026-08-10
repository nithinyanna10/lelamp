from __future__ import annotations

from pydantic import BaseModel


class FSMTimings(BaseModel):
    """Tuneable time-based transition thresholds, seconds. Placeholders -- not tuned
    against a real user, just plausible enough to demo the escalation ladder."""

    disengaging_to_seeking1_s: float = 30.0
    seeking1_to_seeking2_s: float = 20.0
    seeking2_to_seeking3_s: float = 20.0
    seeking3_to_sleeping_s: float = 30.0
    idle_to_sleeping_s: float = 60.0
    tick_interval_s: float = 0.5

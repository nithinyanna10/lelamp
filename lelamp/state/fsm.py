from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from lelamp.perception.face_gaze import GazeEvent


class LampState(StrEnum):
    IDLE = "idle"
    ENGAGED = "engaged"
    LISTENING = "listening"
    SPEAKING = "speaking"
    SCANNING = "scanning"
    SEEKING = "seeking"
    SLEEPING = "sleeping"


class HysteresisConfig(BaseModel):
    engage_threshold: float = 0.7
    engage_duration_s: float = 0.4
    disengage_threshold: float = 0.3
    disengage_duration_s: float = 1.5
    seek_after_disengaged_s: float = 30.0
    sleep_after_disengaged_s: float = 60.0


class StateTransition(BaseModel):
    timestamp: float
    from_state: LampState
    to_state: LampState
    reason: str


class LampFSM:
    def __init__(self, config: HysteresisConfig | None = None) -> None:
        self.config = config or HysteresisConfig()
        self.state = LampState.IDLE

    def on_gaze_event(self, event: GazeEvent) -> StateTransition | None:
        raise NotImplementedError

    def on_timer_tick(self, timestamp: float) -> StateTransition | None:
        raise NotImplementedError

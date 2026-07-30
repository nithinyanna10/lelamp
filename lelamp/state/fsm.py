from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from lelamp.perception.hysteresis import EngagementTransition
from lelamp.telemetry import get_tracer

_tracer = get_tracer(__name__)


class LampState(StrEnum):
    IDLE = "idle"
    ENGAGED = "engaged"
    LISTENING = "listening"
    SPEAKING = "speaking"
    SCANNING = "scanning"
    SEEKING = "seeking"
    SLEEPING = "sleeping"


class StateTransition(BaseModel):
    timestamp: float
    from_state: LampState
    to_state: LampState
    reason: str


class LampFSM:
    """IDLE <-> ENGAGED for step 2; SCANNING/SEEKING/SLEEPING etc. are defined but
    unreachable until later steps wire in their triggers."""

    def __init__(self) -> None:
        self.state = LampState.IDLE

    def on_engagement_transition(self, event: EngagementTransition) -> StateTransition | None:
        if event.engaged and self.state == LampState.IDLE:
            new_state, reason = LampState.ENGAGED, "gaze_engaged"
        elif not event.engaged and self.state == LampState.ENGAGED:
            new_state, reason = LampState.IDLE, "gaze_disengaged"
        else:
            return None

        with _tracer.start_as_current_span("fsm.transition") as span:
            span.set_attribute("from_state", str(self.state))
            span.set_attribute("to_state", str(new_state))
            span.set_attribute("trigger", reason)
            transition = StateTransition(
                timestamp=event.timestamp, from_state=self.state, to_state=new_state, reason=reason
            )
            self.state = new_state
            return transition

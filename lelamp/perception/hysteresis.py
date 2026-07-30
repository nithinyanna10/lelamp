from __future__ import annotations

import time

from pydantic import BaseModel


class HysteresisConfig(BaseModel):
    engage_threshold: float = 0.7
    engage_hold_ms: float = 400.0
    disengage_threshold: float = 0.3
    disengage_hold_ms: float = 1500.0


class EngagementTransition(BaseModel):
    timestamp: float
    engaged: bool


class HysteresisDebugState(BaseModel):
    engaged: bool
    candidate: bool | None
    hold_elapsed_ms: float
    hold_target_ms: float


class HysteresisGate:
    """Edge-triggered engage/disengage state machine.

    DISENGAGED -> (score > engage_threshold, sustained continuously for
    engage_hold_ms) -> ENGAGED -> (score < disengage_threshold, sustained
    continuously for disengage_hold_ms) -> DISENGAGED.

    Scores in the dead zone between disengage_threshold and engage_threshold,
    or any dip back across the *current* threshold mid-hold, reset the hold
    timer -- the dwell must be continuous, not cumulative. This is what makes
    it hysteresis rather than a debounce: the two thresholds are different, so
    noise near one boundary can't flap the state by itself.

    update() takes an explicit timestamp for unit-testability; callers that
    don't have one (e.g. driving this live off wall-clock samples) can omit it
    and get time.monotonic() ns-precision.
    """

    def __init__(self, config: HysteresisConfig | None = None) -> None:
        self.config = config or HysteresisConfig()
        self._engaged = False
        self._candidate: bool | None = None
        self._candidate_since: float | None = None

    @property
    def engaged(self) -> bool:
        return self._engaged

    def update(self, score: float, timestamp: float | None = None) -> EngagementTransition | None:
        if timestamp is None:
            timestamp = time.monotonic_ns() / 1e9

        if not self._engaged:
            target_state = True
            crossing = score > self.config.engage_threshold
            hold_ms = self.config.engage_hold_ms
        else:
            target_state = False
            crossing = score < self.config.disengage_threshold
            hold_ms = self.config.disengage_hold_ms

        if not crossing:
            self._candidate = None
            self._candidate_since = None
            return None

        if self._candidate is not target_state or self._candidate_since is None:
            self._candidate = target_state
            self._candidate_since = timestamp
            return None

        if (timestamp - self._candidate_since) * 1000.0 >= hold_ms:
            self._engaged = target_state
            self._candidate = None
            self._candidate_since = None
            return EngagementTransition(timestamp=timestamp, engaged=target_state)

        return None

    def debug_state(self, timestamp: float | None = None) -> HysteresisDebugState:
        if timestamp is None:
            timestamp = time.monotonic_ns() / 1e9
        hold_target_ms = (
            self.config.disengage_hold_ms if self._engaged else self.config.engage_hold_ms
        )
        elapsed_ms = 0.0
        if self._candidate_since is not None:
            elapsed_ms = max(0.0, (timestamp - self._candidate_since) * 1000.0)
        return HysteresisDebugState(
            engaged=self._engaged,
            candidate=self._candidate,
            hold_elapsed_ms=min(elapsed_ms, hold_target_ms),
            hold_target_ms=hold_target_ms,
        )

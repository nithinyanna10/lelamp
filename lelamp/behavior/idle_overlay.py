from __future__ import annotations

import math


class IdleOverlay:
    """Small sinusoidal joint offsets, additively composed on top of whatever
    the current expression's keyframe targets are (see expression_player.py) so
    the lamp never looks dead when nothing else is happening. Always on except
    during SLEEPING -- callers gate that with set_enabled(), not by not calling
    sample()."""

    def __init__(self, amplitude_deg: float = 2.0, period_s: float = 4.0) -> None:
        self.amplitude_rad = math.radians(amplitude_deg)
        self.period_s = period_s
        self._enabled = True

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def sample(self, t_seconds: float) -> dict[str, float]:
        if not self._enabled:
            return {}
        phase = 2 * math.pi * t_seconds / self.period_s
        return {
            "shoulder_lift": self.amplitude_rad * math.sin(phase),
            "wrist_flex": 0.5 * self.amplitude_rad * math.sin(phase + math.pi / 4),
        }

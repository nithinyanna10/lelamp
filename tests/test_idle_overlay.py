from __future__ import annotations

import math

import pytest

from lelamp.behavior.idle_overlay import IdleOverlay


def test_default_enabled() -> None:
    overlay = IdleOverlay()
    assert overlay.enabled is True


def test_disabled_returns_empty() -> None:
    overlay = IdleOverlay()
    overlay.set_enabled(False)
    assert overlay.sample(1.23) == {}


def test_enabled_returns_shoulder_lift_and_wrist_flex() -> None:
    overlay = IdleOverlay()
    offsets = overlay.sample(0.0)
    assert set(offsets) == {"shoulder_lift", "wrist_flex"}


def test_bounded_amplitude() -> None:
    overlay = IdleOverlay(amplitude_deg=2.0)
    max_rad = math.radians(2.0)
    for t in [i * 0.05 for i in range(400)]:  # several full periods
        offsets = overlay.sample(t)
        assert abs(offsets["shoulder_lift"]) <= max_rad + 1e-9
        assert abs(offsets["wrist_flex"]) <= 0.5 * max_rad + 1e-9


def test_periodic() -> None:
    overlay = IdleOverlay(period_s=4.0)
    a = overlay.sample(1.7)
    b = overlay.sample(1.7 + 4.0)
    assert a["shoulder_lift"] == pytest.approx(b["shoulder_lift"])
    assert a["wrist_flex"] == pytest.approx(b["wrist_flex"])


def test_zero_at_t_zero() -> None:
    overlay = IdleOverlay()
    offsets = overlay.sample(0.0)
    assert abs(offsets["shoulder_lift"]) < 1e-9


def test_reenabling_restores_output() -> None:
    overlay = IdleOverlay()
    overlay.set_enabled(False)
    assert overlay.sample(1.0) == {}
    overlay.set_enabled(True)
    assert overlay.sample(1.0) != {}

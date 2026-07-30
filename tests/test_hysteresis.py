from __future__ import annotations

from lelamp.perception.hysteresis import HysteresisConfig, HysteresisGate

CFG = HysteresisConfig(
    engage_threshold=0.7, engage_hold_ms=400.0, disengage_threshold=0.3, disengage_hold_ms=1500.0
)


def test_defaults_match_spec() -> None:
    config = HysteresisConfig()
    assert config.engage_threshold == 0.7
    assert config.engage_hold_ms == 400.0
    assert config.disengage_threshold == 0.3
    assert config.disengage_hold_ms == 1500.0


def test_starts_disengaged() -> None:
    gate = HysteresisGate(CFG)
    assert gate.engaged is False


def test_engages_after_sustained_high_score() -> None:
    gate = HysteresisGate(CFG)
    assert gate.update(0.9, timestamp=0.0) is None
    assert gate.update(0.9, timestamp=0.399) is None
    transition = gate.update(0.9, timestamp=0.4)
    assert transition is not None
    assert transition.engaged is True
    assert gate.engaged is True


def test_does_not_engage_if_score_never_sustained() -> None:
    gate = HysteresisGate(CFG)
    assert gate.update(0.9, timestamp=0.0) is None
    assert gate.update(0.9, timestamp=0.2) is None
    # dips below engage_threshold before hold completes -> resets the timer
    assert gate.update(0.5, timestamp=0.25) is None
    assert gate.update(0.9, timestamp=0.3) is None
    assert gate.update(0.9, timestamp=0.5) is None  # only 0.2s since the reset, not enough
    assert gate.engaged is False


def test_disengages_after_sustained_low_score() -> None:
    gate = HysteresisGate(CFG)
    gate.update(0.9, timestamp=0.0)
    gate.update(0.9, timestamp=0.4)
    assert gate.engaged is True
    assert gate.update(0.1, timestamp=1.0) is None
    assert gate.update(0.1, timestamp=2.499) is None
    transition = gate.update(0.1, timestamp=2.5)
    assert transition is not None
    assert transition.engaged is False
    assert gate.engaged is False


def test_dead_zone_scores_do_not_trigger_either_direction() -> None:
    gate = HysteresisGate(CFG)
    for t in range(0, 20):
        assert gate.update(0.5, timestamp=t * 1.0) is None
    assert gate.engaged is False


def test_threshold_boundary_is_strict_greater_than() -> None:
    gate = HysteresisGate(CFG)
    # exactly at threshold does not count as crossing
    assert gate.update(0.7, timestamp=0.0) is None
    assert gate.update(0.7, timestamp=1.0) is None
    assert gate.engaged is False
    assert gate.update(0.70001, timestamp=1.0) is None
    assert gate.update(0.70001, timestamp=1.41) is not None


def test_threshold_boundary_strict_less_than_for_disengage() -> None:
    gate = HysteresisGate(CFG)
    gate.update(0.9, timestamp=0.0)
    gate.update(0.9, timestamp=0.4)
    assert gate.engaged is True
    assert gate.update(0.3, timestamp=1.0) is None  # exactly at threshold, not below
    assert gate.update(0.3, timestamp=3.0) is None
    assert gate.engaged is True


def test_flapping_input_never_engages() -> None:
    gate = HysteresisGate(CFG)
    t = 0.0
    for _ in range(50):
        assert gate.update(0.9, timestamp=t) is None
        t += 0.05
        assert gate.update(0.1, timestamp=t) is None
        t += 0.05
    assert gate.engaged is False


def test_gap_in_stream_still_requires_continuous_dwell() -> None:
    gate = HysteresisGate(CFG)
    assert gate.update(0.9, timestamp=0.0) is None
    # large gap (dropped frames), score resumes high -- the elapsed wall time
    # since the candidate started is what matters, not frame count
    transition = gate.update(0.9, timestamp=5.0)
    assert transition is not None
    assert transition.engaged is True


def test_rapid_reengagement_after_disengage() -> None:
    gate = HysteresisGate(CFG)
    gate.update(0.9, timestamp=0.0)
    gate.update(0.9, timestamp=0.4)
    assert gate.engaged is True
    gate.update(0.1, timestamp=1.0)
    gate.update(0.1, timestamp=2.5)
    assert gate.engaged is False
    # immediately re-engage
    assert gate.update(0.9, timestamp=2.5) is None
    transition = gate.update(0.9, timestamp=2.91)
    assert transition is not None
    assert transition.engaged is True


def test_transition_is_edge_triggered_not_per_frame() -> None:
    gate = HysteresisGate(CFG)
    gate.update(0.9, timestamp=0.0)
    transition = gate.update(0.9, timestamp=0.4)
    assert transition is not None
    # staying engaged with continued high score never re-fires
    assert gate.update(0.9, timestamp=0.5) is None
    assert gate.update(0.9, timestamp=10.0) is None


def test_debug_state_tracks_hold_progress() -> None:
    gate = HysteresisGate(CFG)
    gate.update(0.9, timestamp=0.0)
    state = gate.debug_state(timestamp=0.2)
    assert state.engaged is False
    assert state.candidate is True
    assert 190 < state.hold_elapsed_ms < 210
    assert state.hold_target_ms == CFG.engage_hold_ms

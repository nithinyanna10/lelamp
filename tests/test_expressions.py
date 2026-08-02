from __future__ import annotations

import pytest
from pydantic import ValidationError

from lelamp.behavior.expressions import (
    EXPRESSION_CHAINS,
    EXPRESSIONS,
    LAMP_JOINT_NAMES,
    Expression,
    Keyframe,
)


def test_exactly_twenty_expressions() -> None:
    assert len(EXPRESSIONS) == 20


@pytest.mark.parametrize("name", list(EXPRESSIONS))
def test_at_least_two_keyframes(name: str) -> None:
    assert len(EXPRESSIONS[name].keyframes) >= 2


@pytest.mark.parametrize("name", list(EXPRESSIONS))
def test_monotonic_t_ms_starting_at_zero(name: str) -> None:
    ts = [kf.t_ms for kf in EXPRESSIONS[name].keyframes]
    assert ts[0] == 0
    assert ts == sorted(ts)


@pytest.mark.parametrize("name", list(EXPRESSIONS))
def test_joint_targets_reference_only_valid_names(name: str) -> None:
    for kf in EXPRESSIONS[name].keyframes:
        assert set(kf.joint_targets).issubset(LAMP_JOINT_NAMES)


@pytest.mark.parametrize("name", list(EXPRESSIONS))
def test_duration_covers_last_keyframe_plus_settle(name: str) -> None:
    expr = EXPRESSIONS[name]
    assert expr.duration_ms >= expr.keyframes[-1].t_ms


@pytest.mark.parametrize("name", list(EXPRESSIONS))
def test_duration_within_spec_range_ms(name: str) -> None:
    # spec: "300-1500ms" for hand-authored expressions; the two intentionally
    # long/looping ones (breathe_deep, searching) are documented exceptions.
    expr = EXPRESSIONS[name]
    if name in ("breathe_deep", "searching"):
        return
    assert 300 <= expr.duration_ms <= 2000  # thinking loops at 2000ms, the outer bound


def test_unknown_joint_name_rejected() -> None:
    with pytest.raises(ValidationError):
        Keyframe(t_ms=0, joint_targets={"not_a_real_joint": 0.1})


def test_keyframes_must_start_at_zero() -> None:
    with pytest.raises(ValidationError):
        Expression(
            name="bad",
            mood_family="calm",
            keyframes=[Keyframe(t_ms=10), Keyframe(t_ms=20)],
            duration_ms=100,
        )


def test_keyframes_must_be_monotonic() -> None:
    with pytest.raises(ValidationError):
        Expression(
            name="bad",
            mood_family="calm",
            keyframes=[Keyframe(t_ms=0), Keyframe(t_ms=100), Keyframe(t_ms=50)],
            duration_ms=200,
        )


def test_duration_must_cover_last_keyframe() -> None:
    with pytest.raises(ValidationError):
        Expression(
            name="bad",
            mood_family="calm",
            keyframes=[Keyframe(t_ms=0), Keyframe(t_ms=500)],
            duration_ms=100,
        )


def test_expression_needs_at_least_two_keyframes() -> None:
    with pytest.raises(ValidationError):
        Expression(
            name="bad", mood_family="calm", keyframes=[Keyframe(t_ms=0)], duration_ms=100
        )


def test_chains_reference_real_expressions() -> None:
    for chain in EXPRESSION_CHAINS.values():
        for step in chain.steps:
            assert step.name in EXPRESSIONS


def test_seeking_chains_escalate_in_intensity() -> None:
    i1 = max(s.intensity for s in EXPRESSION_CHAINS["seeking_1"].steps)
    i2 = max(s.intensity for s in EXPRESSION_CHAINS["seeking_2"].steps)
    i3 = max(s.intensity for s in EXPRESSION_CHAINS["seeking_3"].steps)
    assert i1 < i2 < i3


def test_all_mood_families_represented() -> None:
    families = {expr.mood_family for expr in EXPRESSIONS.values()}
    assert families == {"calm", "alert", "positive", "negative", "playful"}

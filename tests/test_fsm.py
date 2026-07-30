from __future__ import annotations

from lelamp.perception.hysteresis import EngagementTransition
from lelamp.state.fsm import LampFSM, LampState


def test_fsm_starts_idle() -> None:
    fsm = LampFSM()
    assert fsm.state == LampState.IDLE


def test_engaged_transition_moves_idle_to_engaged() -> None:
    fsm = LampFSM()
    transition = fsm.on_engagement_transition(EngagementTransition(timestamp=1.0, engaged=True))
    assert transition is not None
    assert transition.from_state == LampState.IDLE
    assert transition.to_state == LampState.ENGAGED
    assert transition.reason == "gaze_engaged"
    assert fsm.state == LampState.ENGAGED


def test_disengaged_transition_moves_engaged_to_idle() -> None:
    fsm = LampFSM()
    fsm.on_engagement_transition(EngagementTransition(timestamp=1.0, engaged=True))
    transition = fsm.on_engagement_transition(EngagementTransition(timestamp=2.0, engaged=False))
    assert transition is not None
    assert transition.from_state == LampState.ENGAGED
    assert transition.to_state == LampState.IDLE
    assert transition.reason == "gaze_disengaged"
    assert fsm.state == LampState.IDLE


def test_redundant_engaged_transition_is_a_noop() -> None:
    fsm = LampFSM()
    fsm.on_engagement_transition(EngagementTransition(timestamp=1.0, engaged=True))
    assert fsm.on_engagement_transition(EngagementTransition(timestamp=1.5, engaged=True)) is None
    assert fsm.state == LampState.ENGAGED


def test_disengaged_transition_while_already_idle_is_a_noop() -> None:
    fsm = LampFSM()
    assert fsm.on_engagement_transition(EngagementTransition(timestamp=1.0, engaged=False)) is None
    assert fsm.state == LampState.IDLE

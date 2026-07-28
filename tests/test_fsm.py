from __future__ import annotations

import pytest

from lelamp.perception.face_gaze import GazeEvent
from lelamp.state.fsm import HysteresisConfig, LampFSM, LampState


def test_fsm_starts_idle() -> None:
    fsm = LampFSM()
    assert fsm.state == LampState.IDLE


def test_hysteresis_defaults_match_spec() -> None:
    config = HysteresisConfig()
    assert config.engage_threshold == 0.7
    assert config.engage_duration_s == 0.4
    assert config.disengage_threshold == 0.3
    assert config.disengage_duration_s == 1.5


def test_on_gaze_event_not_yet_implemented() -> None:
    fsm = LampFSM()
    event = GazeEvent(frame_id=1, timestamp=0.0, gaze_score=0.9, num_faces=1)
    with pytest.raises(NotImplementedError):
        fsm.on_gaze_event(event)

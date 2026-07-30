from __future__ import annotations

from lelamp.behavior.motor import DEFAULT_HOME

# Hardcoded joint vectors for step 2. Full expression library comes in step 3;
# this is just enough to prove the perception -> FSM -> motor loop end-to-end.
# [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper], radians.
WAKE_POSE: list[float] = [0.0, -0.6, 0.4, 0.3, 0.0, 0.4]
WAKE_DURATION_S = 0.5

HOME_POSE: list[float] = DEFAULT_HOME
HOME_DURATION_S = 0.8

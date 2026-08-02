from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

from lelamp.behavior.motor import DEFAULT_HOME

# Hardcoded joint vectors for step 2. Kept as-is: main.py's step-2 FSM wiring
# (LampFSM -> WAKE/HOME) still imports these directly and is not touched in
# step 3 ("no FSM changes"). The 20-expression system below is additive.
# [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper], radians.
WAKE_POSE: list[float] = [0.0, -0.6, 0.4, 0.3, 0.0, 0.4]
WAKE_DURATION_S = 0.5

HOME_POSE: list[float] = DEFAULT_HOME
HOME_DURATION_S = 0.8


def _deg(degrees: float) -> float:
    return math.radians(degrees)


# Lamp-role names, in the REAL physical channel order (matches DEFAULT_JOINT_LIMITS /
# every move_to() call in behavior/motor.py) -- NOT the order roles happen to be
# listed in prose. See assets/so_arm100/README.md for the full mapping table and
# the wrist_yaw physical-axis caveat (it's closer to a tilt than an independent
# left-right yaw; kept anyway since renaming the MJCF joint isn't in scope here).
LAMP_JOINT_NAMES: tuple[str, str, str, str, str, str] = (
    "base_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "wrist_yaw",
)


def lamp_targets_to_vector(targets: dict[str, float], base: list[float]) -> list[float]:
    vector = list(base)
    for name, value in targets.items():
        vector[LAMP_JOINT_NAMES.index(name)] = value
    return vector


class Keyframe(BaseModel):
    t_ms: int
    joint_targets: dict[str, float] = {}
    light_intensity: float | None = None
    light_color: tuple[float, float, float] | None = None
    audio_cue: str | None = None

    @field_validator("joint_targets")
    @classmethod
    def _valid_joint_names(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = set(v) - set(LAMP_JOINT_NAMES)
        if unknown:
            raise ValueError(f"unknown joint name(s): {sorted(unknown)}")
        return v


class Expression(BaseModel):
    name: str
    mood_family: Literal["calm", "alert", "positive", "negative", "playful"]
    keyframes: list[Keyframe]
    duration_ms: int
    interruptible: bool = True
    return_to_idle: bool = True
    loop: bool = False
    tags: list[str] = []

    @field_validator("keyframes")
    @classmethod
    def _at_least_two_keyframes(cls, v: list[Keyframe]) -> list[Keyframe]:
        if len(v) < 2:
            raise ValueError("expression needs >= 2 keyframes")
        return v

    @model_validator(mode="after")
    def _monotonic_and_settled(self) -> Expression:
        ts = [kf.t_ms for kf in self.keyframes]
        if ts != sorted(ts) or ts[0] != 0:
            raise ValueError("keyframe t_ms must start at 0 and be non-decreasing")
        if self.duration_ms < ts[-1]:
            raise ValueError("duration_ms must be >= last keyframe's t_ms")
        return self


class ExpressionChainStep(BaseModel):
    name: str
    intensity: float = 1.0


class ExpressionChain(BaseModel):
    name: str
    steps: list[ExpressionChainStep]
    audio_cue: str | None = None


# ----------------------------------------------------------------------------
# The 20 expressions. All keyframes anchor at t_ms=0 with an empty joint_targets
# dict -- that keyframe holds whatever the arm's actual pose is when play()
# starts (see expression_player.py), not a hardcoded value. Every later keyframe
# only needs to name the joints that change; the rest hold their previous value.
#
# Sign convention: negative wrist_flex = tilt/nod down, positive = up. Positive
# base_pan/wrist_roll/wrist_yaw are arbitrary but used consistently below.
# ----------------------------------------------------------------------------

EXPRESSIONS: dict[str, Expression] = {
    "home": Expression(
        name="home",
        mood_family="calm",
        keyframes=[
            Keyframe(t_ms=0),
            Keyframe(t_ms=450, joint_targets=dict.fromkeys(LAMP_JOINT_NAMES, 0.0)),
        ],
        duration_ms=600,
        return_to_idle=False,
        tags=["neutral", "default"],
    ),
    "breathe_deep": Expression(
        name="breathe_deep",
        mood_family="calm",
        keyframes=[
            Keyframe(t_ms=0),
            Keyframe(t_ms=1250, joint_targets={"shoulder_lift": _deg(5), "wrist_flex": _deg(3)}),
            Keyframe(t_ms=2500, joint_targets={"shoulder_lift": 0.0, "wrist_flex": 0.0}),
            Keyframe(t_ms=3750, joint_targets={"shoulder_lift": _deg(-5), "wrist_flex": _deg(-3)}),
            Keyframe(t_ms=5000, joint_targets={"shoulder_lift": 0.0, "wrist_flex": 0.0}),
        ],
        duration_ms=5000,
        loop=True,
        return_to_idle=False,
        tags=["idle", "calm"],
    ),
    "sleep": Expression(
        name="sleep",
        mood_family="calm",
        keyframes=[
            Keyframe(t_ms=0),
            Keyframe(
                t_ms=1200,
                joint_targets={"shoulder_lift": _deg(-40), "wrist_flex": _deg(-30)},
                light_intensity=0.05,
                light_color=(1.0, 0.7, 0.4),
            ),
        ],
        duration_ms=1400,
        return_to_idle=False,
        tags=["sleep"],
    ),
    "wake": Expression(
        name="wake",
        mood_family="calm",
        keyframes=[
            Keyframe(t_ms=0),
            Keyframe(t_ms=450, joint_targets={"shoulder_lift": _deg(-15)}, light_intensity=0.4),
            Keyframe(
                t_ms=750,
                joint_targets=dict.fromkeys(LAMP_JOINT_NAMES, 0.0),
                light_intensity=0.9,
                audio_cue="chime_soft.wav",
            ),
        ],
        duration_ms=900,
        return_to_idle=True,
        tags=["wake"],
    ),
    "notice_user": Expression(
        name="notice_user",
        mood_family="alert",
        keyframes=[
            Keyframe(t_ms=0, light_intensity=0.3),
            Keyframe(
                t_ms=200,
                joint_targets={
                    "base_pan": _deg(20),
                    "wrist_yaw": _deg(15),
                    "shoulder_lift": _deg(8),
                },
                light_intensity=0.9,
                audio_cue="chime_soft.wav",
            ),
        ],
        duration_ms=500,
        return_to_idle=False,
        tags=["engage", "alert"],
    ),
    "listen": Expression(
        name="listen",
        mood_family="alert",
        keyframes=[
            Keyframe(t_ms=0),
            Keyframe(t_ms=400, joint_targets={"wrist_flex": _deg(5)}, light_intensity=0.85),
        ],
        duration_ms=550,
        return_to_idle=False,
        tags=["listening"],
    ),
    "thinking": Expression(
        name="thinking",
        mood_family="alert",
        keyframes=[
            Keyframe(t_ms=0),
            Keyframe(
                t_ms=200,
                joint_targets={"wrist_flex": _deg(10), "wrist_yaw": _deg(15)},
                light_intensity=0.65,
            ),
            Keyframe(t_ms=700, light_intensity=0.8),
            Keyframe(t_ms=1200, light_intensity=0.5),
            Keyframe(t_ms=1700, light_intensity=0.8),
            Keyframe(t_ms=2000, light_intensity=0.65),
        ],
        duration_ms=2000,
        loop=True,
        return_to_idle=True,
        tags=["thinking", "alert"],
    ),
    "searching": Expression(
        name="searching",
        mood_family="alert",
        keyframes=[
            Keyframe(t_ms=0),
            Keyframe(t_ms=875, joint_targets={"base_pan": _deg(30), "wrist_flex": _deg(3)}),
            Keyframe(t_ms=1750, joint_targets={"base_pan": 0.0, "wrist_flex": _deg(-3)}),
            Keyframe(t_ms=2625, joint_targets={"base_pan": _deg(-30), "wrist_flex": _deg(3)}),
            Keyframe(t_ms=3500, joint_targets={"base_pan": 0.0, "wrist_flex": 0.0}),
        ],
        duration_ms=3500,
        return_to_idle=True,
        tags=["searching", "alert"],
    ),
    "happy_nod": Expression(
        name="happy_nod",
        mood_family="positive",
        keyframes=[
            Keyframe(t_ms=0),
            Keyframe(
                t_ms=150,
                joint_targets={"wrist_flex": _deg(-15), "shoulder_lift": _deg(3)},
                light_intensity=1.0,
                audio_cue="chirp_cheerful.wav",
            ),
            Keyframe(
                t_ms=300,
                joint_targets={"wrist_flex": _deg(15), "shoulder_lift": 0.0},
                light_intensity=0.8,
            ),
            Keyframe(
                t_ms=450,
                joint_targets={"wrist_flex": _deg(-15), "shoulder_lift": _deg(3)},
                light_intensity=1.0,
            ),
            Keyframe(
                t_ms=600,
                joint_targets={"wrist_flex": 0.0, "shoulder_lift": 0.0},
                light_intensity=0.85,
            ),
        ],
        duration_ms=800,
        return_to_idle=True,
        tags=["positive", "nod"],
    ),
    "excited": Expression(
        name="excited",
        mood_family="positive",
        keyframes=[
            Keyframe(t_ms=0),
            Keyframe(
                t_ms=120,
                joint_targets={"wrist_flex": _deg(-18), "base_pan": _deg(5)},
                light_intensity=1.0,
            ),
            Keyframe(
                t_ms=240,
                joint_targets={"wrist_flex": _deg(18), "base_pan": _deg(-5)},
                light_intensity=0.7,
            ),
            Keyframe(
                t_ms=360,
                joint_targets={"wrist_flex": _deg(-18), "base_pan": _deg(5)},
                light_intensity=1.0,
            ),
            Keyframe(
                t_ms=480,
                joint_targets={"wrist_flex": _deg(18), "base_pan": _deg(-5)},
                light_intensity=0.7,
            ),
            Keyframe(
                t_ms=600,
                joint_targets={"wrist_flex": _deg(-18), "base_pan": 0.0},
                light_intensity=1.0,
            ),
            Keyframe(t_ms=700, joint_targets={"wrist_flex": 0.0}, light_intensity=0.85),
        ],
        duration_ms=900,
        return_to_idle=True,
        tags=["positive", "excited"],
    ),
    "greeting": Expression(
        name="greeting",
        mood_family="positive",
        keyframes=[
            Keyframe(t_ms=0),
            Keyframe(
                t_ms=300,
                joint_targets={"wrist_flex": _deg(20), "shoulder_lift": _deg(5)},
                light_intensity=0.95,
                audio_cue="chime_warm.wav",
            ),
            Keyframe(t_ms=550, joint_targets={"wrist_flex": 0.0, "shoulder_lift": 0.0}),
        ],
        duration_ms=700,
        return_to_idle=True,
        tags=["greeting", "positive"],
    ),
    "acknowledge": Expression(
        name="acknowledge",
        mood_family="positive",
        keyframes=[
            Keyframe(t_ms=0),
            Keyframe(t_ms=150, joint_targets={"wrist_flex": _deg(-10)}),
            Keyframe(t_ms=280, joint_targets={"wrist_flex": 0.0}),
        ],
        duration_ms=400,
        return_to_idle=True,
        tags=["acknowledge"],
    ),
    "confused": Expression(
        name="confused",
        mood_family="negative",
        keyframes=[
            Keyframe(t_ms=0),
            Keyframe(t_ms=200, joint_targets={"wrist_roll": _deg(15)}, light_intensity=0.7),
            Keyframe(t_ms=800, joint_targets={"wrist_roll": _deg(15)}),
            Keyframe(t_ms=1000, joint_targets={"wrist_roll": _deg(-15)}),
            Keyframe(t_ms=1500, joint_targets={"wrist_roll": _deg(-15)}),
        ],
        duration_ms=1500,
        return_to_idle=True,
        tags=["confused", "negative"],
    ),
    "shake_no": Expression(
        name="shake_no",
        mood_family="negative",
        keyframes=[
            Keyframe(t_ms=0),
            Keyframe(t_ms=100, joint_targets={"base_pan": _deg(10)}, light_intensity=0.5),
            Keyframe(t_ms=250, joint_targets={"base_pan": _deg(-10)}),
            Keyframe(t_ms=400, joint_targets={"base_pan": _deg(10)}),
            Keyframe(t_ms=550, joint_targets={"base_pan": _deg(-10)}),
            Keyframe(t_ms=650, joint_targets={"base_pan": 0.0}, light_intensity=0.85),
        ],
        duration_ms=700,
        return_to_idle=True,
        tags=["negative", "shake"],
    ),
    "disappointed": Expression(
        name="disappointed",
        mood_family="negative",
        keyframes=[
            Keyframe(t_ms=0),
            Keyframe(
                t_ms=800,
                joint_targets={"shoulder_lift": _deg(-10), "wrist_flex": _deg(-20)},
                light_intensity=0.4,
            ),
        ],
        duration_ms=1000,
        return_to_idle=True,
        tags=["disappointed", "negative"],
    ),
    "shy": Expression(
        name="shy",
        mood_family="negative",
        keyframes=[
            Keyframe(t_ms=0),
            Keyframe(
                t_ms=800,
                joint_targets={"base_pan": _deg(-20), "wrist_flex": _deg(-10)},
                light_intensity=0.5,
            ),
        ],
        duration_ms=1000,
        return_to_idle=False,
        tags=["shy", "negative"],
    ),
    "curious_tilt": Expression(
        name="curious_tilt",
        mood_family="playful",
        keyframes=[
            Keyframe(t_ms=0),
            Keyframe(t_ms=550, joint_targets={"wrist_roll": _deg(25), "shoulder_lift": _deg(5)}),
        ],
        duration_ms=700,
        return_to_idle=False,
        tags=["curious", "playful"],
    ),
    "peek": Expression(
        name="peek",
        mood_family="playful",
        keyframes=[
            Keyframe(t_ms=0, light_intensity=0.2),
            Keyframe(
                t_ms=1000,
                joint_targets={"base_pan": _deg(30), "wrist_flex": _deg(15)},
                light_intensity=0.7,
            ),
        ],
        duration_ms=1200,
        return_to_idle=False,
        tags=["peek", "playful"],
    ),
    "bounce": Expression(
        name="bounce",
        mood_family="playful",
        keyframes=[
            Keyframe(t_ms=0),
            Keyframe(t_ms=200, joint_targets={"shoulder_lift": _deg(8), "elbow_flex": _deg(-8)}),
            Keyframe(t_ms=400, joint_targets={"shoulder_lift": _deg(-8), "elbow_flex": _deg(8)}),
            Keyframe(t_ms=600, joint_targets={"shoulder_lift": _deg(8), "elbow_flex": _deg(-8)}),
            Keyframe(t_ms=800, joint_targets={"shoulder_lift": 0.0, "elbow_flex": 0.0}),
        ],
        duration_ms=800,
        return_to_idle=True,
        tags=["bounce", "playful"],
    ),
    "spin": Expression(
        name="spin",
        mood_family="playful",
        keyframes=[
            Keyframe(t_ms=0),
            Keyframe(
                t_ms=100,
                joint_targets={"base_pan": _deg(-90), "wrist_roll": _deg(-15)},
                light_intensity=1.0,
            ),
            Keyframe(
                t_ms=700,
                joint_targets={"base_pan": _deg(90), "wrist_roll": _deg(15)},
                light_intensity=0.6,
            ),
            Keyframe(
                t_ms=1100, joint_targets={"base_pan": 0.0, "wrist_roll": 0.0}, light_intensity=1.0
            ),
        ],
        duration_ms=1200,
        return_to_idle=True,
        tags=["spin", "celebration", "playful"],
    ),
}

EXPRESSION_CHAINS: dict[str, ExpressionChain] = {
    "seeking_1": ExpressionChain(
        name="seeking_1", steps=[ExpressionChainStep(name="peek", intensity=0.6)]
    ),
    "seeking_2": ExpressionChain(
        name="seeking_2",
        steps=[
            ExpressionChainStep(name="peek", intensity=1.0),
            ExpressionChainStep(name="curious_tilt", intensity=1.0),
        ],
    ),
    "seeking_3": ExpressionChain(
        name="seeking_3",
        steps=[
            ExpressionChainStep(name="spin", intensity=1.5),
            ExpressionChainStep(name="bounce", intensity=1.5),
        ],
        audio_cue="chirp_loud.wav",
    ),
}

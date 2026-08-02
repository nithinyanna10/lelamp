from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator

import pytest

import lelamp.behavior.expression_player as expression_player_module
from lelamp.behavior.expression_player import ExpressionPlayer
from lelamp.behavior.expressions import EXPRESSIONS, LAMP_JOINT_NAMES, lamp_targets_to_vector
from lelamp.behavior.idle_overlay import IdleOverlay

# Real durations sum to ~18s across the 18 non-looping expressions; the player's
# tick loop times itself against time.monotonic() (busy-checked, not slept), so
# there's no way to speed it up except by accelerating the clock it reads. This
# monkeypatches only the module-under-test's `time.monotonic`, not FakeMotorBackend
# (which has no internal timing) or motor.py's (unused here) -- so it's safe and
# doesn't change what's being verified, just how long verifying it takes.
_CLOCK_SPEED = 25.0


@pytest.fixture(autouse=True)
def fast_clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    real_monotonic = time.monotonic
    t0 = real_monotonic()

    def fast_monotonic() -> float:
        return t0 + (real_monotonic() - t0) * _CLOCK_SPEED

    monkeypatch.setattr(expression_player_module.time, "monotonic", fast_monotonic)
    yield


@pytest.fixture(autouse=True)
def no_real_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(expression_player_module, "_play_audio_cue", lambda *_a, **_kw: None)


def _expected_final_vector(name: str, intensity: float, reference: list[float]) -> list[float]:
    expr = EXPRESSIONS[name]
    running = list(reference)
    for kf in expr.keyframes:
        running = lamp_targets_to_vector(kf.joint_targets, running)
    return [reference[i] + (running[i] - reference[i]) * intensity for i in range(6)]


NON_LOOPING = [name for name, expr in EXPRESSIONS.items() if not expr.loop]
LOOPING = [name for name, expr in EXPRESSIONS.items() if expr.loop]


@pytest.mark.parametrize("name", NON_LOOPING)
async def test_final_state_matches_last_keyframe(name: str, mock_motor_backend) -> None:
    idle = IdleOverlay()
    idle.set_enabled(False)  # isolate the keyframe-matching check from overlay noise
    player = ExpressionPlayer(mock_motor_backend, idle)

    reference = [0.0] * 6
    await player.play(name)

    expected = _expected_final_vector(name, 1.0, reference)
    actual = mock_motor_backend.sent[-1]
    for e, a in zip(expected, actual, strict=True):
        assert a == pytest.approx(e, abs=1e-6)


@pytest.mark.parametrize("name", LOOPING)
async def test_looping_expressions_run_until_stopped(name: str, mock_motor_backend) -> None:
    player = ExpressionPlayer(mock_motor_backend, IdleOverlay())
    task = asyncio.create_task(player.play(name))
    await asyncio.sleep(0.05)  # >> one full (accelerated) loop cycle
    assert not task.done()  # confirms it's actually looping, not finishing early
    assert player.current() == name

    await player.stop()
    assert player.current() is None
    assert task.done()


async def test_preemption_cancels_and_switches_to_new_expression(mock_motor_backend) -> None:
    player = ExpressionPlayer(mock_motor_backend, IdleOverlay())
    long_task = asyncio.create_task(player.play("searching"))  # 3500ms
    await asyncio.sleep(0.02)

    await player.preempt("acknowledge")

    assert player.current() == "acknowledge"
    expected = _expected_final_vector("acknowledge", 1.0, [0.0] * 6)
    actual = mock_motor_backend.sent[-1]
    for e, a in zip(expected, actual, strict=True):
        assert a == pytest.approx(e, abs=1e-6)

    await long_task  # should already be resolved (cancelled), not hang


async def test_intensity_scales_keyframe_amplitude(mock_motor_backend) -> None:
    idle = IdleOverlay()
    idle.set_enabled(False)
    player = ExpressionPlayer(mock_motor_backend, idle)

    await player.play("curious_tilt", intensity=1.0)
    at_1x = mock_motor_backend.sent[-1][LAMP_JOINT_NAMES.index("wrist_roll")]

    await player.play("home")  # reset back to origin between measurements
    await player.play("curious_tilt", intensity=2.0)
    at_2x = mock_motor_backend.sent[-1][LAMP_JOINT_NAMES.index("wrist_roll")]

    assert at_2x == pytest.approx(2.0 * at_1x, rel=1e-3)


async def test_idle_overlay_composes_additively_not_replacing(mock_motor_backend) -> None:
    idle_off = IdleOverlay()
    idle_off.set_enabled(False)
    player_off = ExpressionPlayer(mock_motor_backend, idle_off)
    await player_off.play("listen")
    without_overlay = mock_motor_backend.sent[-1][LAMP_JOINT_NAMES.index("wrist_flex")]

    idle_on = IdleOverlay(amplitude_deg=10.0, period_s=0.02)  # exaggerated, fast test signal
    player_on = ExpressionPlayer(mock_motor_backend, idle_on)
    await player_on.play("listen")
    with_overlay = mock_motor_backend.sent[-1][LAMP_JOINT_NAMES.index("wrist_flex")]

    # Composed additively means the overlay perturbs the result away from the
    # pure keyframe target -- if it replaced the target instead, this could
    # coincidentally match, but across many ticks with a fast-varying overlay
    # the accumulated final tick's value will differ from the un-perturbed one.
    assert with_overlay != pytest.approx(without_overlay, abs=1e-9)


async def test_look_at_face_overrides_base_pan_and_wrist_yaw_only(mock_motor_backend) -> None:
    player = ExpressionPlayer(mock_motor_backend, IdleOverlay())
    player.look_at_face((0.5, -0.5))
    await player.play("acknowledge")

    final = mock_motor_backend.sent[-1]
    expected_base_pan = 0.5 * expression_player_module._LOOK_AT_BASE_PAN_RANGE_RAD
    expected_wrist_yaw = -0.5 * expression_player_module._LOOK_AT_WRIST_YAW_RANGE_RAD
    assert final[LAMP_JOINT_NAMES.index("base_pan")] == pytest.approx(expected_base_pan)
    assert final[LAMP_JOINT_NAMES.index("wrist_yaw")] == pytest.approx(expected_wrist_yaw)
    # acknowledge's own wrist_flex motion should be unaffected by look_at_face
    assert final[LAMP_JOINT_NAMES.index("wrist_flex")] == pytest.approx(0.0, abs=1e-6)


async def test_current_is_none_before_any_play(mock_motor_backend) -> None:
    player = ExpressionPlayer(mock_motor_backend, IdleOverlay())
    assert player.current() is None

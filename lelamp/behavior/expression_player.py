from __future__ import annotations

import asyncio
import contextlib
import time
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import structlog

from lelamp.behavior.expressions import (
    EXPRESSION_CHAINS,
    EXPRESSIONS,
    HOME_POSE,
    LAMP_JOINT_NAMES,
    Expression,
    lamp_targets_to_vector,
)
from lelamp.behavior.idle_overlay import IdleOverlay
from lelamp.behavior.motor import MotorBackend, interpolate
from lelamp.telemetry import get_tracer

_tracer = get_tracer(__name__)
_log = structlog.get_logger(__name__)

AUDIO_DIR = Path("assets/audio")

# look_at_face maps normalized face position to these two joints (not the full
# 6-vector -- everything else keeps doing whatever the current expression/idle
# overlay says). wrist_yaw gets the *vertical* component here, not horizontal:
# its real rotation axis is closer to a tilt than an independent left-right yaw
# (see assets/so_arm100/README.md), so that's the physically sensible mapping
# even though base_pan+wrist_yaw are the two joints the spec names together.
_LOOK_AT_BASE_PAN_RANGE_RAD = 0.6
_LOOK_AT_WRIST_YAW_RANGE_RAD = 0.4


def _play_audio_cue(filename: str) -> None:
    path = AUDIO_DIR / filename
    if not path.exists():
        _log.warning("expression_player.audio_cue_missing", path=str(path))
        return
    try:
        import sounddevice as sd

        with wave.open(str(path), "rb") as f:
            frames = f.readframes(f.getnframes())
            sample_rate = f.getframerate()
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        sd.play(audio, sample_rate)  # non-blocking: returns immediately
    except Exception:
        _log.warning("expression_player.audio_cue_failed", path=str(path), exc_info=True)


class ExpressionPlayer:
    """Plays Expressions on top of a MotorBackend, with idle-breathing overlay
    and look_at_face composed in continuously (not just at keyframe boundaries).

    Composition trick: keyframe easing reuses motor.py's own `interpolate()`
    (smoothstep, already tested there -- not reimplemented here) to compute a
    base target every tick; the idle overlay's offset and any look_at_face
    override are added on top; the combined vector is sent via a fresh
    `motor.move_to(combined, duration_s=tick_dt)` each tick. Motor.move_to()'s
    own preemption (see motor.py's atomic done-event handling) means each
    tick's call cleanly supersedes the previous one -- no need to touch the
    motor backend to get continuous composition.
    """

    def __init__(
        self,
        motor: MotorBackend,
        idle_overlay: IdleOverlay,
        tick_hz: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        # clock is injectable so tests can accelerate expression playback without
        # touching the process-wide `time` module -- monkeypatching that globally
        # was tried and rejected: it also perturbs asyncio's own internal
        # scheduling (event loop deadlines, asyncio.sleep), which hung the test
        # suite rather than speeding it up.
        self._motor = motor
        self._idle = idle_overlay
        self._tick_dt = 1.0 / tick_hz
        self._clock = clock
        self._task: asyncio.Task[None] | None = None
        self._current_name: str | None = None
        self._current_interruptible: bool = True
        self._look_at: tuple[float, float] | None = None
        self._epoch = self._clock()

        # Tracked, not physically driven: no MotorBackend API exists for light
        # output (that would mean touching motor.py, out of scope this step).
        # Exposed for the demo script / a future light-driver hookup.
        self.current_light_intensity: float = 0.85
        self.current_light_color: tuple[float, float, float] = (1.0, 0.85, 0.6)

    def current(self) -> str | None:
        return self._current_name

    def look_at_face(self, face_xy_normalized: tuple[float, float] | None) -> None:
        self._look_at = face_xy_normalized

    async def play(self, name: str, intensity: float = 1.0) -> None:
        await self._dispatch(name, intensity, force=False)

    async def preempt(self, name: str, intensity: float = 1.0) -> None:
        await self._dispatch(name, intensity, force=True)

    async def play_chain(self, chain_name: str) -> None:
        chain = EXPRESSION_CHAINS[chain_name]
        if chain.audio_cue:
            _play_audio_cue(chain.audio_cue)
        for step in chain.steps:
            await self.play(step.name, step.intensity)

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._current_name = None

    async def _dispatch(self, name: str, intensity: float, force: bool) -> None:
        if self._task is not None and not self._task.done():
            if not force and not self._current_interruptible:
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
            else:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task

        expr = EXPRESSIONS[name]
        self._current_name = name
        self._current_interruptible = expr.interruptible
        self._task = asyncio.create_task(self._run(expr, intensity))
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    async def _run(self, expr: Expression, intensity: float) -> None:
        with _tracer.start_as_current_span("expression.play") as span:
            span.set_attribute("name", expr.name)
            span.set_attribute("intensity", intensity)
            t0 = self._clock()
            preempted = False
            try:
                await self._play_once(expr, intensity)
                while expr.loop:
                    await self._play_once(expr, intensity)
            except asyncio.CancelledError:
                preempted = True
                raise
            finally:
                span.set_attribute("duration_actual_ms", (self._clock() - t0) * 1000.0)
                span.set_attribute("preempted", preempted)

        if expr.return_to_idle and not preempted:
            # Route through _apply_look_at: without this, a lamp actively tracking a
            # face would visibly snap its head away from them the instant any
            # return_to_idle expression finishes -- look_at_face is documented as a
            # continuous override ("everything else continues"), and the home-blend
            # is not an exception to that.
            await self._motor.move_to(self._apply_look_at(list(HOME_POSE)), duration_s=0.4)

    def _apply_look_at(self, vector: list[float]) -> list[float]:
        if self._look_at is None:
            return vector
        x, y = self._look_at
        vector = list(vector)
        vector[LAMP_JOINT_NAMES.index("base_pan")] = x * _LOOK_AT_BASE_PAN_RANGE_RAD
        vector[LAMP_JOINT_NAMES.index("wrist_yaw")] = y * _LOOK_AT_WRIST_YAW_RANGE_RAD
        return vector

    async def _play_once(self, expr: Expression, intensity: float) -> None:
        state = await self._motor.get_state()
        reference = list(state.joint_angles)

        resolved_joints: list[list[float]] = []
        running = list(reference)
        resolved_light: list[float] = []
        light = self.current_light_intensity
        resolved_color: list[tuple[float, float, float]] = []
        color = self.current_light_color
        for kf in expr.keyframes:
            running = lamp_targets_to_vector(kf.joint_targets, running)
            resolved_joints.append(list(running))
            if kf.light_intensity is not None:
                light = kf.light_intensity
            resolved_light.append(light)
            if kf.light_color is not None:
                color = kf.light_color
            resolved_color.append(color)

        n_segments = len(expr.keyframes) - 1
        played_audio: set[int] = set()
        seg_idx = 0
        t_start = self._clock()
        last_kf_ms = expr.keyframes[-1].t_ms

        # Loop until duration_ms, not just the last keyframe's t_ms: the gap between
        # them is a deliberate settle buffer (see Expression's model validator) --
        # skipping it here made every expression with one finish measurably faster
        # than its spec'd duration, caught by timing play() against real durations.
        while True:
            elapsed_ms = (self._clock() - t_start) * 1000.0
            if elapsed_ms >= expr.duration_ms:
                break

            if elapsed_ms >= last_kf_ms:
                base_target = [
                    reference[i] + (resolved_joints[-1][i] - reference[i]) * intensity
                    for i in range(6)
                ]
                self.current_light_intensity = resolved_light[-1]
                self.current_light_color = resolved_color[-1]
            else:
                while seg_idx < n_segments - 1 and elapsed_ms >= expr.keyframes[seg_idx + 1].t_ms:
                    seg_idx += 1
                next_kf = expr.keyframes[seg_idx + 1]
                if next_kf.audio_cue and (seg_idx + 1) not in played_audio:
                    played_audio.add(seg_idx + 1)
                    _play_audio_cue(next_kf.audio_cue)

                kf_a = expr.keyframes[seg_idx]
                seg_duration_s = max(1e-6, (next_kf.t_ms - kf_a.t_ms) / 1000.0)
                seg_elapsed_s = max(0.0, elapsed_ms / 1000.0 - kf_a.t_ms / 1000.0)

                scaled_start = [
                    reference[i] + (resolved_joints[seg_idx][i] - reference[i]) * intensity
                    for i in range(6)
                ]
                scaled_end = [
                    reference[i] + (resolved_joints[seg_idx + 1][i] - reference[i]) * intensity
                    for i in range(6)
                ]
                base_target = interpolate(scaled_start, scaled_end, seg_elapsed_s, seg_duration_s)

                self.current_light_intensity = interpolate(
                    [resolved_light[seg_idx]],
                    [resolved_light[seg_idx + 1]],
                    seg_elapsed_s,
                    seg_duration_s,
                )[0]
                self.current_light_color = resolved_color[seg_idx + 1]

            idle_offset = self._idle.sample(self._clock() - self._epoch)
            combined = list(base_target)
            for joint_name, offset in idle_offset.items():
                combined[LAMP_JOINT_NAMES.index(joint_name)] += offset
            combined = self._apply_look_at(combined)

            await self._motor.move_to(combined, duration_s=self._tick_dt)
            # Explicit cooperative yield: real backends' move_to() already suspends
            # (awaits an asyncio.Event tied to the trajectory thread), so this is a
            # no-op there. But the player shouldn't rely on that for correctness --
            # a MotorBackend whose move_to() never actually awaits anything (a
            # perfectly valid mock; conftest.py's FakeMotorBackend is exactly this)
            # would otherwise let a looping expression's tick loop monopolize the
            # event loop forever, since nothing else -- including a stop() call from
            # another task -- can ever get scheduled to run. Found via a genuine
            # test hang, not theorized.
            await asyncio.sleep(0)

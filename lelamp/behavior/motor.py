from __future__ import annotations

import asyncio
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import mujoco
import numpy as np
import serial
from pydantic import BaseModel

from lelamp.behavior.feetech import angle_to_raw, build_sync_write_goal_position
from lelamp.telemetry import get_tracer

_tracer = get_tracer(__name__)

NUM_JOINTS = 6

# From TheRobotStudio/SO-ARM100 Simulation/SO101/so101_new_calib.xml <joint range="...">.
DEFAULT_JOINT_LIMITS: list[tuple[float, float]] = [
    (-1.9198621771937616, 1.9198621771937634),  # shoulder_pan
    (-1.7453292519943224, 1.7453292519943366),  # shoulder_lift
    (-1.69, 1.69),  # elbow_flex
    (-1.6580628494556928, 1.6580627293335335),  # wrist_flex
    (-2.7438472969992493, 2.841206309382605),  # wrist_roll
    (-0.17453297762778586, 1.7453291995659765),  # gripper
]
DEFAULT_HOME: list[float] = [0.0] * NUM_JOINTS
DEFAULT_MJCF_PATH = "assets/so_arm100/scene.xml"


class MotorState(BaseModel):
    joint_angles: list[float]
    timestamp: float
    is_moving: bool


def _ease_in_out(t: float) -> float:
    """Smoothstep: zero velocity at both ends of the trajectory."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def interpolate(
    start: list[float], end: list[float], elapsed_s: float, duration_s: float
) -> list[float]:
    if duration_s <= 0:
        return list(end)
    frac = _ease_in_out(elapsed_s / duration_s)
    return [s + (e - s) * frac for s, e in zip(start, end, strict=True)]


@dataclass
class _Trajectory:
    start: list[float]
    end: list[float]
    t0: float
    duration_s: float

    def sample(self, now: float) -> tuple[list[float], bool]:
        elapsed = now - self.t0
        done = elapsed >= self.duration_s
        angles = self.end if done else interpolate(self.start, self.end, elapsed, self.duration_s)
        return angles, done


class MotorBackend(ABC):
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def move_to(self, joint_angles: list[float], duration_s: float) -> None: ...

    @abstractmethod
    async def get_state(self) -> MotorState: ...

    @abstractmethod
    async def home(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...


class _ThreadedMotorBackend(MotorBackend):
    """Ease-in-out trajectory generation + a fixed-dt background control thread,
    shared by the serial and MuJoCo backends so primitives.py only ever calls
    move_to/get_state/home/stop -- it never generates a trajectory itself."""

    backend_name = "threaded"

    def __init__(
        self,
        joint_limits: list[tuple[float, float]],
        home_position: list[float],
        tick_hz: float = 100.0,
    ) -> None:
        self._joint_limits = joint_limits
        self._home_position = list(home_position)
        self._tick_hz = tick_hz
        self._lock = threading.Lock()
        self._current = list(home_position)
        self._actual: list[float] | None = None
        self._trajectory: _Trajectory | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._done_event: asyncio.Event | None = None

    @property
    def joint_limits(self) -> list[tuple[float, float]]:
        return self._joint_limits

    def _clamp(self, angles: list[float]) -> list[float]:
        # float(...): mujoco-derived joint_limits are numpy.float64, and min/max against
        # one silently promotes the whole result -- which OTel's span-attribute validator
        # then rejects (span.set_attribute has no Pydantic-style coercion).
        return [
            float(min(max(a, lo), hi))
            for a, (lo, hi) in zip(angles, self._joint_limits, strict=True)
        ]

    async def connect(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._on_connect()
        self._current = list(self._home_position)
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, name=f"{self.backend_name}-control", daemon=True
        )
        self._thread.start()

    def _run_loop(self) -> None:
        dt = 1.0 / self._tick_hz
        while self._running:
            now = time.monotonic()
            # The done-check and the _done_event pop must happen under the same lock
            # acquisition: popping it separately (after re-acquiring the lock) leaves a
            # window where a concurrent move_to() installs a *new* event, which this
            # stale `done=True` would then fire immediately -- completing a motion that
            # never ticked.
            event_to_fire: asyncio.Event | None = None
            with self._lock:
                traj = self._trajectory
                if traj is not None:
                    angles, done = traj.sample(now)
                    self._current = angles
                    if done:
                        self._trajectory = None
                else:
                    angles, done = self._current, False
                if done:
                    event_to_fire, self._done_event = self._done_event, None
            self._apply(angles)
            if event_to_fire is not None and self._loop is not None:
                self._loop.call_soon_threadsafe(event_to_fire.set)
            time.sleep(dt)

    def _report_actual(self, angles: list[float]) -> None:
        with self._lock:
            self._actual = angles

    async def move_to(self, joint_angles: list[float], duration_s: float) -> None:
        with _tracer.start_as_current_span("motor.move_to") as span:
            target = self._clamp(joint_angles)
            span.set_attribute("target_angles", target)
            span.set_attribute("duration_s", duration_s)
            span.set_attribute("backend", self.backend_name)
            event = asyncio.Event()
            with self._lock:
                start = list(self._current)
                old_event, self._done_event = self._done_event, event
                self._trajectory = _Trajectory(
                    start=start, end=target, t0=time.monotonic(), duration_s=duration_s
                )
            if old_event is not None and self._loop is not None:
                self._loop.call_soon_threadsafe(old_event.set)
            await event.wait()

    async def get_state(self) -> MotorState:
        with self._lock:
            angles = self._actual if self._actual is not None else self._current
            return MotorState(
                joint_angles=list(angles),
                timestamp=time.monotonic(),
                is_moving=self._trajectory is not None,
            )

    async def home(self) -> None:
        await self.move_to(self._home_position, duration_s=2.0)

    async def stop(self) -> None:
        with self._lock:
            self._trajectory = None
            event, self._done_event = self._done_event, None
        if event is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(event.set)

    async def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._on_close()

    @abstractmethod
    def _apply(self, angles: list[float]) -> None:
        """Called from the background thread every tick; push angles to hardware/sim."""

    def _on_connect(self) -> None:
        pass

    def _on_close(self) -> None:
        pass


class SerialMotorBackend(_ThreadedMotorBackend):
    """SO-ARM100 over USB serial, Feetech STS3215 sync-write per tick.

    ponytail: writes goal positions open-loop at tick_hz; doesn't read back
    Present_Position each tick (half-duplex round trips at 50Hz would eat the
    control budget). get_state() reports the commanded setpoint, not sensed
    position. Add a read-back thread if closed-loop telemetry is needed later.
    """

    backend_name = "serial"

    def __init__(
        self,
        port: str,
        baudrate: int = 1_000_000,
        joint_limits: list[tuple[float, float]] | None = None,
        home_position: list[float] | None = None,
        servo_ids: list[int] | None = None,
        tick_hz: float = 50.0,
    ) -> None:
        super().__init__(
            joint_limits=joint_limits or DEFAULT_JOINT_LIMITS,
            home_position=home_position or DEFAULT_HOME,
            tick_hz=tick_hz,
        )
        self.port = port
        self.baudrate = baudrate
        self._servo_ids = servo_ids or list(range(1, NUM_JOINTS + 1))
        self._serial: serial.Serial | None = None

    def _on_connect(self) -> None:
        self._serial = serial.Serial(self.port, self.baudrate, timeout=0.05)

    def _apply(self, angles: list[float]) -> None:
        assert self._serial is not None
        raw = [angle_to_raw(a, r) for a, r in zip(angles, self._joint_limits, strict=True)]
        packet = build_sync_write_goal_position(self._servo_ids, raw)
        self._serial.write(packet)

    def _on_close(self) -> None:
        if self._serial is not None:
            self._serial.close()


class MuJoCoMotorBackend(_ThreadedMotorBackend):
    """SO-ARM100 public MJCF, stepped in a background thread. Rendering runs as an
    asyncio task on the *caller's* loop instead of the physics thread: macOS
    requires Cocoa/OpenGL window + cv2.imshow calls to originate on the main
    thread, and an asyncio task scheduled from connect() naturally lands there
    when the caller's own event loop runs on the main thread (the normal case)."""

    backend_name = "mujoco"

    def __init__(
        self,
        mjcf_path: str,
        tick_hz: float = 100.0,
        render: bool = True,
        render_hz: float = 30.0,
    ) -> None:
        super().__init__(
            joint_limits=DEFAULT_JOINT_LIMITS, home_position=DEFAULT_HOME, tick_hz=tick_hz
        )
        self.mjcf_path = mjcf_path
        self._render = render
        self._render_hz = render_hz
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._render_data: mujoco.MjData | None = None
        self._renderer: mujoco.Renderer | None = None
        self._render_task: asyncio.Task[None] | None = None

    def _on_connect(self) -> None:
        self._model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self._data = mujoco.MjData(self._model)
        n = self._model.nu
        self._joint_limits = [tuple(self._model.jnt_range[i]) for i in range(n)]
        self._home_position = [0.0] * n
        if self._render:
            self._render_data = mujoco.MjData(self._model)
            self._renderer = mujoco.Renderer(self._model, height=480, width=640)
            assert self._loop is not None
            self._render_task = self._loop.create_task(self._render_loop())

    def _apply(self, angles: list[float]) -> None:
        assert self._model is not None and self._data is not None
        n = self._model.nu
        self._data.ctrl[:n] = angles[:n]
        substeps = max(1, round((1.0 / self._tick_hz) / self._model.opt.timestep))
        for _ in range(substeps):
            mujoco.mj_step(self._model, self._data)
        self._report_actual(list(self._data.qpos[:n]))

    async def _render_loop(self) -> None:
        assert self._model is not None
        assert self._renderer is not None
        assert self._render_data is not None
        cv2.namedWindow("lelamp sim", cv2.WINDOW_AUTOSIZE)
        try:
            while self._running:
                with self._lock:
                    qpos = np.array(self._actual if self._actual is not None else self._current)
                self._render_data.qpos[: len(qpos)] = qpos
                mujoco.mj_forward(self._model, self._render_data)
                self._renderer.update_scene(self._render_data)
                frame = self._renderer.render()
                cv2.imshow("lelamp sim", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
                await asyncio.sleep(1.0 / self._render_hz)
        finally:
            cv2.destroyWindow("lelamp sim")

    def _on_close(self) -> None:
        if self._render_task is not None:
            self._render_task.cancel()
        if self._renderer is not None:
            self._renderer.close()


def make_motor_backend() -> MotorBackend:
    backend = os.environ.get("LELAMP_MOTOR_BACKEND", "sim").lower()
    if backend == "serial":
        port = os.environ["LELAMP_SERIAL_PORT"]
        return SerialMotorBackend(port=port)
    if backend == "sim":
        mjcf_path = os.environ.get("LELAMP_MJCF_PATH", DEFAULT_MJCF_PATH)
        return MuJoCoMotorBackend(mjcf_path=mjcf_path)
    raise ValueError(f"unknown LELAMP_MOTOR_BACKEND: {backend!r}")

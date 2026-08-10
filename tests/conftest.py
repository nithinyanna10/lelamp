from __future__ import annotations

import numpy as np
import pytest

from lelamp.behavior.motor import NUM_JOINTS, MotorBackend, MotorState


class FakeMotorBackend(MotorBackend):
    """In-memory backend for tests; records every command instead of touching hardware."""

    def __init__(self) -> None:
        self.connected = False
        self.sent: list[list[float]] = []
        self._angles = [0.0] * NUM_JOINTS

    async def connect(self) -> None:
        self.connected = True

    async def move_to(self, joint_angles: list[float], duration_s: float) -> None:
        self.sent.append(joint_angles)
        self._angles = joint_angles

    async def get_state(self) -> MotorState:
        return MotorState(joint_angles=self._angles, timestamp=0.0, is_moving=False)

    async def home(self) -> None:
        await self.move_to([0.0] * NUM_JOINTS, duration_s=2.0)

    async def stop(self) -> None:
        pass

    async def close(self) -> None:
        self.connected = False


class FakeExpressionPlayer:
    """Stands in for ExpressionPlayer: records every dispatch instead of driving a
    real motor. All three dispatch methods return immediately (no internal
    awaits), so callers that create_task() one and yield once (asyncio.sleep(0))
    see it reflected in .calls/.current() right away -- see fsm.py's _go()."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self._current: str | None = None

    def current(self) -> str | None:
        return self._current

    async def play(self, name: str, intensity: float = 1.0) -> None:
        self.calls.append(("play", name))
        self._current = name

    async def preempt(self, name: str, intensity: float = 1.0) -> None:
        self.calls.append(("preempt", name))
        self._current = name

    async def play_chain(self, chain_name: str) -> None:
        self.calls.append(("play_chain", chain_name))
        self._current = chain_name

    async def stop(self) -> None:
        self.calls.append(("stop", ""))
        self._current = None


@pytest.fixture
def mock_expression_player() -> FakeExpressionPlayer:
    return FakeExpressionPlayer()


class FakeSerialPort:
    """Stands in for pyserial.Serial: records every write, needs no real device."""

    def __init__(self) -> None:
        self.written: list[bytes] = []
        self.is_open = True

    def write(self, data: bytes) -> int:
        self.written.append(bytes(data))
        return len(data)

    def close(self) -> None:
        self.is_open = False


class FakeAnthropicClient:
    """Minimal stand-in for the anthropic client; queue canned responses in .responses."""

    def __init__(self) -> None:
        self.responses: list[object] = []
        self.calls: list[dict[str, object]] = []

    async def stream(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.responses.pop(0) if self.responses else None


@pytest.fixture
def mock_motor_backend() -> FakeMotorBackend:
    return FakeMotorBackend()


@pytest.fixture
def mock_serial_port() -> FakeSerialPort:
    return FakeSerialPort()


@pytest.fixture
def mock_llm_client() -> FakeAnthropicClient:
    return FakeAnthropicClient()


@pytest.fixture
def mock_frame_image() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


class FakeVideoCapture:
    """Stands in for cv2.VideoCapture so camera_task can be tested without a webcam."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self._i = 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._i >= len(self._frames):
            return False, None
        frame = self._frames[self._i]
        self._i += 1
        return True, frame

    def release(self) -> None:
        pass


@pytest.fixture
def mock_video_capture(mock_frame_image: np.ndarray) -> FakeVideoCapture:
    return FakeVideoCapture([mock_frame_image] * 5)

"""Step 3 visual gate: the expression library on the sim lamp.

Cycles all 20 expressions once (printing each as it plays, with actual vs.
spec'd timing), then drops into an interactive mode: press a mapped key to
trigger that expression live, 'l' to toggle a Lissajous-driven look_at_face
demo (watch it compose with whatever expression is currently playing), space
for home, Ctrl-C/Esc to quit. Opens the MuJoCo window (via MuJoCoMotorBackend,
untouched) -- watch it alongside this terminal.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import sys
import termios
import time
import tty

from lelamp.behavior.expression_player import ExpressionPlayer
from lelamp.behavior.expressions import EXPRESSIONS
from lelamp.behavior.idle_overlay import IdleOverlay
from lelamp.behavior.motor import MuJoCoMotorBackend

MJCF_PATH = "assets/so_arm100/scene.xml"
CYCLE_PAUSE_S = 0.5
LISSAJOUS_HZ = 30.0

# 1-9, then qwertyuiop, then as: 21 slots for 20 expressions (one spare).
_KEYS = list("123456789qwertyuiopas")
KEY_TO_EXPRESSION = dict(zip(_KEYS, EXPRESSIONS, strict=False))


def _read_key() -> str:
    """Blocking single-keypress read from the terminal, no Enter required."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


async def cycle_all(player: ExpressionPlayer) -> None:
    print("\n=== cycling all 20 expressions ===")
    for name, expr in EXPRESSIONS.items():
        print(f"  -> {name} ({expr.mood_family})")
        t0 = time.monotonic()
        await player.play(name)
        elapsed_ms = (time.monotonic() - t0) * 1000
        print(f"     done in {elapsed_ms:.0f}ms (spec {expr.duration_ms}ms)")
        await asyncio.sleep(CYCLE_PAUSE_S)
    print("=== cycle complete ===\n")


async def _lissajous_look_at(player: ExpressionPlayer) -> None:
    t0 = time.monotonic()
    try:
        while True:
            t = time.monotonic() - t0
            x = math.sin(2.0 * t)
            y = math.sin(3.0 * t + math.pi / 4)
            player.look_at_face((x, y))
            await asyncio.sleep(1.0 / LISSAJOUS_HZ)
    finally:
        player.look_at_face(None)


async def interactive_mode(player: ExpressionPlayer) -> None:
    print("=== interactive mode ===")
    for key, name in KEY_TO_EXPRESSION.items():
        print(f"  [{key}] {name}")
    print("  [l] toggle Lissajous look_at_face demo (composition with the current expression)")
    print("  [space] home")
    print("  [Ctrl-C or Esc] quit\n")

    lissajous_task: asyncio.Task[None] | None = None
    loop = asyncio.get_running_loop()

    try:
        while True:
            key = await loop.run_in_executor(None, _read_key)
            if key in ("\x03", "\x1b"):
                break
            if key == "l":
                if lissajous_task is None:
                    lissajous_task = asyncio.create_task(_lissajous_look_at(player))
                    print("  lissajous look_at_face: ON")
                else:
                    lissajous_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await lissajous_task
                    lissajous_task = None
                    print("  lissajous look_at_face: OFF")
                continue
            if key == " ":
                await player.preempt("home")
                print("  -> home")
                continue
            expr_name = KEY_TO_EXPRESSION.get(key)
            if expr_name is None:
                continue
            t0 = time.monotonic()
            await player.preempt(expr_name)
            print(f"  -> {expr_name} ({(time.monotonic() - t0) * 1000:.0f}ms)")
    finally:
        if lissajous_task is not None:
            lissajous_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await lissajous_task


async def main() -> None:
    motor = MuJoCoMotorBackend(MJCF_PATH)
    await motor.connect()
    player = ExpressionPlayer(motor, IdleOverlay())

    try:
        await cycle_all(player)
        if sys.stdin.isatty():
            await interactive_mode(player)
        else:
            print("stdin is not a terminal -- skipping interactive mode")
    finally:
        await motor.close()


if __name__ == "__main__":
    asyncio.run(main())

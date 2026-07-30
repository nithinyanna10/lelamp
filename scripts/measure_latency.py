"""Eval-harness kickoff: replays a recorded video through the real perception
pipeline (no camera, no live gaze) and reports p50/p95 latency per stage against
the step-2 budget:

  camera-read -> GazeEvent emitted        < 40ms p95   (MediaPipe inference)
  GazeEvent -> hysteresis transition      < 1ms        (in-process function call)
  transition -> motor command dispatched  < 250ms p95  (move_to() reaching its
                                                         first await point)

If no recorded clip is given, synthesizes one under eval_data/clips/ from the
committed face fixtures: blocks of "looking at camera" (portrait.jpg, letterboxed
to preserve aspect ratio -- naive resize distorts the face enough to change the
measured score) long enough to cross the 400ms engage dwell, alternating with
faceless frames (nobody in view, gaze_score=0.0 -- a reliable disengaged case;
business_person.png's head turn alone lands the score in the 0.3-0.7 dead zone
since its iris still reads as roughly centered) long enough to cross the 1500ms
disengage dwell -- so the replay actually produces real engagement transitions to
measure dispatch latency against, not just per-frame inference numbers.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import cv2
import numpy as np

from lelamp.behavior.expressions import HOME_POSE, WAKE_DURATION_S, WAKE_POSE
from lelamp.behavior.motor import MuJoCoMotorBackend
from lelamp.perception.camera import Frame
from lelamp.perception.face_gaze import DEFAULT_MODEL_PATH, _Detector, _process, ensure_model
from lelamp.perception.hysteresis import HysteresisGate
from lelamp.state.fsm import LampFSM, LampState

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures" / "faces"
DEFAULT_CLIP = Path("eval_data/clips/perception_latency_sample.avi")
FPS = 30
LOOKING_BLOCK_S = 1.0  # > 400ms engage dwell
AWAY_BLOCK_S = 2.0  # > 1500ms disengage dwell
CYCLES = 4
MJCF_PATH = "assets/so_arm100/scene.xml"


def _letterbox(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0, y0 = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _synthesize_clip(path: Path) -> None:
    size = (640, 480)
    portrait = cv2.imread(str(FIXTURES / "portrait.jpg"))
    assert portrait is not None, "missing tests/fixtures/faces/portrait.jpg"
    looking = _letterbox(portrait, size)
    away = np.zeros((size[1], size[0], 3), dtype=np.uint8)  # faceless -- nobody in view

    path.parent.mkdir(parents=True, exist_ok=True)
    # Lossless (FFV1): a lossy codec's per-frame compression noise is enough to
    # jitter a score sitting near a threshold across it frame to frame, which
    # would make hysteresis transitions flaky for no reason relevant to what
    # this harness measures.
    fourcc = cv2.VideoWriter.fourcc(*"FFV1")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, size)
    for _ in range(CYCLES):
        for _ in range(int(LOOKING_BLOCK_S * FPS)):
            writer.write(looking)
        for _ in range(int(AWAY_BLOCK_S * FPS)):
            writer.write(away)
    writer.release()
    print(f"synthesized {path} ({CYCLES} look/away cycles at {FPS}fps)")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(p / 100.0 * (len(ordered) - 1))))
    return ordered[idx]


def _report(name: str, values_ms: list[float]) -> None:
    if not values_ms:
        print(f"{name}: no samples")
        return
    print(
        f"{name}: n={len(values_ms)} p50={_percentile(values_ms, 50):.3f}ms "
        f"p95={_percentile(values_ms, 95):.3f}ms max={max(values_ms):.3f}ms"
    )


async def _measure(clip_path: Path) -> None:
    model_path = ensure_model(DEFAULT_MODEL_PATH)
    detector = await asyncio.to_thread(_Detector, model_path, num_faces=2)
    gate = HysteresisGate()
    fsm = LampFSM()
    motor = MuJoCoMotorBackend(MJCF_PATH, render=False)
    await motor.connect()
    await motor.move_to(HOME_POSE, duration_s=0.1)

    cap = cv2.VideoCapture(str(clip_path))
    inference_ms: list[float] = []
    hysteresis_ms: list[float] = []
    dispatch_ms: list[float] = []
    frame_id = 0
    transitions = 0

    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            frame_id += 1
            # Video-paced, not wall-clock: inference alone runs far faster than 30fps
            # (~1-5ms/frame vs. 33ms/frame), so replaying back-to-back would blow
            # through the hysteresis dwell windows in real time before they ever
            # accumulate 400ms/1500ms of "elapsed" video. The gate only cares about
            # the timestamps it's given (that's why update() takes one explicitly),
            # so feed it what a real 30fps camera would have produced. Inference and
            # dispatch latency are still measured with genuine wall-clock time below
            # -- those two are separate timing domains.
            video_timestamp = frame_id / FPS
            frame = Frame(frame_id=frame_id, timestamp=video_timestamp, image=image)

            t0 = time.monotonic()
            result = detector.detect(frame.image)
            gaze_event, _ = _process(frame, result, inference_ms=0.0)
            inference_ms.append((time.monotonic() - t0) * 1000.0)

            t0 = time.monotonic()
            transition = gate.update(gaze_event.gaze_score, timestamp=video_timestamp)
            hysteresis_ms.append((time.monotonic() - t0) * 1000.0)

            if transition is None:
                continue
            fsm_transition = fsm.on_engagement_transition(transition)
            if fsm_transition is None:
                continue
            transitions += 1

            pose = WAKE_POSE if fsm_transition.to_state == LampState.ENGAGED else HOME_POSE
            t0 = time.monotonic()
            task = asyncio.ensure_future(motor.move_to(pose, duration_s=WAKE_DURATION_S))
            await asyncio.sleep(0)  # let move_to() run to its first await point (dispatch done)
            dispatch_ms.append((time.monotonic() - t0) * 1000.0)
            await task
    finally:
        cap.release()
        await motor.close()

    print(f"\nprocessed {frame_id} frames, {transitions} FSM transitions\n")
    _report("camera-read -> GazeEvent (inference)", inference_ms)
    _report("GazeEvent -> hysteresis transition", hysteresis_ms)
    _report("transition -> motor command dispatched", dispatch_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    args = parser.parse_args()

    if not args.clip.exists():
        _synthesize_clip(args.clip)

    asyncio.run(_measure(args.clip))


if __name__ == "__main__":
    main()

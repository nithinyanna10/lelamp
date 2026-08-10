# LeLamp

6-DOF expressive lamp: gaze-based engagement, hand-authored motion primitives,
hybrid spatial/semantic memory, and Claude-driven conversation with tool use.
Single async Python process — no distributed infra.

## Architecture

```
main.py                    orchestrator: creates queues, starts tasks, wires the brain loop
├── perception/
│   ├── camera.py          cv2.VideoCapture -> Frame @ 30fps
│   ├── face_gaze.py       MediaPipe FaceMesh iris landmarks -> GazeEvent (raw gaze score)
│   ├── scene_scan.py      YOLO-World on MPS, on-demand -> Detections
│   └── audio.py           sounddevice + Silero VAD -> SpeechEvent
├── state/
│   └── fsm.py             IDLE/ENGAGED/LISTENING/SPEAKING/SCANNING/SEEKING/SLEEPING,
│                          hysteresis lives here (face_gaze only reports a raw score)
├── behavior/
│   ├── motor.py           MotorBackend ABC (move_to/get_state/home/stop): SerialMotorBackend
│                          (pyserial + Feetech protocol) or MuJoCoMotorBackend (real SO-ARM100/
│                          SO101 MJCF, vendored under assets/) — same call sites either way.
│                          Ease-in-out trajectory generation lives here, not in the caller.
│   ├── feetech.py          Feetech STS3215 sync-write packet framing + angle<->raw conversion
│   ├── ik.py               ikpy chain built by walking the MJCF's compiled kinematic tree
│   └── primitives.py       wake, sleep, nod, curious_tilt, look_at, point_at, breathe
├── memory/
│   ├── db.py                sqlite + sqlite-vec: objects / sightings / scenes, dedupe on write
│   ├── embeddings.py         open_clip on MPS
│   ├── writer.py             write_scan: dedupe-or-insert a scene's detections, save crops
│   └── query.py              LLM tool implementations: query_memory, describe_current_scene,
│                             point_at, remember (+ query_by_class/query_recent/
│                             describe_current_memory for the demo/eval scripts)
├── conversation/
│   ├── stt.py                whisper.cpp/CoreML, faster-whisper fallback
│   ├── llm.py                Claude streaming with tool use
│   └── tts.py                Cartesia streaming, macOS `say` fallback
├── telemetry.py              OpenTelemetry init, frame_id -> audio-output trace propagation
└── eval/
    └── harness.py             `make eval`: engagement P/R/F1, memory Q&A accuracy
```

Modules never call each other directly — everything crosses an `asyncio.Queue`,
and every payload is a Pydantic model. `state/fsm.py` owns the hysteresis
thresholds (engage: score>0.7 for 400ms; disengage: score<0.3 for 1500ms) so
`face_gaze.py` stays a stateless per-frame scorer.

### Why `memory/` sits next to `behavior/`, not under it

The tools list (`query_memory`, `describe_current_scene`, `point_at`, `remember`)
and the `objects`/`sightings`/`scenes` schema are both memory concerns, so
`db.py`, `embeddings.py`, and `query.py` are grouped under a top-level `memory/`
package rather than nested in `behavior/`.

## Assets

`assets/so_arm100/` vendors the real SO-ARM100/SO101 model (Apache-2.0) from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) —
`scene.xml` (arm + ground + lighting), `so101_new_calib.xml`, and the STL
meshes it references. `ik.py` and both motor backends load this directly, so
serial and sim share one kinematic model with no hand-authored geometry.

## Setup

```bash
uv sync
```

## Running

```bash
make demo                          # run the live orchestrator (needs camera + mic; falls back
                                    # to MuJoCo sim if $LELAMP_SERIAL_PORT is unset)
uv run python scripts/demo_motion.py  # motor-layer smoke test: home, sweep each joint, look_at
make eval                          # run the eval harness against eval_data/clips
make test                          # pytest, fully mocked — no camera/motor/LLM required
make lint                          # ruff check
make typecheck                     # mypy --strict
```

Environment variables:

- `LELAMP_MOTOR_BACKEND` — `sim` (default) or `serial`.
- `LELAMP_SERIAL_PORT` — required when `LELAMP_MOTOR_BACKEND=serial` (e.g. `/dev/tty.usbmodem*`).
- `LELAMP_MJCF_PATH` — path to the MJCF for the sim backend (default `assets/so_arm100/scene.xml`).
- `ANTHROPIC_API_KEY` — for `conversation/llm.py`.

## Status

1. **Done** — motor backend abstraction + MuJoCo sim: `MotorBackend` ABC
   (`move_to`/`get_state`/`home`/`stop`), shared ease-in-out trajectory generation and
   background-thread control loop, `SerialMotorBackend` (Feetech sync-write protocol),
   `MuJoCoMotorBackend` (real MJCF, physics thread + asyncio-scheduled cv2 render loop),
   `ik.py` cross-validated against MuJoCo's own forward kinematics to 1e-16 m, OTel span
   per `move_to`. `scripts/demo_motion.py` is the smoke test.
2. **Done** — camera + MediaPipe gaze + hysteresis, wired end-to-end in `main.py`.
3. **Done** — FSM finalized + attention-seeking escalation: `state/fsm.py`'s
   `LampFSM.handle_event()` is fully event-driven (`GazeEngaged`/`GazeDisengaged`/
   `FacePresent`/`FaceLost`/`ScanComplete`/`Tick`) over 8 states (`SLEEPING`,
   `IDLE`, `ENGAGED`, `SCANNING`, `DISENGAGING`, `SEEKING_1/2/3`; `LISTENING`/
   `SPEAKING` are typed for steps 6/7 but unreachable). Every transition plays an
   expression via `ExpressionPlayer.preempt()`/`play_chain()` and logs
   `from_state`/`to_state`/`reason`/`dispatch_latency_ms` (OTel span + structlog).
   Time-based escalation (`state/config.py`'s `FSMTimings`) is driven by `Tick`
   events from a 500ms timer task, not polling. `debug_overlay.py`'s HUD gained
   state history, current-expression progress bar, seeking countdown, and
   scan/memory panels. `scripts/demo_full_loop.py` is the smoke test (thin
   wrapper over `main.py`, same as `demo_perception.py`).
4. **Done** — memory schema + CLIP embeddings + dedupe: `db.py` (sqlite + sqlite-vec,
   2D normalized position -- 3D/point_at comes in step 7), `embeddings.py` (open_clip
   ViT-B-32, batched), `writer.py` (dedupe: same class + cosine>0.85 + position within
   0.3 + seen within 30min -> update, else insert + save crop), `query.py` (query_memory
   hybrid semantic search, query_by_class/query_recent structured, describe_current_memory,
   describe_current_scene, point_at, remember). `scene_scan.py` runs YOLO-World on demand,
   triggered whenever the FSM first enters ENGAGED (single scan from home pose;
   multi-viewpoint sweep synced to a pan sweep is a TODO). `scripts/demo_memory.py`
   is the smoke test.
5. Claude tools + streaming conversation
6. Eval harness against recorded sessions

## Testing without hardware

`tests/conftest.py` provides `mock_motor_backend` (records commands instead of
touching serial/MuJoCo), `mock_serial_port` (fake pyserial port, captures written
bytes), `mock_llm_client` (canned Anthropic responses), and `mock_video_capture`
(feeds fixed frames instead of `cv2.VideoCapture`). `MuJoCoMotorBackend(..., render=False)`
runs the real physics headless, no display required — that's what CI exercises.

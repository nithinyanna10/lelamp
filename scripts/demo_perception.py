"""Step 2 end-to-end demo: webcam -> MediaPipe gaze -> hysteresis -> FSM -> sim lamp.

Opens two windows: "perception debug" (landmarks + gaze score + engagement state)
and the MuJoCo viewport (the lamp waking/sleeping in response to your gaze). State
transitions and their dispatch latency print to stdout as they happen. Ctrl-C to
stop -- shutdown is graceful (tasks cancelled, windows closed, motor backend and
OTel flushed).
"""

from lelamp.main import main

if __name__ == "__main__":
    main()

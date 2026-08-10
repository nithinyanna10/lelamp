"""Step 5 end-to-end demo: the full autonomous loop, driven by real perception.

SLEEPING -> (face detected) -> IDLE -> (gaze engaged) -> ENGAGED -> SCANNING
(scans your desk) -> ENGAGED -> (look away) -> DISENGAGING -> SEEKING_1 -> 2 -> 3
-> SLEEPING, or back to ENGAGED at any point you look back. Opens the webcam +
debug HUD window (state history, current-expression progress bar, attention-
seeking countdown, scan/memory status) and the MuJoCo viewport. Every FSM
transition prints to stdout (structlog, via fsm.py's _record_transition) with
its timestamp, from/to state, reason, and dispatch latency. Ctrl-C to stop.
"""

from lelamp.main import main

if __name__ == "__main__":
    main()

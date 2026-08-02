"""One-off generator for the 5 expression audio cues -- procedural sine-wave
synthesis via stdlib `wave` + numpy, no scipy needed. Run once, commit the
output under assets/audio/.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).parent.parent / "assets" / "audio"
SAMPLE_RATE = 44100


def _bell_envelope(n: int, attack_n: int, decay_tau_n: float) -> np.ndarray:
    t = np.arange(n)
    attack = np.clip(t / max(1, attack_n), 0.0, 1.0)
    decay = np.exp(-np.maximum(0, t - attack_n) / decay_tau_n)
    return attack * decay


def _tone(
    freq: float,
    duration_s: float,
    amp: float = 0.3,
    harmonics: tuple[float, ...] = (1.0,),
) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n) / SAMPLE_RATE
    wave_sum = np.zeros(n)
    for i, weight in enumerate(harmonics, start=1):
        wave_sum += weight * np.sin(2 * math.pi * freq * i * t)
    wave_sum /= sum(harmonics)
    envelope = _bell_envelope(
        n, attack_n=int(SAMPLE_RATE * 0.005), decay_tau_n=SAMPLE_RATE * duration_s * 0.3
    )
    return np.asarray(amp * wave_sum * envelope)


def _chirp(f0: float, f1: float, duration_s: float, amp: float = 0.3) -> np.ndarray:
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n) / SAMPLE_RATE
    freq_t = f0 + (f1 - f0) * (t / duration_s)
    phase = 2 * math.pi * np.cumsum(freq_t) / SAMPLE_RATE
    signal = np.sin(phase)
    attack_n = int(SAMPLE_RATE * 0.01)
    release_n = int(SAMPLE_RATE * 0.03)
    envelope = np.ones(n)
    envelope[:attack_n] = np.linspace(0, 1, attack_n)
    envelope[-release_n:] = np.linspace(1, 0, release_n)
    return amp * signal * envelope


def _concat(*parts: np.ndarray, gap_s: float = 0.02) -> np.ndarray:
    gap = np.zeros(int(SAMPLE_RATE * gap_s))
    pieces = []
    for i, part in enumerate(parts):
        pieces.append(part)
        if i < len(parts) - 1:
            pieces.append(gap)
    return np.concatenate(pieces)


def write_wav(path: Path, samples: np.ndarray) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(pcm.tobytes())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Two ascending warm-bell notes: C5 -> E5.
    chime_soft = _concat(
        _tone(523.25, 0.14, amp=0.35, harmonics=(1.0, 0.5, 0.25)),
        _tone(659.25, 0.16, amp=0.35, harmonics=(1.0, 0.5, 0.25)),
    )
    write_wav(OUT_DIR / "chime_soft.wav", chime_soft)

    # Single warm chord: A4 + its fifth + octave.
    chime_warm = _tone(440.0, 0.35, amp=0.32, harmonics=(1.0, 0.7, 0.4))
    write_wav(OUT_DIR / "chime_warm.wav", chime_warm)

    chirp_cheerful = _chirp(400, 800, 0.18, amp=0.35)
    write_wav(OUT_DIR / "chirp_cheerful.wav", chirp_cheerful)

    chirp_quiet = _chirp(350, 550, 0.12, amp=0.18)
    write_wav(OUT_DIR / "chirp_quiet.wav", chirp_quiet)

    chirp_loud = _chirp(300, 900, 0.22, amp=0.5)
    write_wav(OUT_DIR / "chirp_loud.wav", chirp_loud)

    for name in ["chime_soft", "chime_warm", "chirp_cheerful", "chirp_quiet", "chirp_loud"]:
        path = OUT_DIR / f"{name}.wav"
        print(f"wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

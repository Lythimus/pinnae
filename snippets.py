from __future__ import annotations

import os
import threading
from collections import deque
from datetime import datetime
from typing import Optional

import numpy as np

import config


class AudioRingBuffer:
    """Thread-safe ring buffer of raw audio chunks for post-hoc snippet extraction."""

    def __init__(self) -> None:
        max_chunks = int(np.ceil(
            config.RING_BUFFER_SECONDS * config.SAMPLE_RATE / config.CHUNK_SAMPLES
        ))
        self._buf: deque[tuple[float, np.ndarray]] = deque(maxlen=max_chunks)
        self._lock = threading.Lock()

    def push(self, chunk_time: float, samples: np.ndarray) -> None:
        """Append a copy of samples with its start timestamp. Call from audio callback."""
        with self._lock:
            self._buf.append((chunk_time, samples.copy()))

    def extract(self, start_ts: float, end_ts: float) -> np.ndarray:
        """Return concatenated samples whose chunk window overlaps [start_ts, end_ts]."""
        chunk_dur = config.CHUNK_SAMPLES / config.SAMPLE_RATE
        with self._lock:
            pieces = [
                s for (t, s) in self._buf
                if t + chunk_dur >= start_ts and t <= end_ts
            ]
        return np.concatenate(pieces) if pieces else np.array([], dtype=np.float32)


def save_snippet(
    ring: AudioRingBuffer,
    event_start: float,
    event_end: float,
    band: str,
    flagged: bool = False,
) -> None:
    """
    Save a WAV file and spectrogram PNG for a detection event.
    Runs on the consumer thread — imports are lazy to avoid loading matplotlib
    into the audio callback path.
    """
    import soundfile as sf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pre_s  = config.SNIPPET_PRE_MS  / 1000.0
    post_s = config.SNIPPET_POST_MS / 1000.0
    audio  = ring.extract(event_start - pre_s, event_end + post_s)

    if audio.size == 0:
        return

    os.makedirs(config.DETECTION_SNIPPET_DIR, exist_ok=True)

    ts_str  = datetime.fromtimestamp(event_start).strftime("%Y%m%dT%H%M%S_%f")[:-3]
    flag_str = "_FLAGGED" if flagged else ""
    stem    = os.path.join(config.DETECTION_SNIPPET_DIR, f"{ts_str}_{band}{flag_str}")

    sf.write(f"{stem}.wav", audio, config.SAMPLE_RATE, subtype="FLOAT")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.specgram(
        audio,
        Fs=config.SAMPLE_RATE,
        NFFT=1024,
        noverlap=512,
        cmap="inferno",
        vmin=-120,
        vmax=-20,
    )
    # Mark the alarm band region
    ax.axhline(config.BANDS[band][0], color="cyan", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.axhline(config.BANDS[band][1], color="cyan", linewidth=0.7, linestyle="--", alpha=0.6)
    # Mark detection window
    ax.axvline(pre_s, color="lime", linewidth=0.8, linestyle=":")
    ax.axvline(pre_s + (event_end - event_start), color="red", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    title = f"{band}  {ts_str}"
    if flagged:
        title += "  [FLAGGED: duration exceeded]"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=100)
    plt.close(fig)

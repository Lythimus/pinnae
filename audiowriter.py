from __future__ import annotations

import collections
import os
import threading
from datetime import datetime
from typing import Optional

import numpy as np
import soundfile as sf


class AudioRingBuffer:
    """Thread-safe ring buffer of raw audio chunks tagged with a monotonic chunk index."""

    def __init__(self, maxlen_chunks: int) -> None:
        self._buf: collections.deque[tuple[int, np.ndarray]] = collections.deque(maxlen=maxlen_chunks)
        self._lock = threading.Lock()

    def append(self, idx: int, samples: np.ndarray) -> None:
        with self._lock:
            self._buf.append((idx, samples))

    def extract(self, start_idx: int, end_idx: int) -> Optional[np.ndarray]:
        """Return concatenated samples for chunks in [start_idx, end_idx], or None if evicted."""
        with self._lock:
            buf_list = list(self._buf)
        if not buf_list:
            return None
        if buf_list[0][0] > start_idx:
            return None  # start chunk evicted — buffer too small for this call
        chunks = [s for i, s in buf_list if start_idx <= i <= end_idx]
        if not chunks:
            return None
        return np.concatenate(chunks)


class AudioClipWriter:
    """Writes detected call clips as FLAC files under base_dir/{band}/{datetime}.{ext}."""

    def __init__(self, base_dir: str, sample_rate: int, fmt: str) -> None:
        self._base_dir = base_dir
        self._sample_rate = sample_rate
        self._fmt = fmt.lower()

    def write_clip(self, band: str, timestamp_abs: float, samples: np.ndarray) -> str:
        dt = datetime.fromtimestamp(timestamp_abs)
        ms = dt.microsecond // 1000
        name = dt.strftime("%Y-%m-%dT%H-%M-%S") + f".{ms:03d}.{self._fmt}"
        band_dir = os.path.join(self._base_dir, band)
        os.makedirs(band_dir, exist_ok=True)
        path = os.path.join(band_dir, name)
        sf.write(path, samples, self._sample_rate, format=self._fmt.upper())
        return path

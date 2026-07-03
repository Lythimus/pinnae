from __future__ import annotations

import collections
import os
import sys
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

import config


class VideoRingBuffer:
    """Thread-safe ring buffer of JPEG-encoded camera frames tagged with Unix timestamps.

    Stores frames as JPEG bytes rather than raw arrays to keep memory bounded
    (~135 MB for 30 s at 720p/30 fps vs ~2.4 GB raw).
    """

    def __init__(self, maxlen_frames: int, jpeg_quality: int = config.VIDEO_JPEG_QUALITY) -> None:
        self._buf: collections.deque[tuple[float, bytes]] = collections.deque(maxlen=maxlen_frames)
        self._lock = threading.Lock()
        self._jpeg_quality = jpeg_quality

    def append(self, timestamp_abs: float, frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        if ok:
            with self._lock:
                self._buf.append((timestamp_abs, encoded.tobytes()))

    def extract(self, t_start: float, t_end: float) -> Optional[list[np.ndarray]]:
        """Return decoded frames whose timestamp ∈ [t_start, t_end], or None if evicted."""
        with self._lock:
            buf_list = list(self._buf)
        if not buf_list:
            return None
        if buf_list[0][0] > t_start:
            return None  # start of window evicted — buffer too small
        frames = []
        for ts, jpeg_bytes in buf_list:
            if t_start <= ts <= t_end:
                arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    frames.append(frame)
        return frames if frames else None


class VideoClipWriter:
    """Writes video clips as mp4 files under base_dir/{band}/{datetime}.mp4."""

    def __init__(self, base_dir: str, fps: float, width: int, height: int) -> None:
        self._base_dir = base_dir
        self._fps = fps
        self._width = width
        self._height = height

    def write_clip(self, band: str, timestamp_abs: float, frames: list[np.ndarray]) -> str:
        dt = datetime.fromtimestamp(timestamp_abs)
        ms = dt.microsecond // 1000
        name = dt.strftime("%Y-%m-%dT%H-%M-%S") + f".{ms:03d}.mp4"
        band_dir = os.path.join(self._base_dir, band)
        os.makedirs(band_dir, exist_ok=True)
        path = os.path.join(band_dir, name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self._fps, (self._width, self._height))
        try:
            for frame in frames:
                if frame.shape[1] != self._width or frame.shape[0] != self._height:
                    frame = cv2.resize(frame, (self._width, self._height))
                writer.write(frame)
        finally:
            writer.release()
        return path


class VideoCaptureThread:
    """Daemon thread that reads frames from a camera and feeds the ring buffer.

    If the camera fails to open, sets `self.enabled = False` rather than raising,
    so the audio pipeline is unaffected.
    """

    def __init__(
        self,
        camera: int | str,
        ring_buffer: VideoRingBuffer,
        continuous_writer: Optional[cv2.VideoWriter] = None,
    ) -> None:
        self._camera = camera
        self._ring = ring_buffer
        self._continuous = continuous_writer
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.enabled = False

    def start(self) -> bool:
        """Open the camera and start capturing. Returns True if the camera opened."""
        cap = cv2.VideoCapture(self._camera)
        if not cap.isOpened():
            print(
                f"[!] video: could not open camera {self._camera!r} — "
                "video recording disabled; audio pipeline unaffected.",
                file=sys.stderr,
            )
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.VIDEO_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.VIDEO_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.VIDEO_FPS)
        self._cap = cap
        self.enabled = True
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self.enabled:
            self._cap.release()
        if self._continuous is not None:
            self._continuous.release()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                break
            ts = time.time()
            self._ring.append(ts, frame)
            if self._continuous is not None:
                self._continuous.write(frame)

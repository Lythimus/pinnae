from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

import config


@dataclass
class DetectionEvent:
    band: str
    peak_freq_hz: float
    power_db: float
    duration_ms: float
    timestamp_abs: float       # Unix epoch seconds (start of call)
    flagged_artifact: bool = field(default=False)


def compute_power_spectrum(chunk: np.ndarray) -> np.ndarray:
    """Return one-sided power spectrum in dBFS for a mono float32 chunk."""
    window = np.hanning(len(chunk))
    spectrum = np.fft.rfft(chunk * window)
    power = np.abs(spectrum) ** 2
    return 10.0 * np.log10(np.maximum(power, 1e-12))


class BandDetector:
    """
    Detects USV call events in a single frequency band via FFT thresholding.

    Noise floor is a per-bin rolling median over NOISE_HISTORY_CHUNKS (~30 s).
    A persistent narrowband tone is absorbed into its own bin's floor and stops
    firing — unlike a scalar band-median which remains blind to it forever.

    A call opens when the peak *excess* (band_power - per-bin floor) exceeds
    DETECTION_THRESHOLD_DB and closes after CALL_END_SILENCE_MS of sub-threshold
    activity. Calls that exceed MAX_CALL_DURATION_MS are closed and tagged as
    flagged_artifact=True.
    """

    def __init__(self, band_name: str, freq_low: int, freq_high: int) -> None:
        self.band_name = band_name
        chunk_duration_ms = 1000.0 * config.CHUNK_SAMPLES / config.SAMPLE_RATE
        self._chunk_duration_s = config.CHUNK_SAMPLES / config.SAMPLE_RATE

        freqs = np.fft.rfftfreq(config.CHUNK_SAMPLES, d=1.0 / config.SAMPLE_RATE)
        self._bin_mask = (freqs >= freq_low) & (freqs <= freq_high)
        self._band_freqs = freqs[self._bin_mask]
        n_bins = int(self._bin_mask.sum())

        # Per-bin noise floor: deque of 1-D arrays, one per chunk.
        self._noise_history: deque[np.ndarray] = deque(maxlen=config.NOISE_HISTORY_CHUNKS)
        self._floor_cache: np.ndarray = np.full(n_bins, -120.0)
        self._floor_dirty: bool = True  # recompute median when history changes

        # Call state
        self._in_call = False
        self._call_start: float = 0.0
        self._call_peak_freq: float = 0.0
        self._call_max_power: float = -np.inf
        self._active_chunks: int = 0
        self._silence_chunks: int = 0

        self._min_active = max(1, int(np.ceil(config.MIN_CALL_DURATION_MS / chunk_duration_ms)))
        self._end_silence = max(1, int(np.ceil(config.CALL_END_SILENCE_MS / chunk_duration_ms)))
        self._max_active = max(1, int(np.ceil(config.MAX_CALL_DURATION_MS / chunk_duration_ms)))

    def _noise_floor(self) -> np.ndarray:
        if self._floor_dirty and len(self._noise_history) >= 5:
            self._floor_cache = np.median(np.stack(self._noise_history), axis=0)
            self._floor_dirty = False
        return self._floor_cache

    def process(self, power_spectrum: np.ndarray, chunk_time: float) -> list[DetectionEvent]:
        """
        Process one chunk's power spectrum; return any calls that just completed.

        power_spectrum must be the full rfft power array (length CHUNK_SAMPLES//2 + 1).
        chunk_time is the Unix epoch time of the start of this chunk.
        """
        band_power = power_spectrum[self._bin_mask]

        self._noise_history.append(band_power.copy())
        self._floor_dirty = True

        if len(self._noise_history) < 5:
            return []  # not enough history to estimate floor

        noise_floor = self._noise_floor()
        excess = band_power - noise_floor
        peak_idx = int(np.argmax(excess))
        peak_excess = float(excess[peak_idx])
        peak_power = float(band_power[peak_idx])
        peak_freq = float(self._band_freqs[peak_idx])

        is_active = peak_excess >= config.DETECTION_THRESHOLD_DB

        events: list[DetectionEvent] = []

        if is_active:
            self._silence_chunks = 0
            if not self._in_call:
                self._in_call = True
                self._call_start = chunk_time
                self._call_peak_freq = peak_freq
                self._call_max_power = peak_power
                self._active_chunks = 1
            else:
                self._active_chunks += 1
                if peak_power > self._call_max_power:
                    self._call_max_power = peak_power
                    self._call_peak_freq = peak_freq

                # Max-duration guard: close and flag if exceeding literature ceiling.
                if self._active_chunks >= self._max_active:
                    events.append(self._close_call(chunk_time, flagged=True))
        elif self._in_call:
            self._silence_chunks += 1
            if self._silence_chunks >= self._end_silence:
                if self._active_chunks >= self._min_active:
                    events.append(self._close_call(chunk_time))
                else:
                    self._reset()

        return events

    def flush(self, current_time: float) -> list[DetectionEvent]:
        """Finalize any in-progress call. Call at session end before closing the stream."""
        if not self._in_call:
            return []
        events = []
        if self._active_chunks >= self._min_active:
            events.append(self._close_call(current_time))
        else:
            self._reset()
        return events

    def _close_call(self, current_time: float, flagged: bool = False) -> DetectionEvent:
        duration_ms = (current_time - self._call_start) * 1000.0
        event = DetectionEvent(
            band=self.band_name,
            peak_freq_hz=self._call_peak_freq,
            power_db=self._call_max_power,
            duration_ms=duration_ms,
            timestamp_abs=self._call_start,
            flagged_artifact=flagged,
        )
        self._reset()
        return event

    def _reset(self) -> None:
        self._in_call = False
        self._call_start = 0.0
        self._call_peak_freq = 0.0
        self._call_max_power = -np.inf
        self._active_chunks = 0
        self._silence_chunks = 0

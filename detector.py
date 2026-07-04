from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import config


@dataclass
class DetectionEvent:
    band: str
    peak_freq_hz: float
    power_db: float
    duration_ms: float
    timestamp_abs: float   # Unix epoch seconds (start of call)
    start_chunk_idx: int = 0
    end_chunk_idx: int = 0
    flagged_artifact: bool = field(default=False)


def compute_power_spectrum(chunk: np.ndarray) -> np.ndarray:
    """Return one-sided power spectrum in dBFS for a mono float32 chunk."""
    window = np.hanning(len(chunk))
    spectrum = np.fft.rfft(chunk * window)
    power = np.abs(spectrum) ** 2
    return 10.0 * np.log10(np.maximum(power, 1e-12))


def _max_contiguous_run(mask: np.ndarray) -> int:
    """Length of the longest run of consecutive True values in a 1-D bool array."""
    max_run = 0
    run = 0
    for v in mask:
        if v:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return max_run


def band_shape_features(band_power_db: np.ndarray) -> dict:
    """
    Cheap per-chunk spectral-shape features distinguishing a narrowband FM call
    from broadband noise (e.g. music/playback-chain distortion hash).

    band_power_db: raw band power in dB for the current chunk.

    Occupied bins are defined *relative to this chunk's own peak*
    (a standard "-X dB bandwidth" measure), not relative to the noise floor.
    A floor-relative shoulder was tried first and failed on real alarm-band
    calls: a chunk's general energy can sit slightly above the floor across
    the *whole* band (harmonics, resampling artifacts, general elevation
    during a call) even though the call itself is narrowband, which made
    max_run_fraction saturate near 1.0 for genuine calls. Peak-relative
    occupancy is scale-invariant to that floor-wide elevation.
    """
    n_bins = len(band_power_db)
    if n_bins == 0:
        return {"occupied_fraction": 0.0, "max_run_fraction": 0.0, "peak_concentration_db": 0.0}

    peak_power_db = float(np.max(band_power_db))
    occupied = band_power_db >= (peak_power_db - config.BANDWIDTH_DROPOFF_DB)
    occupied_fraction = float(np.count_nonzero(occupied)) / n_bins
    max_run_fraction = _max_contiguous_run(occupied) / n_bins
    peak_concentration_db = peak_power_db - float(np.median(band_power_db))

    return {
        "occupied_fraction": occupied_fraction,
        "max_run_fraction": max_run_fraction,
        "peak_concentration_db": peak_concentration_db,
    }


def passes_call_shape_gate(features: dict) -> bool:
    """True if shape features look like a real narrowband call rather than broadband noise."""
    if not config.CALL_SHAPE_GATE_ENABLED:
        return True
    return (
        features["max_run_fraction"] <= config.MAX_CALL_BAND_FRACTION
        and features["peak_concentration_db"] >= config.MIN_PEAK_CONCENTRATION_DB
    )


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
        self._call_start_chunk: int = 0

        self._min_active = max(1, int(np.ceil(config.MIN_CALL_DURATION_MS / chunk_duration_ms)))
        self._end_silence = max(1, int(np.ceil(config.CALL_END_SILENCE_MS / chunk_duration_ms)))
        self._max_active = max(1, int(np.ceil(config.MAX_CALL_DURATION_MS / chunk_duration_ms)))

        # Last-chunk diagnostics, exposed read-only for offline evaluation (evaluate.py).
        # Not used by the live pipeline itself.
        self.last_peak_excess: float = -np.inf
        self.last_shape_features: Optional[dict] = None
        self.last_is_active: bool = False

    def _noise_floor(self) -> np.ndarray:
        if self._floor_dirty and len(self._noise_history) >= 5:
            self._floor_cache = np.median(np.stack(self._noise_history), axis=0)
            self._floor_dirty = False
        return self._floor_cache

    def process(self, power_spectrum: np.ndarray, chunk_time: float, chunk_idx: int = 0) -> list[DetectionEvent]:
        """
        Process one chunk's power spectrum; return any calls that just completed.

        power_spectrum must be the full rfft power array (length CHUNK_SAMPLES//2 + 1).
        chunk_time is the Unix epoch time of the start of this chunk.
        chunk_idx is a monotonic counter used to locate the call in the audio ring buffer.
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

        shape_features = band_shape_features(band_power)
        is_active = (
            peak_excess >= config.DETECTION_THRESHOLD_DB
            and passes_call_shape_gate(shape_features)
        )

        self.last_peak_excess = peak_excess
        self.last_shape_features = shape_features
        self.last_is_active = is_active

        events: list[DetectionEvent] = []

        if is_active:
            self._silence_chunks = 0
            if not self._in_call:
                self._in_call = True
                self._call_start = chunk_time
                self._call_start_chunk = chunk_idx
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
                    events.append(self._close_call(chunk_time, chunk_idx, flagged=True))
        elif self._in_call:
            self._silence_chunks += 1
            if self._silence_chunks >= self._end_silence:
                if self._active_chunks >= self._min_active:
                    events.append(self._close_call(chunk_time, chunk_idx))
                else:
                    self._reset()

        return events

    def flush(self, current_time: float, chunk_idx: int = 0) -> list[DetectionEvent]:
        """Finalize any in-progress call. Call at session end before closing the stream."""
        if not self._in_call:
            return []
        events = []
        if self._active_chunks >= self._min_active:
            events.append(self._close_call(current_time, chunk_idx))
        else:
            self._reset()
        return events

    def _close_call(self, current_time: float, chunk_idx: int = 0, flagged: bool = False) -> DetectionEvent:
        duration_ms = (current_time - self._call_start) * 1000.0
        event = DetectionEvent(
            band=self.band_name,
            peak_freq_hz=self._call_peak_freq,
            power_db=self._call_max_power,
            duration_ms=duration_ms,
            timestamp_abs=self._call_start,
            start_chunk_idx=self._call_start_chunk,
            end_chunk_idx=chunk_idx,
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
        self._call_start_chunk = 0

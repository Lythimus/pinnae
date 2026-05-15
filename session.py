from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrackInfo:
    path: str
    start_time_abs: float   # Unix epoch seconds when this track began playing
    duration_s: float
    usv_band_min_hz: int
    usv_band_max_hz: int


class Session:
    """
    Reads a Squeakorithm session manifest and answers "what track is playing right now?"

    The manifest is the thin coupling layer between Squeakorithm (which writes it)
    and squeaker-listen (which reads it).  Format:

        {
          "session_id": "2026-05-15T14:30:00",
          "sample_rate": 96000,
          "tracks": [
            {
              "path": "output/track1_enriched.wav",
              "start_time_abs": 1747318200.0,
              "duration_s": 183.4,
              "usv_band_min_hz": 33000,
              "usv_band_max_hz": 80000
            }
          ]
        }
    """

    def __init__(self, manifest_path: str) -> None:
        with open(manifest_path) as f:
            data = json.load(f)
        self.session_id: str = data["session_id"]
        self.sample_rate: int = data["sample_rate"]
        self.tracks: list[TrackInfo] = [TrackInfo(**t) for t in data["tracks"]]

    def current_track(self, timestamp_abs: float) -> Optional[tuple[TrackInfo, float]]:
        """Return (track, elapsed_seconds) for the track currently playing, or None."""
        for track in self.tracks:
            elapsed = timestamp_abs - track.start_time_abs
            if 0.0 <= elapsed <= track.duration_s:
                return track, elapsed
        return None

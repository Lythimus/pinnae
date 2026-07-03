from __future__ import annotations

import sqlite3
import threading
from typing import Optional


class EventLog:
    """Thread-safe SQLite event log for detected USV calls."""

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS events (
            id                       INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id               TEXT    NOT NULL,
            timestamp_abs            REAL    NOT NULL,
            timestamp_track_relative REAL,
            track_name               TEXT,
            band                     TEXT    NOT NULL,
            peak_freq_hz             REAL    NOT NULL,
            duration_ms              REAL    NOT NULL,
            power_db                 REAL    NOT NULL,
            audio_path               TEXT,
            video_path               TEXT
        )
    """

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(self._SCHEMA)
            # Migrate existing databases that predate the video_path column.
            existing = {row[1] for row in self._conn.execute("PRAGMA table_info(events)")}
            if "video_path" not in existing:
                self._conn.execute("ALTER TABLE events ADD COLUMN video_path TEXT")
            self._conn.commit()

    def log_event(
        self,
        session_id: str,
        timestamp_abs: float,
        band: str,
        peak_freq_hz: float,
        duration_ms: float,
        power_db: float,
        timestamp_track_relative: Optional[float] = None,
        track_name: Optional[str] = None,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO events
                   (session_id, timestamp_abs, timestamp_track_relative, track_name,
                    band, peak_freq_hz, duration_ms, power_db, audio_path, video_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    timestamp_abs,
                    timestamp_track_relative,
                    track_name,
                    band,
                    peak_freq_hz,
                    duration_ms,
                    power_db,
                    audio_path,
                    video_path,
                ),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

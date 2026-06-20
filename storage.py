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
            flagged_artifact         INTEGER NOT NULL DEFAULT 0
        )
    """

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(self._SCHEMA)
            self._conn.commit()

    def log_event(
        self,
        session_id: str,
        timestamp_abs: float,
        band: str,
        peak_freq_hz: float,
        duration_ms: float,
        power_db: float,
        flagged_artifact: bool = False,
        timestamp_track_relative: Optional[float] = None,
        track_name: Optional[str] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO events
                   (session_id, timestamp_abs, timestamp_track_relative, track_name,
                    band, peak_freq_hz, duration_ms, power_db, flagged_artifact)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    timestamp_abs,
                    timestamp_track_relative,
                    track_name,
                    band,
                    peak_freq_hz,
                    duration_ms,
                    power_db,
                    int(flagged_artifact),
                ),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

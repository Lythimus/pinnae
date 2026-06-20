#!/usr/bin/env python3
"""
squeaker-listen — real-time rat USV detector.

Subcommands:
  baseline  Monitor mic only (no music). Alarm + social_low bands.
  playback  Monitor during Squeakorithm playback using a session manifest.
            Alarm band is clean. Social bands (>33 kHz) may see music bleed-through
            until spectrogram masking is added in Phase 3.
  devices   List available audio devices.

Examples:
  python listen.py baseline --duration 600
  python listen.py baseline --device "Ultramic SO.104" --db session1.db
  python listen.py playback --manifest session.json --db session1.db
  python listen.py devices
"""
from __future__ import annotations

import argparse
import json
import queue
import socket
import sys
import threading
import time
from datetime import datetime
from typing import Optional

import numpy as np
import sounddevice as sd

import config
from detector import BandDetector, DetectionEvent, compute_power_spectrum
from session import Session
from snippets import AudioRingBuffer, save_snippet
from storage import EventLog

# Bands used in each mode.
# social_core / social_high overlap the 33–80 kHz music band; reliable only after
# spectrogram masking is added in Phase 3.
BASELINE_BANDS = ["alarm", "social_low"]
PLAYBACK_BANDS = ["alarm", "social_low"]


def _make_detectors(band_names: list[str]) -> dict[str, BandDetector]:
    return {name: BandDetector(name, *config.BANDS[name]) for name in band_names}


def _run_stream(
    *,
    mode: str,
    db_path: str,
    device: Optional[str],
    duration: Optional[float],
    session: Optional[Session],
    band_names: list[str],
    save_snippets: bool = False,
) -> None:
    session_id = session.session_id if session else datetime.now().isoformat(timespec="seconds")
    db = EventLog(db_path)
    detectors = _make_detectors(band_names)
    event_q: queue.SimpleQueue[Optional[DetectionEvent]] = queue.SimpleQueue()
    ring: Optional[AudioRingBuffer] = AudioRingBuffer() if save_snippets else None

    if isinstance(device, str) and device.lstrip("-").isdigit():
        device = int(device)

    _bcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _bcast_sock.setblocking(False)
    _bcast_addr = ("127.0.0.1", 8765)

    # Calibrated once on first callback so chunk timestamps are in Unix epoch seconds.
    _pa_start: list[Optional[float]] = [None]
    _epoch_start: list[float] = [0.0]

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"[!] {status}", file=sys.stderr)

        adc_time: float = time_info.inputBufferAdcTime
        if _pa_start[0] is None:
            _pa_start[0] = adc_time
            _epoch_start[0] = time.time()
        chunk_time = _epoch_start[0] + (adc_time - _pa_start[0])

        if ring is not None:
            ring.push(chunk_time, indata[:, 0])

        ps = compute_power_spectrum(indata[:, 0])
        for det in detectors.values():
            for event in det.process(ps, chunk_time):
                event_q.put(event)

    def _log_and_print(event: DetectionEvent) -> None:
        track_name: Optional[str] = None
        track_rel: Optional[float] = None
        if session:
            result = session.current_track(event.timestamp_abs)
            if result:
                track, track_rel = result
                track_name = track.path

        db.log_event(
            session_id=session_id,
            timestamp_abs=event.timestamp_abs,
            band=event.band,
            peak_freq_hz=event.peak_freq_hz,
            duration_ms=event.duration_ms,
            power_db=event.power_db,
            flagged_artifact=event.flagged_artifact,
            timestamp_track_relative=track_rel,
            track_name=track_name,
        )
        try:
            payload = json.dumps({"ts": event.timestamp_abs * 1000, "type": event.band}).encode()
            _bcast_sock.sendto(payload, _bcast_addr)
        except OSError:
            pass

        ts = datetime.fromtimestamp(event.timestamp_abs).strftime("%H:%M:%S.%f")[:-3]
        track_str = f"  @track+{track_rel:7.2f}s" if track_rel is not None else ""
        flag_str  = "  [FLAGGED]" if event.flagged_artifact else ""

        if config.SPL_AT_0_DBFS is not None:
            spl_db = event.power_db + config.SPL_AT_0_DBFS
            power_str = f"power={event.power_db:+6.1f} dBFS  spl={spl_db:+6.1f} dB"
        else:
            power_str = f"power={event.power_db:+6.1f} dBFS (uncalibrated)"

        print(
            f"[{ts}] {event.band:<12}  "
            f"peak={event.peak_freq_hz:6.0f} Hz  "
            f"{power_str}  "
            f"dur={event.duration_ms:5.1f} ms"
            f"{track_str}{flag_str}"
        )

        if ring is not None:
            event_end = event.timestamp_abs + event.duration_ms / 1000.0
            save_snippet(ring, event.timestamp_abs, event_end, event.band, event.flagged_artifact)

    stop = threading.Event()

    def consumer() -> None:
        while not stop.is_set():
            try:
                event = event_q.get(timeout=0.1)
                if event is not None:
                    _log_and_print(event)
            except queue.Empty:
                continue
        # Drain any remaining events after stop is set.
        while True:
            try:
                event = event_q.get_nowait()
                if event is not None:
                    _log_and_print(event)
            except queue.Empty:
                break

    consumer_thread = threading.Thread(target=consumer, daemon=True)

    print(f"[squeaker-listen] mode={mode}  session={session_id}")
    print(f"[squeaker-listen] bands={', '.join(band_names)}")
    print(f"[squeaker-listen] sample_rate={config.SAMPLE_RATE} Hz  chunk={config.CHUNK_SAMPLES} samples (~{1000*config.CHUNK_SAMPLES/config.SAMPLE_RATE:.1f} ms)")
    print(f"[squeaker-listen] threshold={config.DETECTION_THRESHOLD_DB} dB above floor  db={db_path}")
    print(f"[squeaker-listen] Ctrl+C to stop.")
    print()

    end_time = time.time() + duration if duration else None
    thread_started = False

    try:
        with sd.InputStream(
            device=device,
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            blocksize=config.CHUNK_SAMPLES,
            dtype="float32",
            callback=callback,
        ):
            consumer_thread.start()
            thread_started = True
            while True:
                if end_time and time.time() >= end_time:
                    break
                time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    except (sd.PortAudioError, ValueError) as exc:
        print(f"\n[squeaker-listen] Audio device error: {exc}", file=sys.stderr)
        print("[squeaker-listen] Run 'python listen.py devices' to list available devices.", file=sys.stderr)
    finally:
        # Flush any call that was still in progress when the stream closed.
        now = time.time()
        for det in detectors.values():
            for event in det.flush(now):
                event_q.put(event)
        stop.set()
        if thread_started:
            consumer_thread.join(timeout=2.0)
        db.close()
        _bcast_sock.close()
        print(f"\n[squeaker-listen] Session ended. Events saved to {db_path}")


def cmd_baseline(args: argparse.Namespace) -> None:
    _run_stream(
        mode="baseline",
        db_path=args.db,
        device=args.device,
        duration=args.duration,
        session=None,
        band_names=BASELINE_BANDS,
        save_snippets=args.save_snippets,
    )


def cmd_playback(args: argparse.Namespace) -> None:
    session = Session(args.manifest)
    print(f"[squeaker-listen] Loaded manifest: {args.manifest}")
    print(f"[squeaker-listen] {len(session.tracks)} track(s) in session {session.session_id}")
    _run_stream(
        mode="playback",
        db_path=args.db,
        device=args.device,
        duration=None,
        session=session,
        band_names=PLAYBACK_BANDS,
        save_snippets=args.save_snippets,
    )


def cmd_devices(_args: argparse.Namespace) -> None:
    print(sd.query_devices())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="listen.py",
        description="Real-time rat USV detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    bl = sub.add_parser("baseline", help="Record without music (alarm + social_low bands)")
    bl.add_argument("--device", default=None, metavar="NAME",
                    help="Audio input device name or index (default: system default)")
    bl.add_argument("--db", default="events.db", metavar="PATH",
                    help="SQLite output path (default: events.db)")
    bl.add_argument("--duration", type=float, default=None, metavar="SECS",
                    help="Stop after this many seconds (default: run until Ctrl+C)")
    bl.add_argument("--save-snippets", action="store_true", default=False,
                    help=f"Save WAV + spectrogram PNG for each detection to {config.DETECTION_SNIPPET_DIR}/")
    bl.set_defaults(func=cmd_baseline)

    pl = sub.add_parser("playback", help="Monitor during Squeakorithm playback")
    pl.add_argument("--manifest", required=True, metavar="PATH",
                    help="Squeakorithm session manifest JSON")
    pl.add_argument("--device", default=None, metavar="NAME",
                    help="Audio input device name or index (default: system default)")
    pl.add_argument("--db", default="events.db", metavar="PATH",
                    help="SQLite output path (default: events.db)")
    pl.add_argument("--save-snippets", action="store_true", default=False,
                    help=f"Save WAV + spectrogram PNG for each detection to {config.DETECTION_SNIPPET_DIR}/")
    pl.set_defaults(func=cmd_playback)

    dv = sub.add_parser("devices", help="List available audio devices")
    dv.set_defaults(func=cmd_devices)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

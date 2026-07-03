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
import os
import queue
import socket
import sys
import threading
import time
from datetime import datetime
from typing import Optional

import math

import numpy as np
import sounddevice as sd

import cv2

import config
from audiowriter import AudioClipWriter, AudioRingBuffer
from detector import BandDetector, DetectionEvent, compute_power_spectrum
from session import Session
from storage import EventLog
from videowriter import VideoClipWriter, VideoRingBuffer, VideoCaptureThread

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
    audio_dir: str,
    enable_audio: bool,
    video_dir: str = config.VIDEO_OUTPUT_DIR,
    enable_video: bool = False,
    video_continuous: bool = False,
    camera: int | str = config.VIDEO_CAMERA_INDEX,
) -> None:
    session_id = session.session_id if session else datetime.now().isoformat(timespec="seconds")
    db = EventLog(db_path)
    detectors = _make_detectors(band_names)
    event_q: queue.SimpleQueue[Optional[DetectionEvent]] = queue.SimpleQueue()

    if isinstance(device, str) and device.lstrip("-").isdigit():
        device = int(device)

    _bcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _bcast_sock.setblocking(False)
    _bcast_addr = ("127.0.0.1", 8765)

    ring_buffer: Optional[AudioRingBuffer] = (
        AudioRingBuffer(config.AUDIO_BUFFER_CHUNKS) if enable_audio else None
    )
    clip_writer: Optional[AudioClipWriter] = (
        AudioClipWriter(audio_dir, config.SAMPLE_RATE, config.AUDIO_FORMAT) if enable_audio else None
    )
    _pre_chunks = math.ceil(config.CLIP_PRE_ROLL_MS * config.SAMPLE_RATE / (config.CHUNK_SAMPLES * 1000.0))
    _post_chunks = math.ceil(config.CLIP_POST_ROLL_MS * config.SAMPLE_RATE / (config.CHUNK_SAMPLES * 1000.0))

    # Video setup — independent capture thread + timestamp-keyed ring buffer.
    video_ring: Optional[VideoRingBuffer] = None
    video_clip_writer: Optional[VideoClipWriter] = None
    capture_thread: Optional[VideoCaptureThread] = None
    if enable_video:
        continuous_writer: Optional[cv2.VideoWriter] = None
        if video_continuous:
            cont_dir = os.path.join(video_dir, "continuous")
            os.makedirs(cont_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            cont_path = os.path.join(cont_dir, f"{session_id.replace(':', '-')}.mp4")
            continuous_writer = cv2.VideoWriter(
                cont_path, fourcc, config.VIDEO_FPS, (config.VIDEO_WIDTH, config.VIDEO_HEIGHT)
            )
        video_ring = VideoRingBuffer(config.VIDEO_BUFFER_FRAMES)
        video_clip_writer = VideoClipWriter(video_dir, config.VIDEO_FPS, config.VIDEO_WIDTH, config.VIDEO_HEIGHT)
        capture_thread = VideoCaptureThread(camera, video_ring, continuous_writer)
        if not capture_thread.start():
            # Camera unavailable — disable video without crashing audio pipeline.
            video_ring = None
            video_clip_writer = None
            capture_thread = None

    # Per-band last-clip-end-time for bout de-duplication.
    # Rats call in bursts; overlapping ±5 s windows would produce many near-identical clips.
    # If the new clip's window start is already inside the previous clip, skip it.
    _last_video_clip_end: dict[str, float] = {}

    # Calibrated once on first callback so chunk timestamps are in Unix epoch seconds.
    _pa_start: list[Optional[float]] = [None]
    _epoch_start: list[float] = [0.0]
    _chunk_idx: list[int] = [0]

    def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"[!] {status}", file=sys.stderr)

        adc_time: float = time_info.inputBufferAdcTime
        if _pa_start[0] is None:
            _pa_start[0] = adc_time
            _epoch_start[0] = time.time()
        chunk_time = _epoch_start[0] + (adc_time - _pa_start[0])

        idx = _chunk_idx[0]
        if ring_buffer is not None:
            ring_buffer.append(idx, indata[:, 0].copy())
        ps = compute_power_spectrum(indata[:, 0])
        for det in detectors.values():
            for event in det.process(ps, chunk_time, idx):
                event_q.put(event)
        _chunk_idx[0] += 1

    def _log_and_print(event: DetectionEvent) -> None:
        audio_path: Optional[str] = None
        if ring_buffer is not None and clip_writer is not None:
            try:
                s_idx = max(0, event.start_chunk_idx - _pre_chunks)
                e_idx = event.end_chunk_idx + _post_chunks
                samples = ring_buffer.extract(s_idx, e_idx)
                if samples is not None:
                    audio_path = clip_writer.write_clip(event.band, event.timestamp_abs, samples)
            except Exception as exc:
                print(f"[!] clip write failed: {exc}", file=sys.stderr)

        video_path: Optional[str] = None
        if video_ring is not None and video_clip_writer is not None:
            t_start = event.timestamp_abs - config.VIDEO_CLIP_PRE_ROLL_S
            t_end = event.timestamp_abs + event.duration_ms / 1000.0 + config.VIDEO_CLIP_POST_ROLL_S
            last_end = _last_video_clip_end.get(event.band, 0.0)
            if t_start >= last_end:
                try:
                    frames = video_ring.extract(t_start, t_end)
                    if frames:
                        video_path = video_clip_writer.write_clip(event.band, event.timestamp_abs, frames)
                        _last_video_clip_end[event.band] = t_end
                except Exception as exc:
                    print(f"[!] video clip write failed: {exc}", file=sys.stderr)

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
            audio_path=audio_path,
            video_path=video_path,
        )
        try:
            payload = json.dumps({"ts": event.timestamp_abs * 1000, "type": event.band}).encode()
            _bcast_sock.sendto(payload, _bcast_addr)
        except OSError:
            pass

        ts = datetime.fromtimestamp(event.timestamp_abs).strftime("%H:%M:%S.%f")[:-3]
        track_str = f"  @track+{track_rel:7.2f}s" if track_rel is not None else ""
        clip_str = f"  → {audio_path}" if audio_path else ""
        vid_str = f"  → {video_path}" if video_path else ""
        flag_str = "  [FLAGGED]" if event.flagged_artifact else ""

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
            f"{track_str}"
            f"{flag_str}"
            f"{clip_str}"
            f"{vid_str}"
        )

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
    if enable_audio:
        print(f"[squeaker-listen] audio clips → {audio_dir}/  (±{config.CLIP_PRE_ROLL_MS:.0f} ms pre/post roll)")
    if capture_thread is not None:
        print(f"[squeaker-listen] video clips → {video_dir}/  (±{config.VIDEO_CLIP_PRE_ROLL_S:.0f} s pre/post roll  camera={camera})")
        if video_continuous:
            print(f"[squeaker-listen] continuous video → {video_dir}/continuous/")
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
        final_idx = _chunk_idx[0]
        for det in detectors.values():
            for event in det.flush(now, final_idx):
                event_q.put(event)
        stop.set()
        if thread_started:
            consumer_thread.join(timeout=2.0)
        if capture_thread is not None:
            capture_thread.stop()
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
        audio_dir=args.audio_dir,
        enable_audio=not args.no_audio,
        video_dir=args.video_dir,
        enable_video=args.video,
        video_continuous=args.video_continuous,
        camera=args.camera,
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
        audio_dir=args.audio_dir,
        enable_audio=not args.no_audio,
        video_dir=args.video_dir,
        enable_video=args.video,
        video_continuous=args.video_continuous,
        camera=args.camera,
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
    bl.add_argument("--audio-dir", default=config.AUDIO_OUTPUT_DIR, metavar="PATH",
                    help=f"Directory for FLAC audio clips (default: {config.AUDIO_OUTPUT_DIR})")
    bl.add_argument("--no-audio", action="store_true",
                    help="Disable audio clip recording")
    bl.add_argument("--video", action="store_true",
                    help="Enable video clip recording (requires a connected camera)")
    bl.add_argument("--video-dir", default=config.VIDEO_OUTPUT_DIR, metavar="PATH",
                    help=f"Directory for mp4 video clips (default: {config.VIDEO_OUTPUT_DIR})")
    bl.add_argument("--video-continuous", action="store_true",
                    help="Also write a continuous full-session video to video/continuous/")
    bl.add_argument("--camera", default=config.VIDEO_CAMERA_INDEX, metavar="INDEX",
                    help=f"Camera device index or path (default: {config.VIDEO_CAMERA_INDEX})")
    bl.set_defaults(func=cmd_baseline)

    pl = sub.add_parser("playback", help="Monitor during Squeakorithm playback")
    pl.add_argument("--manifest", required=True, metavar="PATH",
                    help="Squeakorithm session manifest JSON")
    pl.add_argument("--device", default=None, metavar="NAME",
                    help="Audio input device name or index (default: system default)")
    pl.add_argument("--db", default="events.db", metavar="PATH",
                    help="SQLite output path (default: events.db)")
    pl.add_argument("--audio-dir", default=config.AUDIO_OUTPUT_DIR, metavar="PATH",
                    help=f"Directory for FLAC audio clips (default: {config.AUDIO_OUTPUT_DIR})")
    pl.add_argument("--no-audio", action="store_true",
                    help="Disable audio clip recording")
    pl.add_argument("--video", action="store_true",
                    help="Enable video clip recording (requires a connected camera)")
    pl.add_argument("--video-dir", default=config.VIDEO_OUTPUT_DIR, metavar="PATH",
                    help=f"Directory for mp4 video clips (default: {config.VIDEO_OUTPUT_DIR})")
    pl.add_argument("--video-continuous", action="store_true",
                    help="Also write a continuous full-session video to video/continuous/")
    pl.add_argument("--camera", default=config.VIDEO_CAMERA_INDEX, metavar="INDEX",
                    help=f"Camera device index or path (default: {config.VIDEO_CAMERA_INDEX})")
    pl.set_defaults(func=cmd_playback)

    dv = sub.add_parser("devices", help="List available audio devices")
    dv.set_defaults(func=cmd_devices)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

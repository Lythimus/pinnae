# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`pinnae` (working name: `squeaker-listen`) — a real-time rat USV (ultrasonic vocalization) detector. It listens via a Dodotronic Ultramic SO.104 at 192 kHz, detects calls in defined frequency bands using FFT thresholding, and logs events to SQLite. It is a companion tool to Squeakorithm (separate repo), which plays enrichment audio; the two communicate only through a thin session manifest JSON.

See `ARCHITECTURE-PLAN.md` for the full rationale, phased roadmap, and band/overlap analysis.

## Setup and running

```bash
pip install -r requirements.txt

python listen.py devices                                        # find the SO.104 device name
python listen.py baseline --device "Ultramic SO.104" --db events.db
python listen.py baseline --duration 600                       # stop after 10 min
python listen.py playback --manifest session.json --db events.db
```

There is no build step and no test suite yet. Syntax-check all modules with:

```bash
python3 -c "import ast, sys; [ast.parse(open(f).read()) or print('OK', f) for f in ['config.py','detector.py','storage.py','session.py','audiowriter.py','videowriter.py','listen.py','evaluate.py']]"
```

## Architecture

The pipeline per 42.7 ms audio chunk (8192 samples @ 192 kHz):

```
sounddevice callback (audio thread)
  └─ compute_power_spectrum(chunk)     # Hann-windowed rfft → dBFS array
       └─ BandDetector.process(ps, t)  # one instance per monitored band
            └─ queue.SimpleQueue       # events handed off to main thread
                 └─ EventLog.log_event → SQLite
```

**`config.py`** is the single source of truth for all constants — band Hz ranges, `SAMPLE_RATE`, `CHUNK_SAMPLES`, and detection thresholds. Change values here; nothing is hard-coded elsewhere.

**`detector.py`** — `BandDetector` maintains a rolling 50-chunk noise floor (median of per-chunk band medians) and an onset/offset state machine. A call opens when peak band power exceeds `DETECTION_THRESHOLD_DB` above the floor, **and** passes the call-shape gate, and closes after `CALL_END_SILENCE_MS` of silence. Call `flush()` before closing the stream to capture any in-progress call. `compute_power_spectrum` is a module-level function used by `listen.py`; it is not a method of `BandDetector`.

**Call-shape gate** (`CALL_SHAPE_GATE_ENABLED` in `config.py`) — broadband transients (e.g. music-playback-chain distortion, percussive hash) can cross `DETECTION_THRESHOLD_DB` without being a narrowband FM call. `band_shape_features()` computes, per chunk, an occupied-band fraction and the widest contiguous run of bins within `BANDWIDTH_DROPOFF_DB` of *the chunk's own peak power in that band* (a standard "-X dB bandwidth" measure), plus peak-to-median concentration in dB. `passes_call_shape_gate()` rejects onset when the occupied run spans more than `MAX_CALL_BAND_FRACTION` of the band or concentration falls below `MIN_PEAK_CONCENTRATION_DB`. The occupied-bin threshold is deliberately peak-relative, not floor-relative: an earlier floor-relative "shoulder" design saturated to ~100% occupancy on real 22 kHz alarm calls (general chunk-wide energy elevation above the floor, not just the call, was miscounted as occupied), which cut recall to ~50% at defaults — measured with `evaluate.py` against the Olszynski & Polowy alarm playback WAVs and the tickling recording. `MIN_PEAK_CONCENTRATION_DB` is kept low/permissive because concentration varies a lot with recording context (busy multi-rat cage recording vs. curated clean stimulus WAV) — `max_run_fraction` does most of the rejection work. Current defaults preserve ~93-100% recall on those two real positive corpora but are **not yet validated against a real music negative** — capture one with `listen.py record` and re-run `evaluate.py --sweep` before trusting the rejection side in the field. `BandDetector` also exposes `last_peak_excess` / `last_shape_features` / `last_is_active` as read-only diagnostics for `evaluate.py`; the live pipeline doesn't use them.

**`session.py`** — reads the Squeakorithm manifest JSON and answers `current_track(epoch_time) → (TrackInfo, elapsed_s)`. Used only in playback mode to attach track-relative timestamps to events.

**`storage.py`** — thread-safe `EventLog` wraps SQLite with a mutex. The consumer thread (not the audio callback) calls `log_event`; the audio callback only enqueues `DetectionEvent` objects. Events include `audio_path TEXT`, `video_path TEXT`, and `valence TEXT` columns. On init, the schema runs a migration: any of these columns absent in an existing DB is added via `ALTER TABLE`.

**`audiowriter.py`** — `AudioRingBuffer` retains the last `AUDIO_BUFFER_SECONDS` (30 s) of raw mic chunks tagged by index; `AudioClipWriter` writes lossless FLAC clips to `audio/{band}/{datetime}.flac`. Both classes are guarded by `threading.Lock`, mirroring `EventLog`'s mutex pattern.

**`videowriter.py`** — optional video capture (enabled with `--video`). `VideoRingBuffer` stores the last `VIDEO_BUFFER_SECONDS` (30 s) of camera frames as JPEG bytes (~135 MB vs ~2.4 GB raw) tagged by Unix timestamp. `VideoCaptureThread` is a daemon thread that opens a UVC camera via OpenCV (`cv2.VideoCapture`), runs at ~30 fps, and feeds the ring buffer; if the camera fails to open it prints a warning and disables video without crashing the audio pipeline. `VideoClipWriter.write_clip` extracts a ±5 s window around each call and writes an mp4. Bout de-duplication skips overlapping windows so a burst of calls produces one clip, not dozens. Designed for IR-modified USB cameras under infrared illumination — camera-agnostic via OpenCV index or path.

**`listen.py`** — audio callback stays thin: ring-buffer append + FFT + enqueue only. A daemon consumer thread drains the queue and handles all I/O (print + SQLite + FLAC write + mp4 write). Timestamps use PortAudio's ADC clock calibrated on first callback invocation (`inputBufferAdcTime`) rather than `time.time()` per chunk. Pass `--no-audio` to disable audio clips; `--audio-dir` overrides the default `audio/` directory. Pass `--video` to enable video clips (off by default); `--video-dir`, `--video-continuous`, and `--camera` control the camera index/path, output directory, and whether a continuous full-session mp4 is also written. Each event is tagged with a `valence` label (`_classify_valence`) — a minimal band/peak-frequency heuristic (22 kHz alarm and 44 kHz → `negative`, 50–70 kHz → `positive`, else `unknown`), unvalidated groundwork rather than a trusted classifier. The `record` subcommand writes raw mic input straight to a WAV with no detection/gate applied — used to capture negative corpora (e.g. music played back with no rat present) for `evaluate.py`.

**`evaluate.py`** — standalone offline harness that replays labeled WAV corpora through the real `compute_power_spectrum` + `BandDetector` pipeline to measure and tune the call-shape gate. Resamples every input file to `config.SAMPLE_RATE` via `scipy.signal.resample_poly` (eval-only dependency, not in `requirements.txt`'s runtime set). `--gate on|off` toggles `CALL_SHAPE_GATE_ENABLED` for a single-run report of per-file/per-band detection counts; `--sweep` grid-searches `MAX_CALL_BAND_FRACTION` × `MIN_PEAK_CONCENTRATION_DB` and reports positive-recall-kept vs. negative-rejected fractions relative to a gate-off baseline, plus prints raw shape-feature percentiles (pos vs. neg) so thresholds are chosen from data.

## Frequency bands and overlap

Squeakorithm enforces a **33 kHz hard floor** on all USV generators, so the **alarm band (18–26 kHz) is always clean** — no music energy, no masking needed. `social_low` (30–45 kHz) has a clean wedge at 30–33 kHz but overlaps music above 33 kHz.

`BASELINE_BANDS` and `PLAYBACK_BANDS` in `listen.py` both use `["alarm", "social_low"]`. `social_core` (45–70 kHz) and `social_high` (70–80 kHz) are defined in `config.BANDS` but are not active until Phase 3 spectrogram masking is implemented.

## Phase 3 notes (not yet implemented)

Phase 3 adds duplex streaming: push FLAC playback samples to the audio interface while pulling mic input, compute both STFTs per chunk, and zero mic bins where playback energy exceeds `MASK_FLOOR_DB`. The propagation delay constant (`PROPAGATION_DELAY_S = 0.00534` s, 6 ft room) is already in `config.py`. `soundfile` is already an active dependency (added for clip recording).

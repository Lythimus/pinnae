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
python3 -c "import ast, sys; [ast.parse(open(f).read()) or print('OK', f) for f in ['config.py','detector.py','storage.py','session.py','audiowriter.py','listen.py']]"
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

**`detector.py`** — `BandDetector` maintains a rolling 50-chunk noise floor (median of per-chunk band medians) and an onset/offset state machine. A call opens when peak band power exceeds `DETECTION_THRESHOLD_DB` above the floor and closes after `CALL_END_SILENCE_MS` of silence. Call `flush()` before closing the stream to capture any in-progress call. `compute_power_spectrum` is a module-level function used by `listen.py`; it is not a method of `BandDetector`.

**`session.py`** — reads the Squeakorithm manifest JSON and answers `current_track(epoch_time) → (TrackInfo, elapsed_s)`. Used only in playback mode to attach track-relative timestamps to events.

**`storage.py`** — thread-safe `EventLog` wraps SQLite with a mutex. The consumer thread (not the audio callback) calls `log_event`; the audio callback only enqueues `DetectionEvent` objects. Events include an `audio_path TEXT` column linking to the saved FLAC clip.

**`audiowriter.py`** — `AudioRingBuffer` retains the last `AUDIO_BUFFER_SECONDS` (30 s) of raw mic chunks tagged by index; `AudioClipWriter` writes lossless FLAC clips to `audio/{band}/{datetime}.flac`. Both classes are guarded by `threading.Lock`, mirroring `EventLog`'s mutex pattern.

**`listen.py`** — audio callback stays thin: ring-buffer append + FFT + enqueue only. A daemon consumer thread drains the queue and handles all I/O (print + SQLite + FLAC write). Timestamps use PortAudio's ADC clock calibrated on first callback invocation (`inputBufferAdcTime`) rather than `time.time()` per chunk. Pass `--no-audio` to disable clip recording; `--audio-dir` overrides the default `audio/` directory.

## Frequency bands and overlap

Squeakorithm enforces a **33 kHz hard floor** on all USV generators, so the **alarm band (18–26 kHz) is always clean** — no music energy, no masking needed. `social_low` (30–45 kHz) has a clean wedge at 30–33 kHz but overlaps music above 33 kHz.

`BASELINE_BANDS` and `PLAYBACK_BANDS` in `listen.py` both use `["alarm", "social_low"]`. `social_core` (45–70 kHz) and `social_high` (70–80 kHz) are defined in `config.BANDS` but are not active until Phase 3 spectrogram masking is implemented.

## Phase 3 notes (not yet implemented)

Phase 3 adds duplex streaming: push FLAC playback samples to the audio interface while pulling mic input, compute both STFTs per chunk, and zero mic bins where playback energy exceeds `MASK_FLOOR_DB`. The propagation delay constant (`PROPAGATION_DELAY_S = 0.00534` s, 6 ft room) is already in `config.py`. `soundfile` is already an active dependency (added for clip recording).

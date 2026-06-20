# Pinnae

Real-time rat ultrasonic vocalization (USV) detector. Listens through a high-frequency microphone, detects calls in defined frequency bands via FFT thresholding, and logs every event to a SQLite database with precise timestamps.

Designed to run alongside [Squeakorithm](https://github.com/josephcoco/squeakorithm) enrichment sessions, but works entirely standalone.

---

## Requirements

- Python 3.10+
- [Sonorous Objects SO.104](https://sonorousobjects.nyc/products/so-104-ultrasonic-omni-microphone) (or any mic that reaches 192 kHz)
- `sounddevice`, `numpy` (see `requirements.txt`)

### Windows 11 Setup

1. **Install Python 3.10+**
   - Download from [python.org](https://www.python.org/downloads/) or install via Microsoft Store
   - During installation, check "Add Python to PATH"

2. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

   The PortAudio library is bundled with `sounddevice` on Windows — no additional system packages needed.

3. **Connect your microphone**
   - Plug in the USB microphone (or connect via audio interface)
   - Windows should recognize it automatically
   - Verify with `python listen.py devices`
   - If using an audio interface (e.g. MOTU M Series), prefer the **Windows WASAPI** variant from the list — lowest latency, least signal processing. A mic on IN 1L is channel 0 of the stereo input.

**Note:** WSL2 does not support USB audio devices properly. You must run Pinnae on native Windows.

### macOS Setup

PortAudio is required by sounddevice:

```bash
brew install portaudio
pip install -r requirements.txt
```

### Linux Setup

Install PortAudio development libraries:

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev
pip install -r requirements.txt
```

---

## Home Ratification

Before running the detector, read the [**Ratification Guide**](RATIFICATION.md) to survey your home for non-rat sources of ultrasonic noise (CFLs, LED bulbs, chargers, electronics, etc.). Eliminating these noise sources ensures better quality of life for rats and accurate detection.

---

## Live dashboard

A browser dashboard shows the real-time event timeline, rolling-baseline statistics (1 H / 3 H / 6 H / 12 H / 1 D / 1 W / 1 MO), and per-band activity. It runs as a separate server process so the HTTP/WebSocket load never touches the audio capture thread.

### Start the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

On Windows 11 the `uvicorn` script may not be on your `PATH`. Either use:

```powershell
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

or add `%APPDATA%\Python\Python3xx\Scripts` (e.g. `C:\Users\<you>\AppData\Roaming\Python\Python313\Scripts`) to your user `PATH` via **System Properties → Environment Variables**.

Open `http://<lan-ip>:8000` in any browser on the same network.

### Start capture alongside the server

Run both in separate terminals:

```bash
# Terminal 1 — capture
python listen.py baseline --device "Ultramic 384K BL" --db events.db

# Terminal 2 — dashboard server
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

The detector broadcasts each event over UDP loopback (`127.0.0.1:8765`); the server fans it out to all connected WebSocket clients in real time. If the server is not running, the packet drops silently and SQLite still receives the row — no audio impact.

### Architecture

```
listen.py (audio process)          server/ (uvicorn process)
  audio callback → FFT                UDP listener ← UDP 127.0.0.1:8765
  consumer thread → SQLite            ConnectionManager → WebSocket clients
  consumer thread → UDP send          GET /api/usv/events  (SQLite read-only)
                                      GET /              (static/index.html)
```

- `events.db` is the single source of truth. The server opens it read-only in WAL mode so reads never block writes.
- On browser connect the dashboard fetches history via `GET /api/usv/events?since=<ms>&limit=2000`, then opens the WebSocket for live events — no gap in coverage.

### Verify the server independently

Inject a fake event without the mic:

```bash
python -c "
import socket, json, time
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.sendto(json.dumps({'ts': time.time()*1000, 'type': 'alarm'}).encode(), ('127.0.0.1', 8765))
"
```

The dashboard should flash an alarm tick within ~50 ms.

---

## Usage

### Find your microphone

```bash
python listen.py devices
```

Look for your microphone in the output and note its name or index. On Windows, if the same device appears under multiple APIs (MME, DirectSound, WASAPI), use the **WASAPI** entry.

### Baseline recording (no music playing)

```bash
python listen.py baseline --device "Ultramic 384K BL" --db session.db
python listen.py baseline --device "Ultramic 384K BL" --db session.db --duration 600
```

Monitors the **alarm band (18–26 kHz)** and **social-low band (30–45 kHz)**.

### Playback mode (Squeakorithm session running)

```bash
python listen.py playback --manifest session.json --db session.db
```

Same bands as baseline. Each logged event additionally records `track_name` and `timestamp_track_relative` (seconds since the track started) by cross-referencing the session manifest.

### Inspect logged events

```bash
sqlite3 session.db "SELECT datetime(timestamp_abs,'unixepoch','localtime'), band, peak_freq_hz, duration_ms, power_db FROM events ORDER BY timestamp_abs;"
```

---

## Detection bands

| Band | Range | Clean during playback? |
|---|---|---|
| `alarm` | 18–26 kHz | **Yes** — 7 kHz gap below Squeakorithm's 33 kHz floor |
| `social_low` | 30–45 kHz | Mostly (30–33 kHz clean; 33–45 kHz partial overlap) |
| `social_core` | 45–70 kHz | No — requires spectrogram masking (Phase 3) |
| `social_high` | 70–80 kHz | No — requires spectrogram masking (Phase 3) |

---

## How detection works

Each 42.7 ms audio chunk (8192 samples at 192 kHz) is processed in the sounddevice callback:

1. Apply a Hann window and compute the one-sided FFT power spectrum (dBFS).
2. For each monitored band, find the peak power and its frequency.
3. Compare peak power against a rolling noise floor (50-chunk median of band medians, ~2 s window).
4. If peak exceeds floor by ≥ `DETECTION_THRESHOLD_DB` (default 20 dB): mark the chunk as active.
5. A call is finalized and logged when `CALL_END_SILENCE_MS` (15 ms) of sub-threshold chunks follow.
6. Calls shorter than `MIN_CALL_DURATION_MS` (5 ms) are discarded as transients.

All thresholds are in `config.py`.

---

## Squeakorithm integration

Pinnae and Squeakorithm are decoupled. The only link is an optional session manifest JSON that Squeakorithm writes when `--write-manifest` is passed to `enrich_track.py`:

```json
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
```

Without a manifest, `playback` mode still detects calls — it just omits the track-relative timestamp columns.

---

## Roadmap

- **Phase 3** — Duplex stream: push Squeakorithm FLAC output to speaker while pulling mic; apply per-bin spectrogram mask (mic bins zeroed where playback energy exceeds threshold, with 5.3 ms propagation delay compensation). Enables reliable `social_core` and `social_high` detection during music.
- **Phase 4** — Lightweight CNN (ONNX, <2 ms/chunk) trained on baseline ground-truth clips from DeepSqueak export to replace FFT threshold for the 45–80 kHz social range.

---

## Future features

- **`--no-broadcast` flag** — suppress the UDP send in `listen.py` for quiet/headless runs where the dashboard server is not running.
- **`max_ts` history ceiling on the dashboard** — the `/api/usv/events` endpoint already accepts a `max_ts` query parameter; the dashboard client should pass `max_ts=<t_open_ms>` (the WebSocket open time) when fetching history so the WS buffer and the backfill HTTP response have a hard overlap boundary, eliminating any dupe/skip at the handoff seam.
- **`--db` flag for the server** — currently the server always reads `events.db` in the working directory; a CLI flag (or env var) would let it point at an arbitrary path, matching the `--db` option in `listen.py`.
- **Minute-bucket pre-aggregation** — for long-running sessions the full-resolution history payload can grow large; a `minute_bucket` materialized view would keep `GET /api/usv/events` fast beyond ~100k rows.
- **5-second server-side reconciler** — if the UDP socket buffer fills under a high burst rate, live packets can drop; a background task emitting `MAX(id)` deltas from SQLite would close the gap for reconnecting clients.
- **spectrogram masking (Phase 3)** — `social_core` (45–70 kHz) and `social_high` (70–80 kHz) bands are defined in `config.py` but disabled until the duplex masking pipeline is in place; enabling them in `PLAYBACK_BANDS` before then will produce false positives from music bleed-through.

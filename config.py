import math

# ── Hardware ──────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 192_000   # Hz — Dodotronic Ultramic SO.104 capability
CHANNELS      = 1
CHUNK_SAMPLES = 8192      # ~42.7 ms per chunk at 192 kHz

# ── Frequency bands (Hz) ─────────────────────────────────────────────────────
# Band definitions from USV literature, aligned to Squeakorithm config constants.
# Squeakorithm enforces a 33 kHz hard floor on all USV generators, so
# 18–26 kHz (alarm) is fully clean during enrichment playback.
BAND_ALARM       = (18_000,  26_000)
BAND_SOCIAL_LOW  = (30_000,  45_000)
BAND_SOCIAL_CORE = (45_000,  70_000)
BAND_SOCIAL_HIGH = (70_000,  80_000)
BAND_PUP         = (35_000,  70_000)

BANDS: dict[str, tuple[int, int]] = {
    "alarm":       BAND_ALARM,
    "social_low":  BAND_SOCIAL_LOW,
    "social_core": BAND_SOCIAL_CORE,
    "social_high": BAND_SOCIAL_HIGH,
    "pup":         BAND_PUP,
}

# ── Detection thresholds ──────────────────────────────────────────────────────
# Literature-confirmed 22 kHz alarm call parameters (verified from primary PDFs):
#   Duration:      100 ms – well over 3000 ms; Portfors 2007 reports 300–4000 ms;
#                  short calls <100 ms exist but are rare (Brudzynski 2009 p4)
#   Peak frequency: 20–23 kHz typical, can reach 26 kHz; Litvin 2007 p2: 18–24 kHz
#   SPL:           65–85 dB SPL (Portfors 2007 p1)
#   Bandwidth:     narrow, 1–4 kHz; minimal FM (nearly flat)
#   50 kHz calls:  30–50 ms, 32–96 kHz, 1–7 kHz bandwidth (Portfors 2007 p1)
DETECTION_THRESHOLD_DB = 20.0   # dB above rolling noise floor to count as active
MIN_CALL_DURATION_MS   = 85.0   # discard transients shorter than this (~2 chunks)
CALL_END_SILENCE_MS    = 42.7   # consecutive below-threshold time to close a call (1 chunk)
# Per-bin floor window: long enough that a genuine <=4 s call doesn't absorb its
# own floor, short enough to track slow drift. ~30 s / 700 chunks.
NOISE_HISTORY_CHUNKS   = 700    # rolling window length (~29.9 s at 42.7 ms/chunk)
MAX_CALL_DURATION_MS   = 4500.0 # flag continuous events longer than literature ceiling

# ── Call-shape gate ───────────────────────────────────────────────────────────
# Broadband transients (music playback distortion, percussive hash) can exceed
# DETECTION_THRESHOLD_DB without being a narrowband FM call. A real USV
# concentrates energy in a bounded, contiguous sub-band around its own peak;
# broadband noise lights up most of the band relative to its peak too. Gate
# onset on shape in addition to energy. Values below were measured with
# evaluate.py against real alarm/social calls (Olszynski & Polowy playback WAVs,
# tickling recording) — see git history for the sweep that picked them. They
# still need re-validation against a real mic-recorded music negative (capture
# one with `listen.py record`) before trusting the *rejection* side in the field.
CALL_SHAPE_GATE_ENABLED   = True
# Bins count as "occupied" when within this many dB of the chunk's own peak
# power in the band — a standard -X dB bandwidth measure. Deliberately *not*
# relative to the noise floor: a chunk's general energy can sit slightly above
# the floor across the whole band (harmonics, resampling artifacts) even when
# the call itself is narrowband, which saturates a floor-relative measure.
BANDWIDTH_DROPOFF_DB      = 12.0
# Reject if the widest contiguous occupied run spans more than this fraction
# of the band's bins (tolerates FM trill smear across one 42.7 ms chunk, and
# the alarm band's real calls occupying a large share of its narrow 8 kHz width).
MAX_CALL_BAND_FRACTION    = 0.9
# Reject if peak_power - median(band_power) falls below this — real calls
# concentrate energy near the peak; broadband hash does not. Kept low/permissive:
# measured concentration varies a lot with recording context (a busy multi-rat
# cage recording has a noisier band median than a curated clean stimulus WAV),
# so this alone is not a reliable cross-context discriminator — max_run_fraction
# does most of the rejection work.
MIN_PEAK_CONCENTRATION_DB = 6.0

# ── Offender-bin masking (electronic interference) ───────────────────────────
# Steady electronic tones are already absorbed by the per-bin rolling floor
# (see BandDetector docstring), but intermittent narrowband transients (e.g.
# switching-supply/PWM/relay chatter) can spike faster than the floor adapts
# and still pass the call-shape gate above, since they're narrowband too.
# NOISE_NOTCH_HZ excludes specific offender frequency windows from the peak
# search entirely, in every band including alarm — targeted rather than a
# blanket subtraction, so it can't clip a real call at a neighboring bin.
# Windows are derived from an electronics-only capture (electronics running,
# no rat; `listen.py record`) characterized with evaluate.py. Empty until that
# capture is taken on the deployed rig — masking is a no-op until populated.
NOISE_NOTCH_ENABLED = True
NOISE_NOTCH_HZ: list[tuple[float, float]] = []   # [(f_lo_hz, f_hi_hz), ...]

# ── Constant-frequency (FM) rejection gate ───────────────────────────────────
# Rat social USVs are frequency-modulated (they glide); electronic tones sit
# at a fixed frequency. Reject a closing call in FM_GATE_BANDS whose peak
# frequency never moved more than MIN_FREQ_MODULATION_HZ across its duration.
# Deliberately excludes "alarm": 22 kHz alarm calls are themselves
# near-constant-frequency (see literature notes above), so this gate would
# delete real ones there. Off by default and MIN_FREQ_MODULATION_HZ is a
# placeholder — enable only after choosing a threshold from real pos-vs-neg
# freq_span_hz percentiles via evaluate.py (see detector.py DetectionEvent).
FM_GATE_ENABLED = False
MIN_FREQ_MODULATION_HZ = 1000.0
FM_GATE_BANDS: list[str] = ["social_low"]

# ── SPL calibration ───────────────────────────────────────────────────────────
# SPL_AT_0_DBFS: offset so that dB_SPL = power_dbfs + SPL_AT_0_DBFS.
# To calibrate: play a reference tone at a known SPL at the mic, record the peak
# dBFS value reported by the detector, then:
#   SPL_AT_0_DBFS = known_SPL_dB - measured_peak_dbfs
# Obtain the SO.104 sensitivity (dBV/Pa) from the Dodotronic datasheet and the
# actual recording gain setting, or do a one-point calibration with a known source.
# Until calibrated, reported dB SPL will be labeled "(uncalibrated)".
SPL_AT_0_DBFS = None  # set to float (e.g., 105.0) once calibrated

# ── Playback masking (Phase 3) ────────────────────────────────────────────────
# Acoustic propagation: 6 ft / 343 m·s⁻¹ ≈ 5.3 ms speaker-to-mic delay.
PROPAGATION_DELAY_S = 0.00534
# Playback energy above this threshold (dB) masks the corresponding mic bin.
MASK_FLOOR_DB = -40.0

# ── Audio clip recording ──────────────────────────────────────────────────────
AUDIO_OUTPUT_DIR     = "audio"   # base directory for per-call FLAC clips
AUDIO_FORMAT         = "flac"    # soundfile format string ("wav" also valid)
AUDIO_BUFFER_SECONDS = 30.0      # ring-buffer duration (seconds of raw audio)
CLIP_PRE_ROLL_MS     = 200.0     # context before call onset
CLIP_POST_ROLL_MS    = 200.0     # context after call offset
AUDIO_BUFFER_CHUNKS  = math.ceil(AUDIO_BUFFER_SECONDS * SAMPLE_RATE / CHUNK_SAMPLES)

# ── Video clip recording (optional, requires --video) ─────────────────────────
VIDEO_OUTPUT_DIR       = "video"   # base directory for per-call mp4 clips
VIDEO_FORMAT           = "mp4"     # container; codec "mp4v"
VIDEO_FPS              = 30
VIDEO_WIDTH            = 1280      # 720p default; bump to 1920x1080 for finer scoring
VIDEO_HEIGHT           = 720
VIDEO_CAMERA_INDEX     = 0         # cv2.VideoCapture index or device path
VIDEO_BUFFER_SECONDS   = 30.0      # ring-buffer duration (seconds of camera frames)
VIDEO_CLIP_PRE_ROLL_S  = 5.0       # context before call onset
VIDEO_CLIP_POST_ROLL_S = 5.0       # context after call offset
VIDEO_JPEG_QUALITY     = 80        # ring-buffer frame compression (memory bound)
VIDEO_BUFFER_FRAMES    = math.ceil(VIDEO_BUFFER_SECONDS * VIDEO_FPS)

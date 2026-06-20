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

# ── SPL calibration ───────────────────────────────────────────────────────────
# SPL_AT_0_DBFS: offset so that dB_SPL = power_dbfs + SPL_AT_0_DBFS.
# To calibrate: play a reference tone at a known SPL at the mic, record the peak
# dBFS value reported by the detector, then:
#   SPL_AT_0_DBFS = known_SPL_dB - measured_peak_dbfs
# Obtain the SO.104 sensitivity (dBV/Pa) from the Dodotronic datasheet and the
# actual recording gain setting, or do a one-point calibration with a known source.
# Until calibrated, reported dB SPL will be labeled "(uncalibrated)".
SPL_AT_0_DBFS = None  # set to float (e.g., 105.0) once calibrated

# ── Snippet capture (--save-snippets) ─────────────────────────────────────────
SAVE_DETECTION_SNIPPETS = False         # overridden by --save-snippets CLI flag
DETECTION_SNIPPET_DIR   = "detections"  # directory for WAV + PNG files
SNIPPET_PRE_MS          = 500.0         # audio captured before call onset
SNIPPET_POST_MS         = 500.0         # audio captured after call offset
RING_BUFFER_SECONDS     = 20.0          # raw audio ring buffer depth

# ── Playback masking (Phase 3) ────────────────────────────────────────────────
# Acoustic propagation: 6 ft / 343 m·s⁻¹ ≈ 5.3 ms speaker-to-mic delay.
PROPAGATION_DELAY_S = 0.00534
# Playback energy above this threshold (dB) masks the corresponding mic bin.
MASK_FLOOR_DB = -40.0

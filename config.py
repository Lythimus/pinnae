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
DETECTION_THRESHOLD_DB = 20.0   # dB above rolling noise floor to count as active
MIN_CALL_DURATION_MS   = 5.0    # discard transients shorter than this
CALL_END_SILENCE_MS    = 15.0   # consecutive below-threshold time to close a call
NOISE_HISTORY_CHUNKS   = 50     # rolling window length (~2.1 s at 42.7 ms/chunk)

# ── Playback masking (Phase 3) ────────────────────────────────────────────────
# Acoustic propagation: 6 ft / 343 m·s⁻¹ ≈ 5.3 ms speaker-to-mic delay.
PROPAGATION_DELAY_S = 0.00534
# Playback energy above this threshold (dB) masks the corresponding mic bin.
MASK_FLOOR_DB = -40.0

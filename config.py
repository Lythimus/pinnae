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
# tickling recording) — see git history for the sweep that picked them.
#
# A real mic-recorded music negative (41.5 min of an ordinary FLAC album played
# through laptop speakers — not USV-enriched, not played through the
# deployment's ultrasonic Vifa-class speakers — captured with `listen.py
# record` through the SO.104) has now been tested against these defaults via
# `evaluate.py --sweep`, and the result is a real limitation, not just a
# caveat: at MAX_CALL_BAND_FRACTION=0.9 / MIN_PEAK_CONCENTRATION_DB=6.0, recall
# on real calls is ~97% but rejection of the music negative is 0.0% — every
# false-positive chunk that crossed DETECTION_THRESHOLD_DB during the music
# also passed the shape gate. Sweeping MAX_CALL_BAND_FRACTION across 0.2-0.9
# changed nothing (real-call and false-positive candidate chunks both sit well
# under even the tightest tested value, so this feature doesn't discriminate
# this false-positive source at all). MIN_PEAK_CONCENTRATION_DB is the only
# lever that does anything, but the trade-off is harsh: 12.0 dB rejects only
# ~4% of the false positives while cutting recall to ~50-53%; 18.0 dB rejects
# ~55% but cuts recall to ~41-43%. No threshold in that range is a good trade
# — this is why the defaults are unchanged.
#
# Laptop speakers cannot radiate meaningful acoustic energy in 18-80 kHz (steep
# roll-off above ~15-20 kHz), so these false positives are not acoustic
# instrumental ultrasound — the source FLAC carries no USV-band content.
# Near-uniform false-positive counts across every monitored band (alarm 2452,
# social_low 2486, social_core 2468 — including social_core at 45-70 kHz,
# where a laptop speaker emits essentially nothing) point instead to a
# playback+capture-chain artifact: harmonic/intermodulation distortion and/or
# electrical/USB coupling between the laptop and the SO.104, generated
# downstream of the audible-band music rather than present in the source
# audio. Conclusion: this gate's two features distinguish narrowband calls
# from *broadband/percussive* noise (its original design target) but not from
# this chain artifact, whose false-positive chunks are themselves narrowband
# and concentrated. Whether genuine ultrasonic-radiating tonal music (played
# through the deployment's actual ultrasonic speakers) would fare any
# differently is untested — this corpus cannot answer that question. Treat
# alarm/social_low events during *this kind of* ambient playback with that in
# mind; see CLAUDE.md for the source-isolation protocol that narrows down the
# artifact's origin.
#
# Temporal continuity was the last single-source candidate and is now measured
# and REJECTED (see the CONTINUITY note below) — the surviving false positives
# are sustained held tones, which are *more* temporally stable than real
# (swept/trilled) calls, not less. No single-source spectral or temporal
# feature tried so far separates a rat whistle from this chain artifact. The
# remaining path is context, not classification: the alarm band is clean
# during Squeakorithm playback by design (33 kHz floor), and Phase 3
# spectrogram masking removes playback energy directly rather than out-
# classifying it.
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
# Measured to have no effect on real tonal music in the 0.2-0.9 range (see above).
MAX_CALL_BAND_FRACTION    = 0.9
# Reject if peak_power - median(band_power) falls below this — real calls
# concentrate energy near the peak; broadband hash does not. Kept low/permissive:
# measured concentration varies a lot with recording context (a busy multi-rat
# cage recording has a noisier band median than a curated clean stimulus WAV),
# so this alone is not a reliable cross-context discriminator. Against a real
# music negative it is the only lever with any effect, but the recall/rejection
# trade-off is poor throughout the tested range (see above) — raising it is not
# a free win.
MIN_PEAK_CONCENTRATION_DB = 6.0

# ── Sub-harmonic gate (music-artifact discriminator, additive to the shape gate above) ──
# Rat USVs are near-pure whistle tones — essentially no energy at f_peak/2,
# f_peak/3. The music-playback false positives, by contrast, carry real energy
# at those sub-multiples — consistent with harmonic/intermodulation distortion
# products of the loud audible-band music generated downstream in the
# playback+capture chain (see the artifact framing above), not with content
# encoded in the source FLAC. Per-*chunk* subharmonic_ratio_db (strongest
# sub-harmonic power minus the chunk's own peak power), measured via
# evaluate.py against the real corpora (rat tickling + Olszynski/Polowy
# playback WAVs vs. the 41.5-min music_neg), showed a large median gap for
# alarm (call -46.8 dB vs. false-positive +4.0 dB) and social_low (call -27.7
# dB vs. false-positive -0.3 dB) — social_core did NOT separate (call p90 was
# +42.6 dB: complex 50 kHz modulated/trill calls often carry real energy at
# their own sub-multiples), so this gate is scoped to alarm/social_low only.
#
# But per-chunk percentiles overstate real recall: confirmed *events* are the
# unit that matters, and many real calls are only 1-3 chunks long (min 2 chunks
# to open, MIN_CALL_DURATION_MS=85ms), so losing even a single chunk to this
# gate can kill (or truncate below MIN_CALL_DURATION_MS) an entire call. At
# MAX_SUBHARMONIC_RATIO_DB=-1.5 (picked from the alarm/social_low ~90th-
# percentile chunk boundary above), a confirmatory evaluate.py run measured
# **event-level** recall of only ~78% (396/508 kept) against ~58% music
# rejection (2080/4938 events kept, i.e. down from the current-defaults 0%
# rejection) — well short of the ~90% recall bar this project requires before
# shipping a gate on by default. It's the best rejection lever found so far
# (better than MIN_PEAK_CONCENTRATION_DB's 55%-reject/41-43%-recall option
# above), but still not a clean win. Left available but OFF by default; a
# per-call (not per-chunk) version — e.g. requiring N consecutive failing
# chunks before rejecting, rather than a single-chunk hard cutoff — is the
# likely next step if this is revisited, since it would stop short calls from
# being this fragile to one bad chunk.
SUBHARMONIC_GATE_ENABLED = False
SUBHARMONIC_GATE_BANDS = frozenset({"alarm", "social_low"})
MAX_SUBHARMONIC_RATIO_DB = -1.5  # best-measured threshold so far; see note above

# ── Temporal-continuity discriminator (measured and REJECTED — no gate shipped) ─
# Hypothesis: a real call's peak frequency moves as a connected contour from one
# 42.7 ms chunk to the next (small |f_{i+1}-f_i| steps), while an unrelated
# broadband artifact jumping between chunks takes large steps even at similar
# total FM range. This is distinct from the FM-range feature (also rejected):
# range is how far the peak wanders overall; step is chunk-to-chunk
# connectedness. Measured via evaluate.py per confirmed event (using
# detector's active-chunk-only peak_freq_trace, which excludes trailing
# offset-silence chunks whose "peak" is just noise argmax and would inject
# spurious steps), against the real corpora vs. the 41.5-min music_neg
# (ordinary FLAC through laptop speakers — see the artifact framing above).
# The decision criterion was pos-p90 < neg-p50 with daylight (real calls
# clustered at small steps, the artifact at large). The clean
# median_step_hz(>=3) result is the OPPOSITE — pos p90 sits 3-4x *above* neg p50
# in every band, i.e. real calls step MORE than the false positives:
#   alarm       pos p50/p90 = 1640/3830 Hz   neg p50/p90 = 1050/3150 Hz
#   social_low  pos p50/p90 = 1980/7460 Hz   neg p50/p90 = 2330/6580 Hz
#   social_core pos p50/p90 = 3890/11000 Hz  neg p50/p90 = 2840/8650 Hz
# The reversal is diagnostic: the false positives that survive to become
# confirmed events are sustained *held* tones (flat → small steps) —
# consistent with a steady chain-distortion artifact riding on sustained
# musical passages, not with a wandering broadband transient — while real
# social calls sweep and trill (→ large steps). Continuity therefore points
# the wrong way. It also can't apply to most alarm calls at all — only 107 of
# 263 confirmed alarm events have >=3 chunks (59% are 1-2 chunks, too short to
# have a meaningful step contour), the same short-call fragility that sank the
# sub-harmonic gate. No code ships beyond the harness measurement in
# evaluate.py (which reads detector's diagnostic peak_freq_trace, itself
# runtime-inert).

# ── Offender-bin masking (electronic interference) ───────────────────────────
# Distinct problem from the music-artifact investigation above: steady
# electronic tones are already absorbed by the per-bin rolling floor (see
# BandDetector docstring), but intermittent narrowband transients (e.g.
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

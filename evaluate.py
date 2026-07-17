#!/usr/bin/env python3
"""
evaluate.py — offline replay harness for the call-shape gate (see detector.py).

Replays labeled WAV corpora through the *real* runtime pipeline
(compute_power_spectrum + BandDetector) so gate thresholds can be measured and
tuned against real positive (call) and negative (noise) audio without live
playback.

Requires scipy (eval-only; not a runtime dependency of the detector itself) for
sample-rate resampling: `pip install scipy`.

Examples:
  python evaluate.py --pos tickling.wav --pos olszynski_wav_dir --neg music_neg.wav --gate off
  python evaluate.py --pos tickling.wav --neg music_neg.wav --sweep
  python evaluate.py --neg music_neg.wav --artifact-check
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

import config
from detector import BandDetector, classify_valence, compute_power_spectrum

try:
    from scipy.signal import resample_poly
except ImportError:
    resample_poly = None

# Mirrors listen.py's BASELINE_BANDS. Kept independent so this module has no
# dependency on sounddevice/opencv (listen.py's runtime imports).
DEFAULT_BANDS = ["alarm", "social_low"]

AUDIO_EXTENSIONS = {".wav", ".flac"}


def find_audio_files(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(
            f for f in p.rglob("*")
            if f.suffix.lower() in AUDIO_EXTENSIONS and not f.name.startswith("._")
        )
    raise FileNotFoundError(path)


def load_and_resample(path: Path, target_sr: int) -> np.ndarray:
    samples, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if sr == target_sr:
        return mono
    if resample_poly is None:
        raise RuntimeError(
            f"{path} is {sr} Hz, not {target_sr} Hz, and scipy is not installed "
            "to resample it. Run: pip install scipy"
        )
    g = math.gcd(sr, target_sr)
    up, down = target_sr // g, sr // g
    return resample_poly(mono, up, down).astype(np.float32)


def iter_chunks(samples: np.ndarray, chunk_samples: int):
    n = len(samples)
    for start in range(0, n - chunk_samples + 1, chunk_samples):
        yield samples[start : start + chunk_samples]


def replay(samples: np.ndarray, band_names: list[str]) -> tuple[dict[str, list], list[dict], list[dict]]:
    """Run one file through the real detection pipeline.

    Returns (events_by_band, diagnostic_records, fm_records).

    diagnostic_records has one entry per (chunk, band) where the raw peak
    excess alone would have crossed DETECTION_THRESHOLD_DB, independent of
    the shape gate — used to inspect the per-chunk spectral-shape and
    sub-harmonic features that separate real calls from noise.

    fm_records has one entry per *confirmed* event (same onset/offset
    debounce as the live pipeline, gate-independent) with the peak-frequency
    range/std across that event's chunks — used to measure whether FM
    modulation depth (real calls sweep/trill, held music notes don't)
    separates calls from music.
    """
    detectors = {name: BandDetector(name, *config.BANDS[name]) for name in band_names}
    events: dict[str, list] = {name: [] for name in band_names}
    records: list[dict] = []
    chunk_dt = config.CHUNK_SAMPLES / config.SAMPLE_RATE
    t = 0.0
    idx = 0
    for chunk in iter_chunks(samples, config.CHUNK_SAMPLES):
        ps = compute_power_spectrum(chunk)
        for name, det in detectors.items():
            events[name].extend(det.process(ps, t, idx))
            if det.last_peak_excess >= config.DETECTION_THRESHOLD_DB:
                records.append({"band": name, "peak_excess_db": det.last_peak_excess, **det.last_shape_features})
        t += chunk_dt
        idx += 1
    for name, det in detectors.items():
        events[name].extend(det.flush(t, idx))

    fm_records: list[dict] = []
    for name, evs in events.items():
        for event in evs:
            # Use the detector's own active-chunk peak-freq contour (excludes the
            # trailing offset-silence chunks, whose "peak" is just noise argmax).
            segment = list(event.peak_freq_trace)
            if len(segment) < 2:
                continue
            # Step-to-step trajectory (temporal-continuity candidate). Unlike
            # freq_range/std (total spread — already rejected: trills and
            # note-jumping music both spread wide), this measures how far the
            # peak moves *between consecutive chunks*: a real call is a connected
            # contour (small steps even when it trills across a wide range),
            # while music tripping the threshold across chunks jumps between
            # unrelated notes/transients (large steps).
            steps = np.abs(np.diff(np.asarray(segment, dtype=float)))
            if steps.size == 0:
                steps = np.array([0.0])
            fm_records.append({
                "band": name,
                "freq_range_hz": max(segment) - min(segment),
                "freq_std_hz": float(np.std(segment)),
                "max_step_hz": float(steps.max()),
                "median_step_hz": float(np.median(steps)),
                "n_chunks": len(segment),
            })
    return events, records, fm_records


def percentiles(values: list[float], pcts=(10, 50, 90)) -> str:
    if not values:
        return "n=0"
    arr = np.array(values)
    parts = [f"p{p}={np.percentile(arr, p):.3g}" for p in pcts]
    return f"n={len(arr):<5d} " + "  ".join(parts)


def print_feature_summary(records: list[dict], label: str) -> None:
    if not records:
        print(f"  [{label}] no candidate chunks (nothing crossed {config.DETECTION_THRESHOLD_DB} dB)")
        return
    bands = sorted({r["band"] for r in records})
    for band in bands:
        band_records = [r for r in records if r["band"] == band]
        occ = [r["occupied_fraction"] for r in band_records]
        run = [r["max_run_fraction"] for r in band_records]
        conc = [r["peak_concentration_db"] for r in band_records]
        sub = [r["subharmonic_ratio_db"] for r in band_records]
        print(f"  [{label}] {band:<12} occupied_fraction  {percentiles(occ)}")
        print(f"  [{label}] {band:<12} max_run_fraction   {percentiles(run)}")
        print(f"  [{label}] {band:<12} peak_concentration {percentiles(conc)}")
        print(f"  [{label}] {band:<12} subharmonic_ratio  {percentiles(sub)}")


def print_fm_summary(fm_records: list[dict], label: str) -> None:
    """Peak-frequency modulation depth across confirmed events (candidate A2 feature).
    Real calls in the social register sweep/trill; a held music note doesn't."""
    if not fm_records:
        print(f"  [{label}] no confirmed events to measure FM modulation on")
        return
    bands = sorted({r["band"] for r in fm_records})
    for band in bands:
        band_records = [r for r in fm_records if r["band"] == band]
        rng = [r["freq_range_hz"] for r in band_records]
        std = [r["freq_std_hz"] for r in band_records]
        print(f"  [{label}] {band:<12} freq_range_hz      {percentiles(rng)}")
        print(f"  [{label}] {band:<12} freq_std_hz        {percentiles(std)}")
        # Temporal-continuity candidate: chunk-to-chunk peak-freq step. Only
        # events with >=3 chunks have >=2 steps worth judging; a continuity gate
        # must pass shorter calls unconditionally (never reject on absence of
        # data — the lesson from the sub-harmonic gate's recall collapse).
        multi = [r for r in band_records if r["n_chunks"] >= 3]
        mx = [r["max_step_hz"] for r in multi]
        md = [r["median_step_hz"] for r in multi]
        print(f"  [{label}] {band:<12} max_step_hz(>=3)   {percentiles(mx)}")
        print(f"  [{label}] {band:<12} median_step_hz(>=3){percentiles(md)}")


# Reference passband for the audible-vs-ultrasonic artifact diagnostics below.
# Laptop speakers roll off steeply above ~15-20 kHz, so 1-15 kHz is where a
# laptop-speaker source's real acoustic energy (and any downstream distortion
# fundamental) actually lives.
AUDIBLE_REF_BAND = (1_000, 15_000)


def artifact_features(samples: np.ndarray, band_names: list[str]) -> dict:
    """
    Offline diagnostic characterizing *what* the in-band false-positive energy
    in a file is, using only the full power spectrum (no detector/floor state).
    Answers three questions about a candidate chain artifact (e.g. laptop-
    speaker playback picked up by the mic through a non-acoustic path):

    - band-uniformity: mean peak power per monitored band. A flat level across
      bands that a laptop speaker cannot physically reach (e.g. social_core,
      45-70 kHz) is inconsistent with acoustic content and consistent with a
      broadband artifact.
    - audible-ultrasonic correlation: per-chunk Pearson correlation between the
      loudest audible-band (1-15 kHz, inside a laptop speaker's passband) peak
      power and each monitored band's peak power. High correlation means the
      ultrasonic energy tracks the loudness of the audible music, i.e. is a
      byproduct of it rather than independent content.
    - harmonic-comb match: among each band's loudest 10% of chunks (a fixed-
      percentile stand-in for "candidate false positive", since this function
      has no rolling noise floor of its own), the fraction whose in-band peak
      frequency sits within 2% of an integer multiple of that chunk's audible-
      band peak frequency — the signature of harmonic/intermodulation
      distortion, where downstream nonlinearity generates energy at N * f0.
    """
    freqs = np.fft.rfftfreq(config.CHUNK_SAMPLES, d=1.0 / config.SAMPLE_RATE)
    audible_mask = (freqs >= AUDIBLE_REF_BAND[0]) & (freqs <= AUDIBLE_REF_BAND[1])
    audible_freqs = freqs[audible_mask]

    band_masks = {name: (freqs >= config.BANDS[name][0]) & (freqs <= config.BANDS[name][1]) for name in band_names}
    band_freqs = {name: freqs[band_masks[name]] for name in band_names}

    audible_peak_db: list[float] = []
    audible_peak_freq: list[float] = []
    band_peak_db: dict[str, list[float]] = {name: [] for name in band_names}
    band_peak_freq: dict[str, list[float]] = {name: [] for name in band_names}

    for chunk in iter_chunks(samples, config.CHUNK_SAMPLES):
        ps = compute_power_spectrum(chunk)
        a_slice = ps[audible_mask]
        a_idx = int(np.argmax(a_slice))
        audible_peak_db.append(float(a_slice[a_idx]))
        audible_peak_freq.append(float(audible_freqs[a_idx]))
        for name in band_names:
            b_slice = ps[band_masks[name]]
            b_idx = int(np.argmax(b_slice))
            band_peak_db[name].append(float(b_slice[b_idx]))
            band_peak_freq[name].append(float(band_freqs[name][b_idx]))

    audible_arr = np.array(audible_peak_db)
    audible_freq_arr = np.array(audible_peak_freq)

    result: dict = {"n_chunks": len(audible_peak_db), "audible_peak_db": percentiles(audible_peak_db)}
    for name in band_names:
        b_arr = np.array(band_peak_db[name])
        corr = float(np.corrcoef(audible_arr, b_arr)[0, 1]) if len(b_arr) > 1 else float("nan")

        threshold = np.percentile(b_arr, 90)
        candidate_idx = np.where(b_arr >= threshold)[0]
        matches = 0
        for i in candidate_idx:
            f0 = audible_freq_arr[i]
            f_peak = band_peak_freq[name][i]
            if f0 <= 0:
                continue
            n = round(f_peak / f0)
            if n < 1:
                continue
            nearest_harmonic = n * f0
            if nearest_harmonic > 0 and abs(f_peak - nearest_harmonic) / nearest_harmonic <= 0.02:
                matches += 1
        comb_fraction = matches / len(candidate_idx) if len(candidate_idx) else float("nan")

        result[name] = {
            "mean_peak_db": float(np.mean(b_arr)),
            "audible_correlation": corr,
            "harmonic_comb_fraction": comb_fraction,
            "n_candidate_chunks": int(len(candidate_idx)),
        }
    return result


def print_artifact_summary(files: list[tuple[Path, np.ndarray]], band_names: list[str], label: str) -> None:
    for path, samples in files:
        features = artifact_features(samples, band_names)
        print(f"\n[{label}] {path.name} ({features['n_chunks']} chunks, audible-band peak {features['audible_peak_db']})")
        for name in band_names:
            b = features[name]
            print(
                f"  [{label}] {name:<12} mean_peak_db={b['mean_peak_db']:.1f}  "
                f"audible_correlation={b['audible_correlation']:.3f}  "
                f"harmonic_comb_fraction={b['harmonic_comb_fraction']:.2%} "
                f"(n_candidate={b['n_candidate_chunks']})"
            )


def load_corpus(paths: list[str], target_sr: int) -> list[tuple[Path, np.ndarray]]:
    loaded = []
    for path in paths:
        for f in find_audio_files(path):
            print(f"[evaluate] loading {f} ...", file=sys.stderr)
            loaded.append((f, load_and_resample(f, target_sr)))
    return loaded


def report_counts(files: list[tuple[Path, np.ndarray]], band_names: list[str], label: str) -> dict:
    """Run replay on each file, print per-file/per-band counts, return {path: {band: n_events}}."""
    result = {}
    all_records = []
    all_fm_records = []
    for path, samples in files:
        events, records, fm_records = replay(samples, band_names)
        all_records.extend(records)
        all_fm_records.extend(fm_records)
        duration_min = len(samples) / config.SAMPLE_RATE / 60.0
        result[path] = {name: len(evs) for name, evs in events.items()}
        counts_str = "  ".join(
            f"{name}={len(evs)} ({len(evs)/duration_min:.2f}/min)" for name, evs in events.items()
        )
        print(f"[{label}] {path.name:<50} {counts_str}")
    return result, all_records, all_fm_records


def run_valence_check(files: list[tuple[Path, np.ndarray]], band_names: list[str]) -> None:
    """For each file, replay it and tally classify_valence() labels across all detected
    events. Filenames of curated exemplar corpora (e.g. the Olszynski/Polowy playback
    WAVs) encode the expected call category, so the printed counts can be compared by
    eye against the filename to sanity-check the heuristic against real labeled calls."""
    for path, samples in files:
        events, _, _ = replay(samples, band_names)
        counts: dict[str, int] = {}
        for evs in events.values():
            for event in evs:
                label = classify_valence(event)
                counts[label] = counts.get(label, 0) + 1
        counts_str = ", ".join(f"{label}={n}" for label, n in sorted(counts.items())) or "no events"
        print(f"[valence] {path.name:<50} {counts_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pos", action="append", default=[], metavar="PATH",
                        help="Positive (real-call) WAV file or directory. Repeatable.")
    parser.add_argument("--neg", action="append", default=[], metavar="PATH",
                        help="Negative (noise) WAV file or directory. Repeatable.")
    parser.add_argument("--bands", default=",".join(DEFAULT_BANDS), metavar="LIST",
                        help=f"Comma-separated band names (default: {','.join(DEFAULT_BANDS)})")
    parser.add_argument("--gate", choices=["on", "off"], default="on",
                        help="Call-shape gate state for the single-run report (default: on)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep gate thresholds and report kept/rejected fractions instead of a single run")
    parser.add_argument("--valence-check", action="store_true",
                        help="Report classify_valence() label counts per file instead of a detection-count report")
    parser.add_argument("--artifact-check", action="store_true",
                        help="Report band-uniformity, audible-ultrasonic correlation, and harmonic-comb "
                             "diagnostics per file instead of a detection-count report — characterizes "
                             "whether a negative source's in-band energy looks like a playback/capture-"
                             "chain artifact rather than acoustic content")
    parser.add_argument("--sweep-fractions", default="0.2,0.3,0.4,0.5,0.6,0.7", metavar="LIST",
                        help="Comma-separated MAX_CALL_BAND_FRACTION values to sweep")
    parser.add_argument("--sweep-concentrations", default="3,6,9,12,15", metavar="LIST",
                        help="Comma-separated MIN_PEAK_CONCENTRATION_DB values to sweep")
    args = parser.parse_args()

    if not args.pos and not args.neg:
        parser.error("at least one of --pos or --neg is required")

    band_names = args.bands.split(",")

    pos_files = load_corpus(args.pos, config.SAMPLE_RATE)
    neg_files = load_corpus(args.neg, config.SAMPLE_RATE)

    if args.valence_check:
        config.CALL_SHAPE_GATE_ENABLED = args.gate == "on"
        print(f"\n=== valence check (gate={args.gate}, bands={','.join(band_names)}) ===")
        run_valence_check(pos_files, band_names)
        run_valence_check(neg_files, band_names)
        return

    if args.artifact_check:
        print(f"\n=== artifact check (bands={','.join(band_names)}) ===")
        if pos_files:
            print_artifact_summary(pos_files, band_names, "pos")
        if neg_files:
            print_artifact_summary(neg_files, band_names, "neg")
        return

    if not args.sweep:
        config.CALL_SHAPE_GATE_ENABLED = args.gate == "on"
        print(f"\n=== gate={args.gate} ===")
        print("\n-- positive files --")
        _, pos_records, pos_fm_records = report_counts(pos_files, band_names, "pos")
        print("\n-- negative files --")
        _, neg_records, neg_fm_records = report_counts(neg_files, band_names, "neg")

        if args.gate == "off":
            print("\n-- shape-feature distributions (candidate chunks, gate off) --")
            print_feature_summary(pos_records, "pos")
            print_feature_summary(neg_records, "neg")
            print("\n-- FM modulation distributions (confirmed events, gate off) --")
            print_fm_summary(pos_fm_records, "pos")
            print_fm_summary(neg_fm_records, "neg")
        return

    # --sweep: baseline with gate off (recall/rejection denominator), then try each combo.
    config.CALL_SHAPE_GATE_ENABLED = False
    print("\n=== baseline (gate off) ===")
    pos_baseline, pos_records, pos_fm_records = report_counts(pos_files, band_names, "pos")
    neg_baseline, neg_records, neg_fm_records = report_counts(neg_files, band_names, "neg")

    print("\n-- shape-feature distributions (candidate chunks, gate off) --")
    print_feature_summary(pos_records, "pos")
    print_feature_summary(neg_records, "neg")
    print("\n-- FM modulation distributions (confirmed events, gate off) --")
    print_fm_summary(pos_fm_records, "pos")
    print_fm_summary(neg_fm_records, "neg")

    fractions = [float(x) for x in args.sweep_fractions.split(",")]
    concentrations = [float(x) for x in args.sweep_concentrations.split(",")]

    pos_baseline_total = sum(sum(counts.values()) for counts in pos_baseline.values()) or 1
    neg_baseline_total = sum(sum(counts.values()) for counts in neg_baseline.values()) or 1

    print("\n=== sweep (gate on) ===")
    print(f"{'max_run_fraction':>18}  {'min_peak_concentration_db':>26}  {'pos_kept':>10}  {'neg_rejected':>12}")
    config.CALL_SHAPE_GATE_ENABLED = True
    for frac in fractions:
        for conc in concentrations:
            config.MAX_CALL_BAND_FRACTION = frac
            config.MIN_PEAK_CONCENTRATION_DB = conc
            pos_on, _, _ = report_counts_quiet(pos_files, band_names)
            neg_on, _, _ = report_counts_quiet(neg_files, band_names)
            pos_kept = pos_on / pos_baseline_total
            neg_rejected = 1.0 - (neg_on / neg_baseline_total)
            print(f"{frac:>18.2f}  {conc:>26.1f}  {pos_kept:>10.2%}  {neg_rejected:>12.2%}")


def report_counts_quiet(files: list[tuple[Path, np.ndarray]], band_names: list[str]) -> tuple[int, list, list]:
    total = 0
    all_records = []
    all_fm_records = []
    for _path, samples in files:
        events, records, fm_records = replay(samples, band_names)
        all_records.extend(records)
        all_fm_records.extend(fm_records)
        total += sum(len(evs) for evs in events.values())
    return total, all_records, all_fm_records


if __name__ == "__main__":
    main()

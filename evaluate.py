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
from detector import BandDetector, compute_power_spectrum

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
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS)
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


def replay(samples: np.ndarray, band_names: list[str]) -> tuple[dict[str, list], list[dict]]:
    """Run one file through the real detection pipeline.

    Returns (events_by_band, diagnostic_records). diagnostic_records has one
    entry per (chunk, band) where the raw peak excess alone would have crossed
    DETECTION_THRESHOLD_DB, independent of the shape gate — used to inspect
    the feature distributions that separate real calls from noise.
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
    return events, records


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
        print(f"  [{label}] {band:<12} occupied_fraction  {percentiles(occ)}")
        print(f"  [{label}] {band:<12} max_run_fraction   {percentiles(run)}")
        print(f"  [{label}] {band:<12} peak_concentration {percentiles(conc)}")


def print_freq_span_summary(events_by_band: dict[str, list], label: str) -> None:
    """Report freq_span_hz percentiles for closed events — the signal MIN_FREQ_MODULATION_HZ
    is tuned from (see detector.py DetectionEvent, config.py FM gate)."""
    for band, evs in events_by_band.items():
        spans = [e.freq_span_hz for e in evs]
        if not spans:
            print(f"  [{label}] {band:<12} freq_span_hz       n=0")
            continue
        print(f"  [{label}] {band:<12} freq_span_hz       {percentiles(spans)}")


def load_corpus(paths: list[str], target_sr: int) -> list[tuple[Path, np.ndarray]]:
    loaded = []
    for path in paths:
        for f in find_audio_files(path):
            print(f"[evaluate] loading {f} ...", file=sys.stderr)
            loaded.append((f, load_and_resample(f, target_sr)))
    return loaded


def report_counts(files: list[tuple[Path, np.ndarray]], band_names: list[str], label: str) -> dict:
    """Run replay on each file, print per-file/per-band counts.

    Returns ({path: {band: n_events}}, diagnostic_records, {band: [DetectionEvent, ...]}).
    """
    result = {}
    all_records = []
    all_events: dict[str, list] = {name: [] for name in band_names}
    for path, samples in files:
        events, records = replay(samples, band_names)
        all_records.extend(records)
        for name, evs in events.items():
            all_events[name].extend(evs)
        duration_min = len(samples) / config.SAMPLE_RATE / 60.0
        result[path] = {name: len(evs) for name, evs in events.items()}
        counts_str = "  ".join(
            f"{name}={len(evs)} ({len(evs)/duration_min:.2f}/min)" for name, evs in events.items()
        )
        print(f"[{label}] {path.name:<50} {counts_str}")
    return result, all_records, all_events


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
    parser.add_argument("--sweep-fractions", default="0.2,0.3,0.4,0.5,0.6,0.7", metavar="LIST",
                        help="Comma-separated MAX_CALL_BAND_FRACTION values to sweep")
    parser.add_argument("--sweep-concentrations", default="3,6,9,12,15", metavar="LIST",
                        help="Comma-separated MIN_PEAK_CONCENTRATION_DB values to sweep")
    parser.add_argument("--sweep-fm", action="store_true",
                        help="Sweep MIN_FREQ_MODULATION_HZ x notch on/off instead of the shape-gate sweep")
    parser.add_argument("--sweep-fm-thresholds", default="250,500,1000,2000,3000,5000", metavar="LIST",
                        help="Comma-separated MIN_FREQ_MODULATION_HZ values to sweep")
    args = parser.parse_args()

    if not args.pos and not args.neg:
        parser.error("at least one of --pos or --neg is required")

    band_names = args.bands.split(",")

    pos_files = load_corpus(args.pos, config.SAMPLE_RATE)
    neg_files = load_corpus(args.neg, config.SAMPLE_RATE)

    if args.sweep_fm:
        sweep_fm(pos_files, neg_files, band_names, args)
        return

    if not args.sweep:
        config.CALL_SHAPE_GATE_ENABLED = args.gate == "on"
        print(f"\n=== gate={args.gate} ===")
        print("\n-- positive files --")
        _, pos_records, pos_events = report_counts(pos_files, band_names, "pos")
        print("\n-- negative files --")
        _, neg_records, neg_events = report_counts(neg_files, band_names, "neg")

        if args.gate == "off":
            print("\n-- shape-feature distributions (candidate chunks, gate off) --")
            print_feature_summary(pos_records, "pos")
            print_feature_summary(neg_records, "neg")

        print("\n-- freq_span_hz distributions (closed events) --")
        print_freq_span_summary(pos_events, "pos")
        print_freq_span_summary(neg_events, "neg")
        return

    # --sweep: baseline with gate off (recall/rejection denominator), then try each combo.
    config.CALL_SHAPE_GATE_ENABLED = False
    print("\n=== baseline (gate off) ===")
    pos_baseline, pos_records, _ = report_counts(pos_files, band_names, "pos")
    neg_baseline, neg_records, _ = report_counts(neg_files, band_names, "neg")

    print("\n-- shape-feature distributions (candidate chunks, gate off) --")
    print_feature_summary(pos_records, "pos")
    print_feature_summary(neg_records, "neg")

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
            pos_on, _ = report_counts_quiet(pos_files, band_names)
            neg_on, _ = report_counts_quiet(neg_files, band_names)
            pos_kept = pos_on / pos_baseline_total
            neg_rejected = 1.0 - (neg_on / neg_baseline_total)
            print(f"{frac:>18.2f}  {conc:>26.1f}  {pos_kept:>10.2%}  {neg_rejected:>12.2%}")


def sweep_fm(
    pos_files: list[tuple[Path, np.ndarray]],
    neg_files: list[tuple[Path, np.ndarray]],
    band_names: list[str],
    args: argparse.Namespace,
) -> None:
    """Sweep MIN_FREQ_MODULATION_HZ x notch on/off, reporting recall/rejection
    relative to a gate-off/notch-off baseline. Mirrors the --sweep shape-gate
    flow. Detectors are constructed fresh per file (see replay()), so mutating
    config here between iterations is safe."""
    config.FM_GATE_ENABLED = False
    config.NOISE_NOTCH_ENABLED = False
    print("\n=== FM/notch baseline (FM gate off, notch off) ===")
    pos_baseline, pos_records, pos_events = report_counts(pos_files, band_names, "pos")
    neg_baseline, neg_records, neg_events = report_counts(neg_files, band_names, "neg")

    print("\n-- freq_span_hz distributions (closed events, gates off) --")
    print_freq_span_summary(pos_events, "pos")
    print_freq_span_summary(neg_events, "neg")

    pos_baseline_total = sum(sum(counts.values()) for counts in pos_baseline.values()) or 1
    neg_baseline_total = sum(sum(counts.values()) for counts in neg_baseline.values()) or 1

    thresholds = [float(x) for x in args.sweep_fm_thresholds.split(",")]

    print("\n=== sweep (FM gate on) ===")
    print(f"{'notch':>8}  {'min_freq_mod_hz':>16}  {'pos_kept':>10}  {'neg_rejected':>12}")
    config.FM_GATE_ENABLED = True
    for notch_enabled in (False, True):
        config.NOISE_NOTCH_ENABLED = notch_enabled
        for thr in thresholds:
            config.MIN_FREQ_MODULATION_HZ = thr
            pos_on, _ = report_counts_quiet(pos_files, band_names)
            neg_on, _ = report_counts_quiet(neg_files, band_names)
            pos_kept = pos_on / pos_baseline_total
            neg_rejected = 1.0 - (neg_on / neg_baseline_total)
            print(f"{str(notch_enabled):>8}  {thr:>16.1f}  {pos_kept:>10.2%}  {neg_rejected:>12.2%}")


def report_counts_quiet(files: list[tuple[Path, np.ndarray]], band_names: list[str]) -> tuple[int, list]:
    total = 0
    all_records = []
    for _path, samples in files:
        events, records = replay(samples, band_names)
        all_records.extend(records)
        total += sum(len(evs) for evs in events.values())
    return total, all_records


if __name__ == "__main__":
    main()

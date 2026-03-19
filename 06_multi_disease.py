"""
06_multi_disease.py — Download & analyze additional PhysioNet cardiac databases
True North Research | Cardiac Torus Pipeline

Extends the torus analysis to multiple cardiac conditions:

  DATABASE                          | CODE  | SUBJECTS | CONDITION
  ----------------------------------|-------|----------|---------------------------
  MIT-BIH Normal Sinus Rhythm       | nsrdb | 18       | Healthy controls
  MIT-BIH Atrial Fibrillation       | afdb  | 25       | Paroxysmal AF
  BIDMC Congestive Heart Failure     | chfdb | 15       | Severe CHF (NYHA 3-4)
  MIT-BIH Supraventricular Arrhyth. | svdb  | 78       | Supraventricular arrhythmias
  Long-Term AF Database             | ltafdb| 84       | Paroxysmal/sustained AF
  CHF RR Interval Database          | chf2db| 29       | CHF (NYHA 1-3) [RR only]
  MIT-BIH Arrhythmia (already done) | mitdb | 48       | Mixed arrhythmias

For each database, we:
  1. Download records from PhysioNet
  2. Extract RR intervals (from beat annotations or signal)
  3. Map to torus and compute curvature
  4. Compute per-record Gini, burst stats, quadrant distribution
  5. Assign disease label
  6. Merge into unified analysis

The question: can torus geometry alone distinguish these conditions?
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from config import (RESULTS_DIR, FIGURES_DIR, RR_MIN_MS, RR_MAX_MS,
                     BURST_PERCENTILE, BURST_MIN_LENGTH, BURST_MERGE_GAP)

try:
    import wfdb
except ImportError:
    print("Install wfdb: pip install wfdb")
    sys.exit(1)

# =====================================================================
# DATABASE DEFINITIONS
# =====================================================================

DATABASES = {
    'nsrdb': {
        'name': 'MIT-BIH Normal Sinus Rhythm',
        'physionet_id': 'nsrdb',
        'condition': 'Normal',
        'condition_short': 'NSR',
        'fs': 128,
        'ann_ext': 'atr',
        'description': '18 healthy subjects, long-term ECG (~24h each)',
        'max_beats_per_record': 50000,  # cap to keep manageable
    },
    'afdb': {
        'name': 'MIT-BIH Atrial Fibrillation',
        'physionet_id': 'afdb',
        'condition': 'Atrial Fibrillation',
        'condition_short': 'AF',
        'fs': 250,
        'ann_ext': 'qrs',
        'description': '25 subjects with paroxysmal AF, 10h recordings',
        'max_beats_per_record': 50000,
    },
    'chfdb': {
        'name': 'BIDMC Congestive Heart Failure',
        'physionet_id': 'chfdb',
        'condition': 'Congestive Heart Failure',
        'condition_short': 'CHF',
        'fs': 250,
        'ann_ext': 'ecg',
        'description': '15 subjects, severe CHF (NYHA 3-4), ~20h each',
        'max_beats_per_record': 50000,
    },
    'svdb': {
        'name': 'MIT-BIH Supraventricular Arrhythmia',
        'physionet_id': 'svdb',
        'condition': 'Supraventricular Arrhythmia',
        'condition_short': 'SVA',
        'fs': 128,
        'ann_ext': 'atr',
        'description': '78 half-hour recordings, supraventricular arrhythmias',
        'max_beats_per_record': 50000,
    },
}

DATA_BASE_DIR = Path(__file__).parent / "data"


# =====================================================================
# DOWNLOAD
# =====================================================================

def download_database(db_key: str, db_info: dict) -> list[str]:
    """Download a PhysioNet database and return list of record names."""
    db_dir = DATA_BASE_DIR / db_key
    db_dir.mkdir(parents=True, exist_ok=True)

    physio_id = db_info['physionet_id']
    print(f"\n  Downloading {db_info['name']} ({physio_id})...")

    try:
        # Get record list from PhysioNet
        record_list = wfdb.get_record_list(physio_id)
        print(f"    Found {len(record_list)} records")
    except Exception as e:
        print(f"    ERROR getting record list: {e}")
        return []

    downloaded = []
    for i, rec in enumerate(record_list):
        try:
            wfdb.dl_database(
                physio_id,
                dl_dir=str(db_dir),
                records=[rec],
                overwrite=False
            )
            downloaded.append(rec)
            if (i + 1) % 10 == 0 or i == len(record_list) - 1:
                print(f"    [{i+1}/{len(record_list)}] downloaded")
        except Exception as e:
            print(f"    [SKIP] {rec}: {e}")

    print(f"    Completed: {len(downloaded)}/{len(record_list)} records")
    return downloaded


# =====================================================================
# RR EXTRACTION (handles different annotation formats)
# =====================================================================

def extract_rr_from_record(db_dir: Path, record_name: str, db_info: dict,
                           max_beats: int = 50000) -> dict | None:
    """
    Extract RR intervals from a single record.
    
    Different databases use different annotation formats:
    - .atr: full beat annotations with symbols (like MIT-BIH Arrhythmia)
    - .qrs: QRS detection annotations (beat locations, minimal symbols)
    - .ecg: unaudited beat annotations
    
    We only need beat times → RR intervals for the torus.
    """
    rec_path = str(db_dir / record_name)
    ann_ext = db_info['ann_ext']

    try:
        # Try to read annotations
        ann = wfdb.rdann(rec_path, ann_ext)
    except Exception as e:
        # Some records may not have this annotation type
        # Try alternative extensions
        for alt_ext in ['atr', 'qrs', 'ecg']:
            if alt_ext == ann_ext:
                continue
            try:
                ann = wfdb.rdann(rec_path, alt_ext)
                break
            except:
                continue
        else:
            print(f"      [SKIP] {record_name}: no readable annotations")
            return None

    # Get beat sample locations
    beat_samples = ann.sample
    fs = db_info['fs']

    # Try to read actual fs from header
    try:
        rec_header = wfdb.rdheader(rec_path)
        fs = rec_header.fs
    except:
        pass

    if len(beat_samples) < 10:
        return None

    # Cap beats
    if len(beat_samples) > max_beats:
        beat_samples = beat_samples[:max_beats]

    # Compute RR intervals (ms)
    rr_intervals = np.diff(beat_samples) / fs * 1000.0

    # Filter physiologically plausible RR intervals
    valid_mask = (rr_intervals >= RR_MIN_MS) & (rr_intervals <= RR_MAX_MS)
    
    # We need consecutive valid pairs for the torus
    # Build beat-pair features where both RR_pre and RR_post are valid
    beats = []
    for i in range(1, len(rr_intervals) - 1):
        rr_pre = rr_intervals[i - 1]
        rr_post = rr_intervals[i]

        if not (RR_MIN_MS <= rr_pre <= RR_MAX_MS):
            continue
        if not (RR_MIN_MS <= rr_post <= RR_MAX_MS):
            continue

        beats.append({
            'RR_pre_ms': round(float(rr_pre), 1),
            'RR_post_ms': round(float(rr_post), 1),
            'beat_idx': i,
        })

    if len(beats) < 10:
        return None

    return {
        'record': record_name,
        'beats': beats,
        'n_total_beats': len(beat_samples),
        'n_valid_pairs': len(beats),
        'fs': fs,
        'mean_rr': round(float(np.mean(rr_intervals[valid_mask])), 1),
        'std_rr': round(float(np.std(rr_intervals[valid_mask])), 1),
        'mean_hr': round(float(60000 / np.mean(rr_intervals[valid_mask])), 1),
    }


# =====================================================================
# TORUS MAPPING (reused from step 03, simplified for RR-only)
# =====================================================================

def to_angle(value, vmin, vmax):
    clipped = np.clip(value, vmin, vmax)
    return 2 * np.pi * (clipped - vmin) / (vmax - vmin)


def torus_geodesic_distance(t1a, t2a, t1b, t2b):
    d1 = np.abs(t1a - t1b)
    d1 = np.minimum(d1, 2 * np.pi - d1)
    d2 = np.abs(t2a - t2b)
    d2 = np.minimum(d2, 2 * np.pi - d2)
    return np.sqrt(d1**2 + d2**2)


def menger_curvature_torus(p1, p2, p3):
    a = torus_geodesic_distance(p2[0], p2[1], p3[0], p3[1])
    b = torus_geodesic_distance(p1[0], p1[1], p3[0], p3[1])
    c = torus_geodesic_distance(p1[0], p1[1], p2[0], p2[1])
    if a < 1e-10 or b < 1e-10 or c < 1e-10:
        return 0.0
    s = (a + b + c) / 2
    area_sq = s * (s - a) * (s - b) * (s - c)
    if area_sq <= 0:
        return 0.0
    return 4 * np.sqrt(area_sq) / (a * b * c)


def classify_quadrant(theta1, theta2):
    if theta1 < np.pi:
        return 'Q1' if theta2 < np.pi else 'Q2'
    else:
        return 'Q4' if theta2 < np.pi else 'Q3'


def gini_coefficient(values):
    v = np.abs(values[values > 0])
    if len(v) < 2:
        return 0.0
    v = np.sort(v)
    n = len(v)
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * v) / (n * np.sum(v))) - (n + 1) / n


def detect_bursts(kappa, threshold, min_length=2, merge_gap=2):
    above = kappa > threshold
    bursts = []
    in_burst = False
    start = 0
    for i in range(len(above)):
        if above[i] and not in_burst:
            start = i
            in_burst = True
        elif not above[i] and in_burst:
            bursts.append((start, i))
            in_burst = False
    if in_burst:
        bursts.append((start, len(above)))
    if len(bursts) > 1:
        merged = [bursts[0]]
        for b in bursts[1:]:
            if b[0] - merged[-1][1] <= merge_gap:
                merged[-1] = (merged[-1][0], b[1])
            else:
                merged.append(b)
        bursts = merged
    return [(s, e) for s, e in bursts if e - s >= min_length]


def process_record_torus(beats: list[dict], record_name: str,
                         condition: str, db_key: str) -> dict | None:
    """
    Map a record's beats onto the torus and compute all curvature features.
    Returns a record-level summary dict.
    """
    n = len(beats)
    if n < 20:
        return None

    rr_pre = np.array([b['RR_pre_ms'] for b in beats])
    rr_post = np.array([b['RR_post_ms'] for b in beats])

    # Map to torus angles
    theta1 = np.array([to_angle(rr, RR_MIN_MS, RR_MAX_MS) for rr in rr_pre])
    theta2 = np.array([to_angle(rr, RR_MIN_MS, RR_MAX_MS) for rr in rr_post])

    # Compute Menger curvature
    kappa = np.zeros(n)
    for i in range(1, n - 1):
        kappa[i] = menger_curvature_torus(
            (theta1[i-1], theta2[i-1]),
            (theta1[i], theta2[i]),
            (theta1[i+1], theta2[i+1])
        )

    valid_kappa = kappa[kappa > 0]
    if len(valid_kappa) < 10:
        return None

    # Gini
    gini = gini_coefficient(valid_kappa)

    # Curvature stats
    result = {
        'record': record_name,
        'database': db_key,
        'condition': condition,
        'n_beats': n,
        'mean_rr_ms': round(float(np.mean(rr_pre)), 1),
        'std_rr_ms': round(float(np.std(rr_pre)), 1),
        'mean_hr_bpm': round(60000 / float(np.mean(rr_pre)), 1),
        'rr_cv': round(float(np.std(rr_pre) / np.mean(rr_pre)), 4),
        'kappa_median': round(float(np.median(valid_kappa)), 4),
        'kappa_mean': round(float(np.mean(valid_kappa)), 4),
        'kappa_std': round(float(np.std(valid_kappa)), 4),
        'kappa_p95': round(float(np.percentile(valid_kappa, 95)), 4),
        'kappa_max': round(float(np.max(valid_kappa)), 4),
        'kappa_cv': round(float(np.std(valid_kappa) / np.mean(valid_kappa)), 4),
        'gini': round(gini, 4),
    }

    # Burst detection
    threshold = np.percentile(valid_kappa, BURST_PERCENTILE)
    burst_ranges = detect_bursts(kappa, threshold, BURST_MIN_LENGTH, BURST_MERGE_GAP)
    result['n_bursts'] = len(burst_ranges)

    if burst_ranges:
        burst_lengths = [e - s for s, e in burst_ranges]
        burst_peaks = [float(np.max(kappa[s:e])) for s, e in burst_ranges]
        result['burst_len_mean'] = round(np.mean(burst_lengths), 2)
        result['burst_len_std'] = round(np.std(burst_lengths), 2)
        result['burst_peak_mean'] = round(np.mean(burst_peaks), 4)

        if len(burst_ranges) > 1:
            ibis = [burst_ranges[i+1][0] - burst_ranges[i][1]
                    for i in range(len(burst_ranges)-1)]
            result['ibi_mean'] = round(np.mean(ibis), 2)
            result['ibi_std'] = round(np.std(ibis), 2)
            result['ibi_cv'] = round(float(np.std(ibis) / np.mean(ibis)), 4) if np.mean(ibis) > 0 else 0.0

    # Quadrant distribution
    quads = [classify_quadrant(t1, t2) for t1, t2 in zip(theta1, theta2)]
    quad_counts = Counter(quads)
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        result[f'quad_{q}_frac'] = round(quad_counts.get(q, 0) / n, 4)

    # Torus spread: standard deviation of angular positions
    # (measures how much of the torus the trajectory explores)
    result['theta1_std'] = round(float(np.std(theta1)), 4)
    result['theta2_std'] = round(float(np.std(theta2)), 4)
    result['torus_spread'] = round(float(np.sqrt(np.std(theta1)**2 + np.std(theta2)**2)), 4)

    # Diagonal deviation: how far beats stray from θ₁ = θ₂ line
    diag_dev = np.abs(theta1 - theta2)
    diag_dev = np.minimum(diag_dev, 2*np.pi - diag_dev)
    result['diag_dev_mean'] = round(float(np.mean(diag_dev)), 4)
    result['diag_dev_std'] = round(float(np.std(diag_dev)), 4)

    # Consecutive angle differences (velocity on torus)
    dtheta1 = np.diff(theta1)
    dtheta1 = np.where(dtheta1 > np.pi, dtheta1 - 2*np.pi, dtheta1)
    dtheta1 = np.where(dtheta1 < -np.pi, dtheta1 + 2*np.pi, dtheta1)
    dtheta2 = np.diff(theta2)
    dtheta2 = np.where(dtheta2 > np.pi, dtheta2 - 2*np.pi, dtheta2)
    dtheta2 = np.where(dtheta2 < -np.pi, dtheta2 + 2*np.pi, dtheta2)
    speed = np.sqrt(dtheta1**2 + dtheta2**2)
    result['torus_speed_mean'] = round(float(np.mean(speed)), 4)
    result['torus_speed_std'] = round(float(np.std(speed)), 4)
    result['torus_speed_cv'] = round(float(np.std(speed) / np.mean(speed)), 4) if np.mean(speed) > 0 else 0.0

    return result


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 65)
    print("Step 06: Multi-Disease Torus Analysis")
    print("True North Research | Cardiac Torus Pipeline")
    print("=" * 65)

    all_records = []
    download_summary = {}

    # ---- Phase 1: Download all databases ----
    print("\n" + "=" * 65)
    print("PHASE 1: Download databases from PhysioNet")
    print("=" * 65)

    for db_key, db_info in DATABASES.items():
        t0 = time.time()
        records = download_database(db_key, db_info)
        elapsed = time.time() - t0
        download_summary[db_key] = {
            'name': db_info['name'],
            'records_downloaded': len(records),
            'time_sec': round(elapsed, 1),
        }

    # Save download summary
    with open(RESULTS_DIR / 'download_summary.json', 'w') as f:
        json.dump(download_summary, f, indent=2)

    print("\n\nDownload summary:")
    for db_key, info in download_summary.items():
        print(f"  {info['name']:45s} {info['records_downloaded']:4d} records  "
              f"({info['time_sec']:.0f}s)")

    # ---- Phase 2: Extract RR intervals ----
    print("\n" + "=" * 65)
    print("PHASE 2: Extract RR intervals & compute torus features")
    print("=" * 65)

    for db_key, db_info in DATABASES.items():
        db_dir = DATA_BASE_DIR / db_key
        if not db_dir.exists():
            print(f"\n  [SKIP] {db_info['name']}: directory not found")
            continue

        print(f"\n  Processing {db_info['name']}...")

        # Get available records
        try:
            record_list = wfdb.get_record_list(db_info['physionet_id'],
                                                records_dir=str(db_dir))
        except:
            # Fall back: scan directory for .hea files
            hea_files = list(db_dir.glob('*.hea'))
            record_list = [f.stem for f in hea_files]

        if not record_list:
            print(f"    No records found in {db_dir}")
            continue

        processed = 0
        skipped = 0

        for rec_name in record_list:
            # Extract RR intervals
            rr_data = extract_rr_from_record(
                db_dir, rec_name, db_info,
                max_beats=db_info.get('max_beats_per_record', 50000)
            )

            if rr_data is None:
                skipped += 1
                continue

            # Compute torus features
            result = process_record_torus(
                rr_data['beats'],
                record_name=rec_name,
                condition=db_info['condition'],
                db_key=db_key,
            )

            if result is not None:
                result['mean_hr_bpm'] = rr_data['mean_hr']
                all_records.append(result)
                processed += 1

        print(f"    Processed: {processed}, Skipped: {skipped}")

    # ---- Also include MIT-BIH Arrhythmia results (already computed) ----
    print("\n  Including MIT-BIH Arrhythmia results (from step 04)...")
    mitdb_stats_path = RESULTS_DIR / 'record_curvature_stats.csv'
    if mitdb_stats_path.exists():
        df_mitdb = pd.read_csv(mitdb_stats_path)
        for _, row in df_mitdb.iterrows():
            # Classify by dominant arrhythmia content
            v_frac = row.get('frac_V', 0)
            s_frac = row.get('frac_S', 0)
            if v_frac > 0.10:
                condition = 'Ventricular Arrhythmia'
            elif s_frac > 0.10:
                condition = 'Supraventricular (MITDB)'
            else:
                condition = 'Normal (MITDB)'

            rec_result = {
                'record': str(int(row['record'])),
                'database': 'mitdb',
                'condition': condition,
                'n_beats': int(row['n_beats']),
                'kappa_median': row.get('kappa_median_A', np.nan),
                'kappa_mean': row.get('kappa_mean_A', np.nan),
                'kappa_std': row.get('kappa_std_A', np.nan),
                'kappa_p95': row.get('kappa_p95_A', np.nan),
                'kappa_max': row.get('kappa_max_A', np.nan),
                'kappa_cv': row.get('kappa_cv_A', np.nan),
                'gini': row.get('gini_A', np.nan),
                'n_bursts': row.get('n_bursts_A', np.nan),
                'quad_Q1_frac': row.get('quad_Q1_frac', np.nan),
                'quad_Q2_frac': row.get('quad_Q2_frac', np.nan),
                'quad_Q3_frac': row.get('quad_Q3_frac', np.nan),
                'quad_Q4_frac': row.get('quad_Q4_frac', np.nan),
            }
            all_records.append(rec_result)
        print(f"    Added {len(df_mitdb)} MIT-BIH Arrhythmia records")

    # ---- Phase 3: Unified analysis ----
    print("\n" + "=" * 65)
    print("PHASE 3: Unified multi-disease analysis")
    print("=" * 65)

    df = pd.DataFrame(all_records)
    df.to_csv(RESULTS_DIR / 'multi_disease_records.csv', index=False)

    print(f"\nTotal records: {len(df)}")
    print(f"\nRecords by condition:")
    for cond, group in df.groupby('condition'):
        print(f"  {cond:35s} n={len(group):4d}  "
              f"median κ={group['kappa_median'].median():.3f}  "
              f"Gini={group['gini'].median():.3f}")

    print(f"\nRecords by database:")
    for db, group in df.groupby('database'):
        print(f"  {db:10s} n={len(group):4d}")

    # ---- Statistical comparisons ----
    from scipy import stats

    conditions = df['condition'].unique()
    print(f"\n\nPairwise comparisons (median κ):")
    print("-" * 70)

    # Get groups with enough samples
    groups = {}
    for cond in conditions:
        subset = df[df['condition'] == cond]['kappa_median'].dropna()
        if len(subset) >= 5:
            groups[cond] = subset.values

    # Kruskal-Wallis
    if len(groups) >= 2:
        group_arrays = list(groups.values())
        H, p = stats.kruskal(*group_arrays)
        print(f"\n  Kruskal-Wallis H = {H:.1f}, p = {p:.2e}")
        print(f"  Groups: {list(groups.keys())}")
        print(f"  Sizes: {[len(g) for g in group_arrays]}")

    # Pairwise
    group_names = sorted(groups.keys())
    print(f"\n  {'Comparison':50s} {'U':>10s} {'p':>12s} {'r':>8s} {'med_1':>8s} {'med_2':>8s}")
    print("  " + "-" * 92)

    comparison_results = []
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            g1, g2 = groups[group_names[i]], groups[group_names[j]]
            if len(g1) < 3 or len(g2) < 3:
                continue
            U, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            r = 1 - 2*U / (len(g1) * len(g2))
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            pair = f"{group_names[i]} vs {group_names[j]}"
            print(f"  {pair:50s} {U:10.0f} {p:12.2e} {r:+8.3f} {np.median(g1):8.3f} {np.median(g2):8.3f} {sig}")

            comparison_results.append({
                'group_1': group_names[i],
                'group_2': group_names[j],
                'U': float(U),
                'p': float(p),
                'r_rankbiserial': round(float(r), 4),
                'median_1': round(float(np.median(g1)), 4),
                'median_2': round(float(np.median(g2)), 4),
                'n_1': len(g1),
                'n_2': len(g2),
            })

    # Save comparison results
    with open(RESULTS_DIR / 'multi_disease_comparisons.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)

    # ---- Feature summary table ----
    print(f"\n\n{'='*65}")
    print("DIAGNOSTIC FEATURE SUMMARY BY CONDITION")
    print(f"{'='*65}")
    print(f"{'Condition':35s} {'n':>4s} {'κ_med':>7s} {'Gini':>6s} {'Q2%':>6s} "
          f"{'Spread':>7s} {'SpeedCV':>8s} {'HR':>6s}")
    print("-" * 85)

    for cond in sorted(df['condition'].unique()):
        g = df[df['condition'] == cond]
        print(f"  {cond:33s} {len(g):4d} "
              f"{g['kappa_median'].median():7.3f} "
              f"{g['gini'].median():6.3f} "
              f"{g['quad_Q2_frac'].median()*100:5.1f}% "
              f"{g['torus_spread'].median():7.3f} " if 'torus_spread' in g.columns and not g['torus_spread'].isna().all() else f"  {cond:33s} {len(g):4d} "
              f"{g['kappa_median'].median():7.3f} "
              f"{g['gini'].median():6.3f} "
              f"{'':>6s} "
              f"{'':>7s} ",
              end='')
        if 'torus_speed_cv' in g.columns and not g['torus_speed_cv'].isna().all():
            print(f"{g['torus_speed_cv'].median():8.3f} ", end='')
        else:
            print(f"{'':>8s} ", end='')
        if 'mean_hr_bpm' in g.columns and not g['mean_hr_bpm'].isna().all():
            print(f"{g['mean_hr_bpm'].median():6.1f}")
        else:
            print()

    print(f"\nSaved: {RESULTS_DIR / 'multi_disease_records.csv'}")
    print(f"Saved: {RESULTS_DIR / 'multi_disease_comparisons.json'}")
    print(f"Saved: {RESULTS_DIR / 'download_summary.json'}")


if __name__ == '__main__':
    main()

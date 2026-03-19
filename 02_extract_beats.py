"""
02_extract_beats.py — Extract beat-by-beat features from MIT-BIH records
True North Research | Cardiac Torus Pipeline

For each annotated beat, extracts:
  - RR_pre:  interval from previous beat (ms)
  - RR_post: interval to next beat (ms)
  - R_amp:   R-peak amplitude (mV, from MLII lead)
  - R_amp_ratio: R_amp / median R_amp for that record
  - beat_type: annotation symbol (N, V, A, F, etc.)
  - aami_class: AAMI-standard grouping (N, S, V, F, Q)
  - record: source record ID
  - beat_idx: sequential beat index within record

Output: results/beat_features.csv (~110,000 rows)
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from config import (DATA_DIR, RESULTS_DIR, ALL_RECORDS, FS, AAMI_MAP,
                     RR_MIN_MS, RR_MAX_MS)

try:
    import wfdb
except ImportError:
    print("Install wfdb: pip install wfdb")
    sys.exit(1)


def extract_record_beats(record_id: str) -> list[dict]:
    """Extract beat features from a single MIT-BIH record."""
    rec_path = str(DATA_DIR / record_id)

    try:
        record = wfdb.rdrecord(rec_path)
        ann = wfdb.rdann(rec_path, 'atr')
    except Exception as e:
        print(f"  [SKIP] {record_id}: {e}")
        return []

    # Use first channel (usually MLII)
    signal = record.p_signal[:, 0]
    fs = record.fs

    # Get beat annotations (filter out non-beat annotations)
    beat_symbols = set(AAMI_MAP.keys())
    beat_mask = [s in beat_symbols for s in ann.symbol]
    beat_samples = ann.sample[beat_mask]
    beat_types = [s for s, m in zip(ann.symbol, beat_mask) if m]

    if len(beat_samples) < 3:
        print(f"  [SKIP] {record_id}: too few beats ({len(beat_samples)})")
        return []

    # Compute R-peak amplitudes
    # Use a small window around annotation to find true peak
    window = int(0.05 * fs)  # 50ms window
    r_amps = []
    for samp in beat_samples:
        lo = max(0, samp - window)
        hi = min(len(signal), samp + window + 1)
        segment = signal[lo:hi]
        r_amps.append(np.max(np.abs(segment)))

    r_amps = np.array(r_amps)
    median_amp = np.median(r_amps[r_amps > 0]) if np.any(r_amps > 0) else 1.0

    # Build beat feature list
    beats = []
    for i in range(1, len(beat_samples) - 1):
        rr_pre = (beat_samples[i] - beat_samples[i - 1]) / fs * 1000  # ms
        rr_post = (beat_samples[i + 1] - beat_samples[i]) / fs * 1000  # ms

        # Skip physiologically impossible intervals
        if not (RR_MIN_MS <= rr_pre <= RR_MAX_MS):
            continue
        if not (RR_MIN_MS <= rr_post <= RR_MAX_MS):
            continue

        amp = r_amps[i]
        amp_ratio = amp / median_amp if median_amp > 0 else 1.0

        bt = beat_types[i]
        aami = AAMI_MAP.get(bt, 'Q')

        beats.append({
            'record': record_id,
            'beat_idx': i,
            'sample': int(beat_samples[i]),
            'time_sec': beat_samples[i] / fs,
            'RR_pre_ms': round(rr_pre, 1),
            'RR_post_ms': round(rr_post, 1),
            'R_amp_mV': round(float(amp), 4),
            'R_amp_ratio': round(float(amp_ratio), 4),
            'beat_type': bt,
            'aami_class': aami,
        })

    return beats


def main():
    print("=" * 60)
    print("Step 02: Extract Beat-by-Beat Features")
    print("=" * 60)

    all_beats = []
    record_stats = {}

    for rec_id in ALL_RECORDS:
        beats = extract_record_beats(rec_id)
        if beats:
            all_beats.extend(beats)
            classes = Counter(b['aami_class'] for b in beats)
            record_stats[rec_id] = {
                'total_beats': len(beats),
                'classes': dict(classes),
            }
            print(f"  [{rec_id}] {len(beats):5d} beats  "
                  f"N={classes.get('N',0)} S={classes.get('S',0)} "
                  f"V={classes.get('V',0)} F={classes.get('F',0)} "
                  f"Q={classes.get('Q',0)}")

    # Save to CSV
    df = pd.DataFrame(all_beats)
    csv_path = RESULTS_DIR / 'beat_features.csv'
    df.to_csv(csv_path, index=False)

    # Save record stats
    stats_path = RESULTS_DIR / 'record_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(record_stats, f, indent=2)

    # Summary
    print()
    print(f"Total beats extracted: {len(df):,}")
    print(f"Records processed: {len(record_stats)}")
    print()
    print("AAMI class distribution:")
    for cls, count in df['aami_class'].value_counts().items():
        pct = 100 * count / len(df)
        print(f"  {cls}: {count:6,} ({pct:.1f}%)")

    print()
    print(f"RR interval range: {df['RR_pre_ms'].min():.0f} – {df['RR_pre_ms'].max():.0f} ms")
    print(f"R amplitude ratio range: {df['R_amp_ratio'].min():.2f} – {df['R_amp_ratio'].max():.2f}")
    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {stats_path}")


if __name__ == '__main__':
    main()

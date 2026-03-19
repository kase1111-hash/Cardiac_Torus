"""
18_fetal_torus.py — Paper V: The Fetal Dance
Cardiac Torus Pipeline

Apply the identical torus framework to fetal heart rate from CTG recordings.
Correlate with delivery outcomes (pH, Apgar scores).
Define fetal dance vocabulary.

CTU-UHB Database: 552 CTG recordings, FHR at 4 Hz, with umbilical cord pH.

Data format:
  - .dat files: binary, 2 channels (FHR in bpm, UC in relative units), 4 Hz
  - .hea files: WFDB header with clinical metadata in comments
  - Clinical fields: pH, BDecf, pCO2, BE, Apgar1, Apgar5, etc.
"""

import sys
import json
import struct
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats, signal

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, FIGURES_DIR, DPI, FIG_FORMAT

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

# =====================================================================
# TORUS FUNCTIONS (identical to Papers I-IV)
# =====================================================================
PI2 = 2 * np.pi

def to_angle(v, mn, mx):
    if mx - mn < 1e-10: return np.pi
    return PI2 * np.clip((v - mn) / (mx - mn), 0, 1)

def menger_curvature_torus(p1, p2, p3):
    def td(a, b):
        d1 = abs(a[0]-b[0]); d1 = min(d1, PI2-d1)
        d2 = abs(a[1]-b[1]); d2 = min(d2, PI2-d2)
        return np.sqrt(d1**2 + d2**2)
    a, b, c = td(p2,p3), td(p1,p3), td(p1,p2)
    if a < 1e-10 or b < 1e-10 or c < 1e-10: return 0.0
    s = (a+b+c)/2; asq = s*(s-a)*(s-b)*(s-c)
    return 4*np.sqrt(asq)/(a*b*c) if asq > 0 else 0.0

def gini_coefficient(values):
    v = np.abs(values[values > 0])
    if len(v) < 2: return 0.0
    v = np.sort(v); n = len(v)
    return (2*np.sum(np.arange(1,n+1)*v)/(n*np.sum(v))) - (n+1)/n

def compute_torus_features(values, label=""):
    """Compute full torus feature set from a 1D sequence."""
    n = len(values) - 1
    if n < 10:
        return None

    mn = np.percentile(values, 2)
    mx = np.percentile(values, 98)

    theta1 = np.array([to_angle(values[i], mn, mx) for i in range(n)])
    theta2 = np.array([to_angle(values[i+1], mn, mx) for i in range(n)])

    kappa = np.zeros(n)
    for i in range(1, n-1):
        kappa[i] = menger_curvature_torus(
            (theta1[i-1], theta2[i-1]),
            (theta1[i], theta2[i]),
            (theta1[i+1], theta2[i+1]))

    kappa = kappa[1:-1]
    kappa = kappa[kappa > 0]

    if len(kappa) < 5:
        return None

    # Quadrant fractions
    q1 = np.sum((theta1[1:-1] < np.pi) & (theta2[1:-1] < np.pi)) / (n-2)
    q2 = np.sum((theta1[1:-1] >= np.pi) & (theta2[1:-1] < np.pi)) / (n-2)
    q3 = np.sum((theta1[1:-1] >= np.pi) & (theta2[1:-1] >= np.pi)) / (n-2)
    q4 = np.sum((theta1[1:-1] < np.pi) & (theta2[1:-1] >= np.pi)) / (n-2)

    # Spread
    spread = float(np.std(theta1[1:-1]) + np.std(theta2[1:-1]))

    # Speed
    speeds = []
    for i in range(1, len(theta1)-1):
        d1 = abs(theta1[i]-theta1[i-1]); d1 = min(d1, PI2-d1)
        d2 = abs(theta2[i]-theta2[i-1]); d2 = min(d2, PI2-d2)
        speeds.append(np.sqrt(d1**2 + d2**2))
    speeds = np.array(speeds)
    speed_cv = float(np.std(speeds)/np.mean(speeds)) if np.mean(speeds) > 0 else 0

    return {
        'kappa_median': round(float(np.median(kappa)), 4),
        'kappa_mean': round(float(np.mean(kappa)), 4),
        'kappa_std': round(float(np.std(kappa)), 4),
        'kappa_p95': round(float(np.percentile(kappa, 95)), 4),
        'kappa_cv': round(float(np.std(kappa)/np.mean(kappa)), 4) if np.mean(kappa) > 0 else 0,
        'gini': round(gini_coefficient(kappa), 4),
        'torus_spread': round(spread, 4),
        'speed_cv': round(speed_cv, 4),
        'quad_Q1': round(float(q1), 4),
        'quad_Q2': round(float(q2), 4),
        'quad_Q3': round(float(q3), 4),
        'quad_Q4': round(float(q4), 4),
        'n_curvature_points': len(kappa),
    }


# =====================================================================
# CTG DATA READING
# =====================================================================

def read_ctg_record(dat_path):
    """
    Read a CTU-UHB CTG record.
    .dat files: 2 channels, 16-bit integers, 4 Hz
    Channel 1: FHR (bpm), Channel 2: UC
    Values of 0 indicate missing/invalid data.
    """
    hea_path = dat_path.with_suffix('.hea')
    if not hea_path.exists():
        return None, {}

    # Parse header for metadata
    metadata = {}
    n_samples = 0
    n_channels = 2
    fs = 4  # default

    with open(hea_path, 'r') as f:
        lines = f.readlines()

    # First line: record info
    parts = lines[0].strip().split()
    if len(parts) >= 4:
        n_channels = int(parts[1])
        fs = int(parts[2])
        n_samples = int(parts[3])

    # Comment lines contain clinical data
    for line in lines:
        line = line.strip()
        if line.startswith('#'):
            # Parse key-value pairs from comments
            content = line[1:].strip()
            if ':' in content:
                key, val = content.split(':', 1)
                metadata[key.strip()] = val.strip()
            elif ' ' in content:
                # Some fields are space-separated
                parts = content.split()
                if len(parts) == 2:
                    metadata[parts[0]] = parts[1]

    # Read binary data
    try:
        with open(dat_path, 'rb') as f:
            raw = f.read()

        # 16-bit signed integers, 2 channels interleaved
        n_total = len(raw) // 2  # number of 16-bit samples
        data = np.frombuffer(raw, dtype=np.int16)

        if n_channels == 2:
            fhr = data[0::2].astype(float)  # Channel 1: FHR
            uc = data[1::2].astype(float)   # Channel 2: UC
        else:
            fhr = data.astype(float)
            uc = np.zeros_like(fhr)

        # Replace 0 with NaN (0 = missing/invalid in this dataset)
        fhr[fhr == 0] = np.nan

        return fhr, metadata

    except Exception as e:
        return None, metadata


def extract_fhr_intervals(fhr, fs=4):
    """
    Convert FHR time series (in bpm at 4 Hz) to beat-to-beat intervals.

    FHR at 4 Hz gives instantaneous heart rate. We convert to RR intervals:
    RR_ms = 60000 / FHR_bpm

    Then we sample at beat boundaries for the torus.
    """
    # Remove NaN
    valid = ~np.isnan(fhr)
    fhr_clean = fhr[valid]

    if len(fhr_clean) < 100:
        return None, {}

    # Convert to RR intervals (ms)
    # FHR is already in bpm, so RR = 60000/FHR
    fhr_clipped = np.clip(fhr_clean, 50, 240)  # physiological range for fetus
    rr_ms = 60000.0 / fhr_clipped

    # The FHR signal at 4 Hz is heavily smoothed — each sample represents
    # an instantaneous rate estimate. For the torus, we want beat-level data.
    # Strategy: subsample at approximately one value per beat.
    # At 140 bpm, one beat ≈ 430ms ≈ 1.7 samples at 4 Hz.
    # We take every other sample as a proxy for beat-level intervals.
    # This gives ~2 samples per beat — close to beat-level resolution.

    # Better: use the FHR to estimate beat times, then compute true intervals
    # For each sample, accumulate time until one beat passes
    beat_intervals = []
    accumulated_time = 0
    sample_dt = 1.0 / fs  # 0.25 seconds per sample

    for i in range(len(fhr_clipped)):
        beat_period = 60.0 / fhr_clipped[i]  # seconds per beat at current rate
        accumulated_time += sample_dt
        if accumulated_time >= beat_period:
            beat_intervals.append(beat_period * 1000)  # convert to ms
            accumulated_time -= beat_period

    beat_intervals = np.array(beat_intervals)

    # Basic stats
    stats_dict = {
        'mean_fhr_bpm': round(float(np.mean(fhr_clipped)), 1),
        'std_fhr_bpm': round(float(np.std(fhr_clipped)), 1),
        'mean_rr_ms': round(float(np.mean(beat_intervals)), 1),
        'std_rr_ms': round(float(np.std(beat_intervals)), 1),
        'n_beats': len(beat_intervals),
        'fhr_valid_pct': round(100 * np.sum(valid) / len(fhr), 1),
        'duration_min': round(len(fhr) / fs / 60, 1),
    }

    return beat_intervals, stats_dict


def parse_clinical_metadata(metadata):
    """Extract clinical outcome variables from header metadata."""
    clinical = {}

    # pH is the primary outcome
    for key in ['pH', 'ph', 'PH']:
        if key in metadata:
            try:
                clinical['pH'] = float(metadata[key])
            except:
                pass

    # Apgar scores
    for key in ['Apgar1', 'apgar1', 'APGAR1']:
        if key in metadata:
            try:
                clinical['Apgar1'] = int(metadata[key])
            except:
                pass

    for key in ['Apgar5', 'apgar5', 'APGAR5']:
        if key in metadata:
            try:
                clinical['Apgar5'] = int(metadata[key])
            except:
                pass

    # Other outcomes
    for key in ['BDecf', 'pCO2', 'BE', 'Gravidity', 'Parity',
                'Weight', 'Sex', 'Age', 'Gest.weeks']:
        if key in metadata:
            try:
                clinical[key] = float(metadata[key])
            except:
                clinical[key] = metadata[key]

    # Delivery type
    for key in ['Delivery', 'delivery']:
        if key in metadata:
            clinical['Delivery'] = metadata[key]

    return clinical


# =====================================================================
# FETAL DANCE CLASSIFICATION
# =====================================================================

# Fetal dance prototypes (initial estimates — will be calibrated from data)
FETAL_DANCES = {
    'Flutter (Reactive)':   {'kappa': 8.0,  'gini': 0.30, 'spread': 2.5},
    'Flat (Distress)':      {'kappa': 25.0, 'gini': 0.20, 'spread': 1.0},
    'Steady (Normal)':      {'kappa': 12.0, 'gini': 0.28, 'spread': 2.0},
}

# pH-based outcome classification
def classify_ph(ph):
    if ph is None or np.isnan(ph):
        return 'Unknown'
    elif ph >= 7.25:
        return 'Normal'
    elif ph >= 7.20:
        return 'Pre-acidosis'
    elif ph >= 7.15:
        return 'Moderate acidosis'
    elif ph >= 7.05:
        return 'Severe acidosis'
    else:
        return 'Critical acidosis'


# =====================================================================
# MAIN
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default=str(Path(__file__).parent / 'data' / 'ctg'))
    parser.add_argument('--skip_download', action='store_true', default=True)
    args = parser.parse_args()

    print("=" * 65)
    print("Step 18: Fetal Heart Rate Torus Analysis")
    print("Cardiac Torus Pipeline — Paper V: The Fetal Dance")
    print("CTU-UHB Intrapartum Cardiotocography Database")
    print("=" * 65)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"\nData directory not found: {data_dir}")
        print("Download from: https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/")
        print("Extract to: {data_dir}")
        return

    # Find all .dat files
    dat_files = sorted(data_dir.glob('*.dat'))
    if not dat_files:
        # Check subdirectories
        dat_files = sorted(data_dir.rglob('*.dat'))

    print(f"\nFound {len(dat_files)} CTG recordings")

    if len(dat_files) == 0:
        print("No .dat files found. Check data directory structure.")
        return

    # =====================================================
    # PROCESS ALL RECORDINGS
    # =====================================================
    print(f"\nProcessing {len(dat_files)} recordings...")

    results = []
    errors = 0

    for dat_path in tqdm(dat_files, desc="CTG Torus"):
        record_id = dat_path.stem

        # Read FHR data
        fhr, metadata = read_ctg_record(dat_path)
        if fhr is None:
            errors += 1
            continue

        # Extract beat intervals
        intervals, interval_stats = extract_fhr_intervals(fhr)
        if intervals is None or len(intervals) < 20:
            errors += 1
            continue

        # Compute torus features
        torus = compute_torus_features(intervals)
        if torus is None:
            errors += 1
            continue

        # Parse clinical outcomes
        clinical = parse_clinical_metadata(metadata)

        # Combine
        record = {
            'record': record_id,
            **interval_stats,
            **torus,
            **clinical,
        }

        # Add pH classification
        if 'pH' in clinical:
            record['ph_class'] = classify_ph(clinical['pH'])

        results.append(record)

    print(f"\nProcessed: {len(results)}, Errors: {errors}")

    if not results:
        print("No results. Check data format.")
        return

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'fetal_torus_results.csv', index=False)

    # =====================================================
    # ANALYSIS
    # =====================================================
    print("\n" + "=" * 65)
    print("FETAL TORUS ANALYSIS")
    print("=" * 65)

    print(f"\nRecords: {len(df)}")
    print(f"Mean FHR: {df['mean_fhr_bpm'].mean():.1f} ± {df['mean_fhr_bpm'].std():.1f} bpm")
    print(f"Mean beats: {df['n_beats'].mean():.0f}")
    print(f"Duration: {df['duration_min'].mean():.1f} ± {df['duration_min'].std():.1f} min")

    # pH distribution
    if 'pH' in df.columns:
        ph_valid = df['pH'].dropna()
        print(f"\npH available: {len(ph_valid)} recordings")
        print(f"pH range: {ph_valid.min():.2f} – {ph_valid.max():.2f}")
        print(f"pH mean: {ph_valid.mean():.3f} ± {ph_valid.std():.3f}")

        if 'ph_class' in df.columns:
            print(f"\npH classes:")
            for cls in ['Normal', 'Pre-acidosis', 'Moderate acidosis',
                        'Severe acidosis', 'Critical acidosis']:
                n = len(df[df['ph_class'] == cls])
                if n > 0:
                    print(f"  {cls:25s}: {n:4d}")

    # Torus features
    print(f"\n{'Metric':25s} {'Mean':>10s} {'Std':>10s} {'Median':>10s}")
    print("-" * 60)
    for col in ['kappa_median', 'kappa_mean', 'gini', 'torus_spread',
                'speed_cv', 'quad_Q1', 'quad_Q2', 'quad_Q3', 'quad_Q4']:
        if col in df.columns:
            print(f"  {col:23s} {df[col].mean():10.4f} {df[col].std():10.4f} {df[col].median():10.4f}")

    # =====================================================
    # CORRELATION WITH pH
    # =====================================================
    if 'pH' in df.columns:
        print("\n" + "=" * 65)
        print("TORUS FEATURES vs DELIVERY pH")
        print("=" * 65)

        ph_valid = df.dropna(subset=['pH'])
        if len(ph_valid) >= 20:
            torus_cols = ['kappa_median', 'kappa_mean', 'kappa_std', 'kappa_p95',
                          'kappa_cv', 'gini', 'torus_spread', 'speed_cv',
                          'quad_Q1', 'quad_Q2', 'quad_Q3', 'quad_Q4',
                          'mean_fhr_bpm', 'std_fhr_bpm', 'mean_rr_ms', 'std_rr_ms']

            print(f"\n  {'Feature':25s} {'Spearman ρ':>12s} {'p-value':>12s} {'Sig':>5s}")
            print("  " + "-" * 60)

            sig_results = []
            for col in torus_cols:
                if col in ph_valid.columns:
                    valid = ph_valid[[col, 'pH']].dropna()
                    if len(valid) >= 20:
                        rho, p = stats.spearmanr(valid[col], valid['pH'])
                        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                        print(f"  {col:25s} {rho:+12.4f} {p:12.2e} {sig:>5s}")
                        if p < 0.05:
                            sig_results.append((col, rho, p))

            if sig_results:
                print(f"\n  {len(sig_results)} features significantly correlated with pH")
                print(f"\n  Best torus predictor of pH:")
                best = max(sig_results, key=lambda x: abs(x[1]))
                print(f"    {best[0]}: ρ = {best[1]:+.4f} (p = {best[2]:.2e})")

    # =====================================================
    # CORRELATION WITH APGAR
    # =====================================================
    for apgar_col in ['Apgar1', 'Apgar5']:
        if apgar_col in df.columns:
            apgar_valid = df.dropna(subset=[apgar_col])
            if len(apgar_valid) >= 20:
                print(f"\n  TORUS vs {apgar_col}:")
                for col in ['kappa_median', 'gini', 'torus_spread', 'speed_cv']:
                    if col in apgar_valid.columns:
                        valid = apgar_valid[[col, apgar_col]].dropna()
                        if len(valid) >= 20:
                            rho, p = stats.spearmanr(valid[col], valid[apgar_col])
                            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                            print(f"    {col:25s} ρ = {rho:+.4f} (p = {p:.2e}) {sig}")

    # =====================================================
    # pH GROUP COMPARISON
    # =====================================================
    if 'ph_class' in df.columns:
        print("\n" + "=" * 65)
        print("TORUS FEATURES BY pH CLASS")
        print("=" * 65)

        normal = df[df['ph_class'] == 'Normal']
        acidotic = df[df['pH'] < 7.20] if 'pH' in df.columns else pd.DataFrame()

        if len(normal) >= 10 and len(acidotic) >= 5:
            print(f"\n  Normal pH (≥7.25): n = {len(normal)}")
            print(f"  Acidotic (<7.20):   n = {len(acidotic)}")

            print(f"\n  {'Feature':25s} {'Normal':>10s} {'Acidotic':>10s} {'r':>8s} {'p':>12s}")
            print("  " + "-" * 70)

            for col in ['kappa_median', 'gini', 'torus_spread', 'speed_cv',
                        'mean_fhr_bpm', 'std_fhr_bpm']:
                if col in df.columns:
                    n_val = normal[col].dropna()
                    a_val = acidotic[col].dropna()
                    if len(n_val) >= 5 and len(a_val) >= 5:
                        U, p = stats.mannwhitneyu(n_val, a_val, alternative='two-sided')
                        r_val = 1 - 2*U/(len(n_val)*len(a_val))
                        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                        print(f"  {col:25s} {n_val.median():10.3f} {a_val.median():10.3f} {r_val:+8.3f} {p:12.2e} {sig}")

    # =====================================================
    # FETAL DANCE CLASSIFICATION (exploratory)
    # =====================================================
    if 'pH' in df.columns:
        print("\n" + "=" * 65)
        print("FETAL DANCE EXPLORATION")
        print("=" * 65)

        # Divide into pH tertiles
        ph_valid = df.dropna(subset=['pH'])
        if len(ph_valid) >= 30:
            tertiles = pd.qcut(ph_valid['pH'], 3, labels=['Low pH', 'Mid pH', 'High pH'])
            ph_valid = ph_valid.copy()
            ph_valid['ph_tertile'] = tertiles

            for t in ['Low pH', 'Mid pH', 'High pH']:
                sub = ph_valid[ph_valid['ph_tertile'] == t]
                print(f"\n  {t}: n={len(sub)}, pH={sub['pH'].mean():.3f}")
                print(f"    κ median: {sub['kappa_median'].median():.3f}")
                print(f"    Gini:     {sub['gini'].median():.3f}")
                print(f"    Spread:   {sub['torus_spread'].median():.3f}")
                print(f"    Speed CV: {sub['speed_cv'].median():.3f}")

    print(f"\n  All results: {RESULTS_DIR / 'fetal_torus_results.csv'}")
    print(f"\n  Done.")


if __name__ == '__main__':
    main()

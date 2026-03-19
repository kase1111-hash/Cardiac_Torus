"""
12_hrv_comparison.py — Head-to-Head: Torus vs Standard HRV
True North Research | Cardiac Torus Pipeline

THE CRITICAL TEST: Does the torus add information beyond existing
HRV metrics, or is it a fancy reformulation of SDNN?

For every record in our multi-disease + CHF replication datasets,
we compute BOTH:
  A) Standard HRV metrics: SDNN, RMSSD, pNN50, SD1, SD2, DFA α1
  B) Torus metrics: median κ, Gini, spread, speed CV, Q2 fraction

Then compare their diagnostic power (effect size r) for each
pairwise disease comparison.

POSSIBLE OUTCOMES:
  1. Torus metrics beat HRV → "Novel diagnostic signal" (best case)
  2. Torus matches HRV → "Elegant reformulation" (still publishable)
  3. HRV beats torus → "Beautiful math, no clinical advantage" (honest)
  4. Torus + HRV combined beats both alone → "Complementary" (also good)

We report ALL outcomes honestly.
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from config import (RESULTS_DIR, FIGURES_DIR, DPI, FIG_FORMAT,
                     RR_MIN_MS, RR_MAX_MS)

try:
    import wfdb
except ImportError:
    print("Install wfdb: pip install wfdb")
    sys.exit(1)

DATA_BASE_DIR = Path(__file__).parent / "data"


# =====================================================================
# STANDARD HRV METRICS
# =====================================================================

def compute_hrv_metrics(rr_intervals_ms: np.ndarray) -> dict:
    """
    Compute standard time-domain and nonlinear HRV metrics.
    
    Input: array of RR intervals in milliseconds.
    
    Returns dict with:
      SDNN: standard deviation of NN intervals (ms)
      RMSSD: root mean square of successive differences (ms)
      pNN50: percentage of successive differences > 50ms
      meanRR: mean RR interval (ms)
      meanHR: mean heart rate (bpm)
      SD1: Poincaré plot short-axis (ms)
      SD2: Poincaré plot long-axis (ms)
      SD1_SD2: ratio SD1/SD2
      DFA_alpha1: detrended fluctuation analysis short-term exponent
      CV_RR: coefficient of variation of RR intervals
    """
    rr = rr_intervals_ms.astype(float)
    
    if len(rr) < 20:
        return None
    
    # Filter valid
    valid = rr[(rr >= RR_MIN_MS) & (rr <= RR_MAX_MS)]
    if len(valid) < 20:
        return None
    
    # Time-domain
    sdnn = float(np.std(valid, ddof=1))
    mean_rr = float(np.mean(valid))
    mean_hr = 60000.0 / mean_rr
    cv_rr = sdnn / mean_rr
    
    # Successive differences
    diff_rr = np.diff(valid)
    rmssd = float(np.sqrt(np.mean(diff_rr**2)))
    pnn50 = float(100.0 * np.sum(np.abs(diff_rr) > 50) / len(diff_rr))
    
    # Poincaré plot descriptors
    # SD1 = std perpendicular to identity line = std(diff_rr) / sqrt(2)
    # SD2 = std along identity line = sqrt(2*SDNN^2 - SD1^2)
    sd1 = float(np.std(diff_rr, ddof=1) / np.sqrt(2))
    sd2_sq = 2 * sdnn**2 - sd1**2
    sd2 = float(np.sqrt(max(0, sd2_sq)))
    sd1_sd2 = sd1 / sd2 if sd2 > 0 else 0.0
    
    # DFA alpha1 (short-term scaling exponent)
    # Simplified implementation: compute over scales 4-16
    alpha1 = compute_dfa_alpha1(valid)
    
    return {
        'SDNN': round(sdnn, 2),
        'RMSSD': round(rmssd, 2),
        'pNN50': round(pnn50, 2),
        'meanRR': round(mean_rr, 1),
        'meanHR': round(mean_hr, 1),
        'SD1': round(sd1, 2),
        'SD2': round(sd2, 2),
        'SD1_SD2': round(sd1_sd2, 4),
        'CV_RR': round(cv_rr, 4),
        'DFA_alpha1': round(alpha1, 4) if alpha1 is not None else None,
    }


def compute_dfa_alpha1(rr_intervals, scales=None):
    """
    Detrended Fluctuation Analysis — short-term exponent α1.
    
    Uses scales 4 to 16 beats (the standard clinical range).
    Returns the slope of log(F(n)) vs log(n).
    """
    if scales is None:
        scales = [4, 5, 6, 7, 8, 10, 12, 16]
    
    rr = rr_intervals - np.mean(rr_intervals)
    y = np.cumsum(rr)  # integrated series
    n = len(y)
    
    flucts = []
    valid_scales = []
    
    for scale in scales:
        if scale > n // 4:
            continue
        
        n_segments = n // scale
        if n_segments < 2:
            continue
        
        rms_values = []
        for seg in range(n_segments):
            start = seg * scale
            end = start + scale
            segment = y[start:end]
            
            # Linear detrend
            x = np.arange(scale)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            residual = segment - trend
            rms_values.append(np.sqrt(np.mean(residual**2)))
        
        if rms_values:
            flucts.append(np.mean(rms_values))
            valid_scales.append(scale)
    
    if len(valid_scales) < 3:
        return None
    
    # Log-log slope
    log_scales = np.log(valid_scales)
    log_flucts = np.log(flucts)
    
    slope, _, _, _, _ = np.polyfit(log_scales, log_flucts, 1, full=True)
    if isinstance(slope, np.ndarray):
        slope = slope[0]
    
    return float(slope)


# =====================================================================
# TORUS METRICS (recomputed for consistency)
# =====================================================================

def to_angle(value, vmin, vmax):
    clipped = np.clip(value, vmin, vmax)
    return 2 * np.pi * (clipped - vmin) / (vmax - vmin)

def menger_curvature_torus(p1, p2, p3):
    def td(a, b):
        d1 = abs(a[0]-b[0]); d1 = min(d1, 2*np.pi-d1)
        d2 = abs(a[1]-b[1]); d2 = min(d2, 2*np.pi-d2)
        return np.sqrt(d1**2 + d2**2)
    a, b, c = td(p2,p3), td(p1,p3), td(p1,p2)
    if a<1e-10 or b<1e-10 or c<1e-10: return 0.0
    s = (a+b+c)/2; asq = s*(s-a)*(s-b)*(s-c)
    if asq <= 0: return 0.0
    return 4*np.sqrt(asq)/(a*b*c)

def gini_coefficient(values):
    v = np.abs(values[values > 0])
    if len(v) < 2: return 0.0
    v = np.sort(v); n = len(v); idx = np.arange(1, n+1)
    return (2*np.sum(idx*v)/(n*np.sum(v))) - (n+1)/n


def compute_torus_metrics(rr_intervals_ms):
    """Compute torus features from raw RR intervals."""
    rr = rr_intervals_ms[(rr_intervals_ms >= RR_MIN_MS) & (rr_intervals_ms <= RR_MAX_MS)]
    
    if len(rr) < 20:
        return None
    
    # Build consecutive pairs
    rr_pre = rr[:-1]
    rr_post = rr[1:]
    
    # Filter pairs where both are valid
    mask = (rr_pre >= RR_MIN_MS) & (rr_pre <= RR_MAX_MS) & \
           (rr_post >= RR_MIN_MS) & (rr_post <= RR_MAX_MS)
    rr_pre = rr_pre[mask]
    rr_post = rr_post[mask]
    n = len(rr_pre)
    
    if n < 20:
        return None
    
    theta1 = np.array([to_angle(r, RR_MIN_MS, RR_MAX_MS) for r in rr_pre])
    theta2 = np.array([to_angle(r, RR_MIN_MS, RR_MAX_MS) for r in rr_post])
    
    kappa = np.zeros(n)
    for i in range(1, n-1):
        kappa[i] = menger_curvature_torus(
            (theta1[i-1], theta2[i-1]),
            (theta1[i], theta2[i]),
            (theta1[i+1], theta2[i+1]))
    
    valid_kappa = kappa[kappa > 0]
    if len(valid_kappa) < 10:
        return None
    
    gini = gini_coefficient(valid_kappa)
    spread = np.sqrt(np.std(theta1)**2 + np.std(theta2)**2)
    
    # Speed
    dt1 = np.diff(theta1)
    dt1 = np.where(dt1 > np.pi, dt1 - 2*np.pi, dt1)
    dt1 = np.where(dt1 < -np.pi, dt1 + 2*np.pi, dt1)
    dt2 = np.diff(theta2)
    dt2 = np.where(dt2 > np.pi, dt2 - 2*np.pi, dt2)
    dt2 = np.where(dt2 < -np.pi, dt2 + 2*np.pi, dt2)
    speed = np.sqrt(dt1**2 + dt2**2)
    speed_cv = float(np.std(speed) / np.mean(speed)) if np.mean(speed) > 0 else 0
    
    # Quadrant
    n_q2 = np.sum((theta1 < np.pi) & (theta2 >= np.pi))
    
    return {
        'kappa_median': round(float(np.median(valid_kappa)), 4),
        'kappa_mean': round(float(np.mean(valid_kappa)), 4),
        'gini_kappa': round(gini, 4),
        'torus_spread': round(float(spread), 4),
        'torus_speed_cv': round(speed_cv, 4),
        'Q2_frac': round(float(n_q2 / n), 4),
    }


# =====================================================================
# DATA LOADING
# =====================================================================

def load_rr_from_database(db_key, db_info, max_beats=50000):
    """Load RR intervals from a PhysioNet database."""
    db_dir = DATA_BASE_DIR / db_key
    if not db_dir.exists():
        return []
    
    results = []
    
    try:
        record_list = wfdb.get_record_list(db_info['physionet_id'],
                                            records_dir=str(db_dir))
    except:
        hea_files = list(db_dir.glob('*.hea'))
        record_list = [f.stem for f in hea_files]
    
    for rec_name in record_list:
        rec_path = str(db_dir / rec_name)
        
        ann = None
        for ext in [db_info.get('ann_ext', 'atr'), 'ecg', 'atr', 'qrs']:
            try:
                ann = wfdb.rdann(rec_path, ext)
                break
            except:
                continue
        
        if ann is None:
            continue
        
        fs = 128.0
        try:
            hdr = wfdb.rdheader(rec_path)
            fs = hdr.fs
        except:
            pass
        
        beat_samples = ann.sample
        if len(beat_samples) > max_beats:
            beat_samples = beat_samples[:max_beats]
        
        rr = np.diff(beat_samples) / fs * 1000.0
        
        if len(rr) < 20:
            continue
        
        results.append({
            'record': rec_name,
            'database': db_key,
            'condition': db_info['condition'],
            'rr_intervals': rr,
        })
    
    return results


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("Step 12: Head-to-Head — Torus vs Standard HRV")
    print("=" * 70)

    DATABASES = {
        'nsrdb': {'physionet_id': 'nsrdb', 'condition': 'Normal (NSR1)', 'ann_ext': 'atr'},
        'nsr2db': {'physionet_id': 'nsr2db', 'condition': 'Normal (NSR2)', 'ann_ext': 'ecg'},
        'chfdb': {'physionet_id': 'chfdb', 'condition': 'CHF (NYHA 3-4)', 'ann_ext': 'ecg'},
        'chf2db': {'physionet_id': 'chf2db', 'condition': 'CHF (NYHA 1-3)', 'ann_ext': 'ecg'},
        'afdb': {'physionet_id': 'afdb', 'condition': 'Atrial Fibrillation', 'ann_ext': 'qrs'},
        'svdb': {'physionet_id': 'svdb', 'condition': 'SVA', 'ann_ext': 'atr'},
    }

    # Load all records
    print("\nLoading RR intervals from all databases...")
    all_records = []
    for db_key, db_info in DATABASES.items():
        records = load_rr_from_database(db_key, db_info)
        all_records.extend(records)
        print(f"  {db_key}: {len(records)} records")

    print(f"\nTotal records: {len(all_records)}")

    # Compute both HRV and torus metrics for every record
    print("\nComputing HRV + torus metrics...")
    all_features = []
    
    for rec in all_records:
        rr = rec['rr_intervals']
        
        hrv = compute_hrv_metrics(rr)
        torus = compute_torus_metrics(rr)
        
        if hrv is None or torus is None:
            continue
        
        row = {
            'record': rec['record'],
            'database': rec['database'],
            'condition': rec['condition'],
        }
        row.update(hrv)
        row.update(torus)
        all_features.append(row)

    df = pd.DataFrame(all_features)
    df.to_csv(RESULTS_DIR / 'hrv_vs_torus_features.csv', index=False)
    print(f"Records with complete features: {len(df)}")

    # ================================================================
    # COMPARISON: Effect sizes for each metric × each disease pair
    # ================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC POWER COMPARISON")
    print("=" * 70)

    hrv_metrics = ['SDNN', 'RMSSD', 'pNN50', 'SD1', 'SD2', 'SD1_SD2', 'CV_RR', 'DFA_alpha1']
    torus_metrics = ['kappa_median', 'gini_kappa', 'torus_spread', 'torus_speed_cv', 'Q2_frac']
    all_metrics = hrv_metrics + torus_metrics

    # Key disease pairs to test
    pairs = [
        ('CHF (NYHA 3-4)', 'Normal (NSR1)'),
        ('CHF (NYHA 1-3)', 'Normal (NSR2)'),
        ('Atrial Fibrillation', 'Normal (NSR1)'),
        ('CHF (NYHA 3-4)', 'CHF (NYHA 1-3)'),
    ]

    comparison_results = []

    for cond1, cond2 in pairs:
        g1_full = df[df['condition'] == cond1]
        g2_full = df[df['condition'] == cond2]
        
        if len(g1_full) < 3 or len(g2_full) < 3:
            continue

        print(f"\n  {cond1} vs {cond2} (n={len(g1_full)} vs n={len(g2_full)})")
        print(f"  {'Metric':20s} {'|r|':>8s} {'p':>12s} {'med_1':>10s} {'med_2':>10s} {'Type':>8s}")
        print("  " + "-" * 65)
        
        pair_results = []
        
        for metric in all_metrics:
            g1 = g1_full[metric].dropna()
            g2 = g2_full[metric].dropna()
            
            if len(g1) < 3 or len(g2) < 3:
                continue
            
            U, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            r = 1 - 2*U/(len(g1)*len(g2))
            
            mtype = "TORUS" if metric in torus_metrics else "HRV"
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            
            print(f"  {metric:20s} {abs(r):8.3f} {p:12.2e} {np.median(g1):10.3f} {np.median(g2):10.3f} {mtype:>8s} {sig}")
            
            pair_results.append({
                'pair': f"{cond1} vs {cond2}",
                'metric': metric,
                'type': mtype,
                'abs_r': round(abs(float(r)), 4),
                'r': round(float(r), 4),
                'p': float(p),
                'median_1': round(float(np.median(g1)), 4),
                'median_2': round(float(np.median(g2)), 4),
            })
        
        # Rank metrics by |r|
        pair_results.sort(key=lambda x: x['abs_r'], reverse=True)
        
        print(f"\n  TOP 5 METRICS:")
        for i, pr in enumerate(pair_results[:5]):
            marker = " <<<" if pr['type'] == 'TORUS' else ""
            print(f"    {i+1}. {pr['metric']:20s} |r|={pr['abs_r']:.3f} ({pr['type']}){marker}")
        
        # Count wins
        top3_types = [pr['type'] for pr in pair_results[:3]]
        torus_in_top3 = top3_types.count('TORUS')
        hrv_in_top3 = top3_types.count('HRV')
        
        print(f"  Score: Torus {torus_in_top3}/3 in top-3, HRV {hrv_in_top3}/3")
        
        comparison_results.extend(pair_results)

    # Save full results
    df_comp = pd.DataFrame(comparison_results)
    df_comp.to_csv(RESULTS_DIR / 'hrv_vs_torus_comparison.csv', index=False)

    # ================================================================
    # AGGREGATE SCORECARD
    # ================================================================
    print("\n" + "=" * 70)
    print("AGGREGATE SCORECARD")
    print("=" * 70)
    
    # For each metric, average |r| across all disease pairs
    metric_scores = {}
    for metric in all_metrics:
        scores = df_comp[df_comp['metric'] == metric]['abs_r']
        if len(scores) > 0:
            metric_scores[metric] = {
                'mean_abs_r': round(float(scores.mean()), 4),
                'max_abs_r': round(float(scores.max()), 4),
                'type': 'TORUS' if metric in torus_metrics else 'HRV',
            }
    
    # Sort by mean |r|
    sorted_metrics = sorted(metric_scores.items(), key=lambda x: x[1]['mean_abs_r'], reverse=True)
    
    print(f"\n  {'Rank':>4s} {'Metric':20s} {'Mean |r|':>10s} {'Max |r|':>10s} {'Type':>8s}")
    print("  " + "-" * 56)
    for rank, (metric, info) in enumerate(sorted_metrics, 1):
        marker = " ***" if info['type'] == 'TORUS' else ""
        print(f"  {rank:4d} {metric:20s} {info['mean_abs_r']:10.3f} {info['max_abs_r']:10.3f} {info['type']:>8s}{marker}")
    
    # Overall: which type wins more often?
    torus_wins = sum(1 for m, i in sorted_metrics[:5] if i['type'] == 'TORUS')
    hrv_wins = sum(1 for m, i in sorted_metrics[:5] if i['type'] == 'HRV')
    
    print(f"\n  Top-5 breakdown: Torus {torus_wins}, HRV {hrv_wins}")

    # ================================================================
    # VERDICT
    # ================================================================
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    
    best_torus = max([(m, i) for m, i in sorted_metrics if i['type'] == 'TORUS'],
                     key=lambda x: x[1]['mean_abs_r'])
    best_hrv = max([(m, i) for m, i in sorted_metrics if i['type'] == 'HRV'],
                   key=lambda x: x[1]['mean_abs_r'])
    
    print(f"\n  Best torus metric: {best_torus[0]} (mean |r| = {best_torus[1]['mean_abs_r']:.3f})")
    print(f"  Best HRV metric:   {best_hrv[0]} (mean |r| = {best_hrv[1]['mean_abs_r']:.3f})")
    
    diff = best_torus[1]['mean_abs_r'] - best_hrv[1]['mean_abs_r']
    
    if diff > 0.05:
        print(f"\n  TORUS WINS by {diff:.3f} → Novel diagnostic signal")
    elif diff > -0.05:
        print(f"\n  COMPARABLE (diff = {diff:+.3f}) → Complementary information")
    else:
        print(f"\n  HRV WINS by {-diff:.3f} → Torus is elegant but not superior")
    
    # ================================================================
    # FIGURE
    # ================================================================
    print("\nGenerating figure...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Metric ranking bar chart
    ax = axes[0]
    names = [m for m, _ in sorted_metrics]
    values = [i['mean_abs_r'] for _, i in sorted_metrics]
    colors = ['#F44336' if i['type'] == 'TORUS' else '#2196F3' for _, i in sorted_metrics]
    
    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Mean |effect size r| across disease pairs')
    ax.set_title('A. Diagnostic power ranking\n(red = torus, blue = HRV)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Panel B: Scatter — best torus vs best HRV per disease pair
    ax = axes[1]
    pair_names = df_comp['pair'].unique()
    
    for pair in pair_names:
        pair_data = df_comp[df_comp['pair'] == pair]
        
        best_t = pair_data[pair_data['type'] == 'TORUS']['abs_r'].max()
        best_h = pair_data[pair_data['type'] == 'HRV']['abs_r'].max()
        
        if pd.notna(best_t) and pd.notna(best_h):
            short = pair.replace('(NYHA 3-4)', '3-4').replace('(NYHA 1-3)', '1-3')
            short = short.replace('Normal ', 'N').replace('Atrial Fibrillation', 'AF')
            short = short.replace('CHF ', 'CHF')
            ax.scatter(best_h, best_t, s=100, zorder=5,
                      edgecolors='black', linewidths=0.5)
            ax.annotate(short, (best_h, best_t), fontsize=7,
                       xytext=(5, 5), textcoords='offset points')
    
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='Equal performance')
    ax.set_xlabel('Best HRV |r|')
    ax.set_ylabel('Best Torus |r|')
    ax.set_title('B. Best metric per disease pair\n(above diagonal = torus wins)')
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    
    fig.suptitle('Figure H1: Head-to-Head — Torus vs Standard HRV Metrics',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    
    path = FIGURES_DIR / f'figH1_hrv_vs_torus.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    
    print(f"\nSaved: {RESULTS_DIR / 'hrv_vs_torus_features.csv'}")
    print(f"Saved: {RESULTS_DIR / 'hrv_vs_torus_comparison.csv'}")


if __name__ == '__main__':
    main()

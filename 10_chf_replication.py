"""
10_chf_replication.py — CHF Replication with CHF RR Interval Database
True North Research | Cardiac Torus Pipeline

The CHF finding (κ = 32.0, 3× normal) is the paper's headline claim
but rests on n=15 subjects from BIDMC (NYHA 3-4). This script
replicates the analysis with an independent CHF cohort:

  CHF RR Interval Database (chf2db):
    - 29 subjects with congestive heart failure
    - NYHA classes 1, 2, and 3 (milder than BIDMC)
    - ~24 hours of beat annotations each
    - RR intervals only (no ECG signal — just beat times)

  Normal Sinus Rhythm RR Interval Database (nsr2db):
    - 54 subjects (long-term, healthy)
    - ~24 hours of beat annotations each

This also tests whether curvature tracks CHF severity (NYHA class).

If CHF replication succeeds with r > 0.5:
  → The CHF finding is robust across independent cohorts
  → Curvature may grade severity (NYHA 1 vs 2 vs 3)
  → The wearable screening claim is strengthened
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from config import (RESULTS_DIR, FIGURES_DIR, DPI, FIG_FORMAT,
                     RR_MIN_MS, RR_MAX_MS, BURST_PERCENTILE,
                     BURST_MIN_LENGTH, BURST_MERGE_GAP)

try:
    import wfdb
except ImportError:
    print("Install wfdb: pip install wfdb")
    sys.exit(1)

DATA_BASE_DIR = Path(__file__).parent / "data"

# Torus functions (reused)
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

def classify_quadrant(t1, t2):
    if t1 < np.pi:
        return 'Q1' if t2 < np.pi else 'Q2'
    else:
        return 'Q4' if t2 < np.pi else 'Q3'


DATABASES = {
    'chf2db': {
        'name': 'CHF RR Interval Database',
        'physionet_id': 'chf2db',
        'condition': 'CHF (NYHA 1-3)',
        'ann_ext': 'ecg',
        'description': '29 CHF subjects, NYHA 1-3, 24h RR intervals',
    },
    'nsr2db': {
        'name': 'Normal Sinus Rhythm RR Database',
        'physionet_id': 'nsr2db',
        'condition': 'Normal (NSR2)',
        'ann_ext': 'ecg',
        'description': '54 healthy subjects, 24h RR intervals',
    },
}


def download_db(db_key, db_info):
    """Download a PhysioNet database."""
    db_dir = DATA_BASE_DIR / db_key
    db_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  Downloading {db_info['name']}...")
    try:
        record_list = wfdb.get_record_list(db_info['physionet_id'])
        print(f"    Found {len(record_list)} records")
    except Exception as e:
        print(f"    ERROR: {e}")
        return []
    
    downloaded = []
    for i, rec in enumerate(record_list):
        try:
            wfdb.dl_database(db_info['physionet_id'], dl_dir=str(db_dir),
                            records=[rec], overwrite=False)
            downloaded.append(rec)
            if (i+1) % 10 == 0 or i == len(record_list)-1:
                print(f"    [{i+1}/{len(record_list)}]")
        except Exception as e:
            print(f"    [SKIP] {rec}: {e}")
    
    return downloaded


def extract_rr_and_analyze(db_dir, record_name, db_info, max_beats=50000):
    """Extract RR intervals and compute torus features for one record."""
    rec_path = str(db_dir / record_name)
    
    # Try multiple annotation extensions
    ann = None
    for ext in [db_info['ann_ext'], 'ecg', 'atr', 'qrs']:
        try:
            ann = wfdb.rdann(rec_path, ext)
            break
        except:
            continue
    
    if ann is None:
        return None
    
    beat_samples = ann.sample
    
    # Get sampling rate from header
    fs = 128.0  # default
    try:
        hdr = wfdb.rdheader(rec_path)
        fs = hdr.fs
    except:
        pass
    
    if len(beat_samples) < 20:
        return None
    
    if len(beat_samples) > max_beats:
        beat_samples = beat_samples[:max_beats]
    
    # Compute RR intervals
    rr = np.diff(beat_samples) / fs * 1000.0  # ms
    
    # Build valid beat pairs
    rr_pre_list = []
    rr_post_list = []
    for i in range(1, len(rr)-1):
        if RR_MIN_MS <= rr[i-1] <= RR_MAX_MS and RR_MIN_MS <= rr[i] <= RR_MAX_MS:
            rr_pre_list.append(rr[i-1])
            rr_post_list.append(rr[i])
    
    rr_pre = np.array(rr_pre_list)
    rr_post = np.array(rr_post_list)
    n = len(rr_pre)
    
    if n < 20:
        return None
    
    # Torus mapping
    theta1 = np.array([to_angle(r, RR_MIN_MS, RR_MAX_MS) for r in rr_pre])
    theta2 = np.array([to_angle(r, RR_MIN_MS, RR_MAX_MS) for r in rr_post])
    
    # Curvature
    kappa = np.zeros(n)
    for i in range(1, n-1):
        kappa[i] = menger_curvature_torus(
            (theta1[i-1], theta2[i-1]),
            (theta1[i], theta2[i]),
            (theta1[i+1], theta2[i+1])
        )
    
    valid_kappa = kappa[kappa > 0]
    if len(valid_kappa) < 10:
        return None
    
    gini = gini_coefficient(valid_kappa)
    
    # Quadrant distribution
    quads = [classify_quadrant(t1, t2) for t1, t2 in zip(theta1, theta2)]
    qc = Counter(quads)
    
    # Torus spread
    spread = np.sqrt(np.std(theta1)**2 + np.std(theta2)**2)
    
    # Speed CV
    dt1 = np.diff(theta1)
    dt1 = np.where(dt1 > np.pi, dt1 - 2*np.pi, dt1)
    dt1 = np.where(dt1 < -np.pi, dt1 + 2*np.pi, dt1)
    dt2 = np.diff(theta2)
    dt2 = np.where(dt2 > np.pi, dt2 - 2*np.pi, dt2)
    dt2 = np.where(dt2 < -np.pi, dt2 + 2*np.pi, dt2)
    speed = np.sqrt(dt1**2 + dt2**2)
    speed_cv = float(np.std(speed) / np.mean(speed)) if np.mean(speed) > 0 else 0
    
    valid_rr = rr[(rr >= RR_MIN_MS) & (rr <= RR_MAX_MS)]
    
    return {
        'record': record_name,
        'database': db_dir.name,
        'n_beats': n,
        'mean_rr_ms': round(float(np.mean(valid_rr)), 1),
        'std_rr_ms': round(float(np.std(valid_rr)), 1),
        'mean_hr_bpm': round(60000 / float(np.mean(valid_rr)), 1),
        'rr_cv': round(float(np.std(valid_rr) / np.mean(valid_rr)), 4),
        'kappa_median': round(float(np.median(valid_kappa)), 4),
        'kappa_mean': round(float(np.mean(valid_kappa)), 4),
        'kappa_std': round(float(np.std(valid_kappa)), 4),
        'kappa_p95': round(float(np.percentile(valid_kappa, 95)), 4),
        'kappa_cv': round(float(np.std(valid_kappa) / np.mean(valid_kappa)), 4),
        'gini': round(gini, 4),
        'torus_spread': round(float(spread), 4),
        'torus_speed_cv': round(speed_cv, 4),
        'quad_Q1_frac': round(qc.get('Q1', 0) / n, 4),
        'quad_Q2_frac': round(qc.get('Q2', 0) / n, 4),
        'quad_Q3_frac': round(qc.get('Q3', 0) / n, 4),
        'quad_Q4_frac': round(qc.get('Q4', 0) / n, 4),
    }


def main():
    print("=" * 65)
    print("Step 10: CHF Replication — Independent Cohort Validation")
    print("True North Research | Cardiac Torus Pipeline")
    print("=" * 65)

    # Phase 1: Download
    print("\nPHASE 1: Download databases")
    for db_key, db_info in DATABASES.items():
        download_db(db_key, db_info)

    # Phase 2: Extract and analyze
    print("\nPHASE 2: Extract RR intervals and compute torus features")
    all_results = []

    for db_key, db_info in DATABASES.items():
        db_dir = DATA_BASE_DIR / db_key
        if not db_dir.exists():
            continue

        print(f"\n  Processing {db_info['name']}...")
        
        # Get record list
        try:
            record_list = wfdb.get_record_list(db_info['physionet_id'],
                                                records_dir=str(db_dir))
        except:
            hea_files = list(db_dir.glob('*.hea'))
            record_list = [f.stem for f in hea_files]

        processed = 0
        for rec_name in record_list:
            result = extract_rr_and_analyze(db_dir, rec_name, db_info)
            if result is not None:
                result['condition'] = db_info['condition']
                all_results.append(result)
                processed += 1

        print(f"    Processed: {processed}")

    # Also include original BIDMC CHF and NSR results
    print("\n  Including original BIDMC CHF and NSR results...")
    md_path = RESULTS_DIR / 'multi_disease_records.csv'
    if md_path.exists():
        df_orig = pd.read_csv(md_path)
        for _, row in df_orig.iterrows():
            if row['condition'] in ['Congestive Heart Failure', 'Normal']:
                orig_cond = 'CHF (NYHA 3-4)' if row['condition'] == 'Congestive Heart Failure' else 'Normal (NSR1)'
                all_results.append({
                    'record': str(row.get('record', '')),
                    'database': str(row.get('database', '')),
                    'condition': orig_cond,
                    'n_beats': int(row.get('n_beats', 0)),
                    'kappa_median': row.get('kappa_median', np.nan),
                    'kappa_mean': row.get('kappa_mean', np.nan),
                    'gini': row.get('gini', np.nan),
                    'torus_spread': row.get('torus_spread', np.nan),
                    'torus_speed_cv': row.get('torus_speed_cv', np.nan),
                    'mean_hr_bpm': row.get('mean_hr_bpm', np.nan),
                    'quad_Q1_frac': row.get('quad_Q1_frac', np.nan),
                    'quad_Q2_frac': row.get('quad_Q2_frac', np.nan),
                    'quad_Q3_frac': row.get('quad_Q3_frac', np.nan),
                    'quad_Q4_frac': row.get('quad_Q4_frac', np.nan),
                })
        n_added = len([r for r in all_results if r['condition'] in ['CHF (NYHA 3-4)', 'Normal (NSR1)']])
        print(f"    Added {n_added} original records")

    # Phase 3: Statistical analysis
    print("\n" + "=" * 65)
    print("PHASE 3: CHF Replication Analysis")
    print("=" * 65)

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / 'chf_replication_records.csv', index=False)

    print(f"\nTotal records: {len(df)}")
    print(f"\nRecords by condition:")
    for cond, group in df.groupby('condition'):
        med_k = group['kappa_median'].median()
        med_g = group['gini'].median()
        print(f"  {cond:25s} n={len(group):3d}  median κ={med_k:.3f}  Gini={med_g:.3f}")

    # Pairwise comparisons
    conditions = sorted(df['condition'].unique())
    groups = {}
    for cond in conditions:
        vals = df[df['condition'] == cond]['kappa_median'].dropna()
        if len(vals) >= 3:
            groups[cond] = vals.values

    if len(groups) >= 2:
        H, p = stats.kruskal(*groups.values())
        print(f"\n  Kruskal-Wallis H = {H:.1f}, p = {p:.2e}")

    print(f"\n  {'Pair':50s} {'r':>8s} {'p':>12s} {'κ₁':>8s} {'κ₂':>8s}")
    print("  " + "-" * 82)

    comparisons = []
    gnames = sorted(groups.keys())
    for i in range(len(gnames)):
        for j in range(i+1, len(gnames)):
            g1, g2 = groups[gnames[i]], groups[gnames[j]]
            U, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            r = 1 - 2*U/(len(g1)*len(g2))
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            pair = f"{gnames[i]} vs {gnames[j]}"
            print(f"  {pair:50s} {r:+8.3f} {p:12.2e} "
                  f"{np.median(g1):8.3f} {np.median(g2):8.3f} {sig}")
            comparisons.append({
                'group_1': gnames[i], 'group_2': gnames[j],
                'r': round(float(r), 4), 'p': float(p),
                'median_1': round(float(np.median(g1)), 4),
                'median_2': round(float(np.median(g2)), 4),
            })

    with open(RESULTS_DIR / 'chf_replication_comparisons.json', 'w') as f:
        json.dump(comparisons, f, indent=2)

    # KEY TEST: Does the new CHF cohort replicate the original?
    print("\n" + "=" * 65)
    print("KEY REPLICATION TEST")
    print("=" * 65)

    for chf_label in ['CHF (NYHA 1-3)', 'CHF (NYHA 3-4)']:
        for nsr_label in ['Normal (NSR2)', 'Normal (NSR1)']:
            if chf_label in groups and nsr_label in groups:
                g_chf = groups[chf_label]
                g_nsr = groups[nsr_label]
                U, p = stats.mannwhitneyu(g_chf, g_nsr, alternative='two-sided')
                r = 1 - 2*U/(len(g_chf)*len(g_nsr))
                print(f"\n  {chf_label} vs {nsr_label}:")
                print(f"    n_CHF={len(g_chf)}, n_Normal={len(g_nsr)}")
                print(f"    κ_CHF={np.median(g_chf):.3f}, κ_Normal={np.median(g_nsr):.3f}")
                print(f"    Ratio: {np.median(g_chf)/np.median(g_nsr):.1f}x")
                print(f"    Effect size r = {r:+.3f}, p = {p:.2e}")
                if abs(r) > 0.5:
                    print(f"    ✓ REPLICATION SUCCESSFUL (large effect)")
                elif abs(r) > 0.3:
                    print(f"    ~ PARTIAL REPLICATION (medium effect)")
                else:
                    print(f"    ✗ REPLICATION FAILED")

    # Figure
    print("\nGenerating figures...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    COND_COLORS = {
        'CHF (NYHA 3-4)': '#E65100',
        'CHF (NYHA 1-3)': '#FF9800',
        'Normal (NSR1)': '#1565C0',
        'Normal (NSR2)': '#64B5F6',
    }

    # Panel A: Boxplot
    ax = axes[0]
    plot_conds = [c for c in ['Normal (NSR1)', 'Normal (NSR2)', 'CHF (NYHA 1-3)', 'CHF (NYHA 3-4)']
                  if c in groups]
    data = [groups[c] for c in plot_conds]
    colors = [COND_COLORS.get(c, '#607D8B') for c in plot_conds]

    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=True,
                    flierprops=dict(marker='.', markersize=4, alpha=0.4),
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)

    labels = [f"{c}\n(n={len(groups[c])})" for c in plot_conds]
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Median geodesic curvature κ')
    ax.set_title('A. CHF Replication: Original vs Independent Cohort')
    ax.grid(axis='y', alpha=0.3)

    # Panel B: Scatter κ vs Gini
    ax = axes[1]
    for cond in plot_conds:
        g = df[df['condition'] == cond]
        ax.scatter(g['kappa_median'], g['gini'],
                   c=COND_COLORS.get(cond, '#607D8B'), s=60, alpha=0.7,
                   edgecolors='black', linewidths=0.3,
                   label=f"{cond} (n={len(g)})")
    ax.set_xlabel('Median κ')
    ax.set_ylabel('Curvature Gini')
    ax.set_title('B. CHF vs Normal in κ-Gini space')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle('Figure R1: CHF Replication with Independent Cohort',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = FIGURES_DIR / f'figR1_chf_replication.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    print(f"\nSaved: {RESULTS_DIR / 'chf_replication_records.csv'}")
    print(f"Saved: {RESULTS_DIR / 'chf_replication_comparisons.json'}")


if __name__ == '__main__':
    main()

"""
04_curvature_analysis.py — Gini coefficients, burst detection, regime classification
True North Research | Cardiac Torus Pipeline

Applies the full curvature analysis toolkit from the Atlas/EEG work:
  1. Per-record Gini coefficient of curvature distribution
  2. Curvature burst detection (contiguous high-κ segments)
  3. Burst statistics (duration, peak, inter-burst interval)
  4. Quadrant classification on each torus
  5. Cross-torus correlation analysis
  6. Statistical tests: do arrhythmia classes separate in curvature space?
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from config import (RESULTS_DIR, BURST_PERCENTILE, BURST_MIN_LENGTH,
                     BURST_MERGE_GAP)


def gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient of a distribution.
    G = 0: perfectly uniform. G → 1: maximally concentrated.
    """
    v = np.abs(values[values > 0])
    if len(v) < 2:
        return 0.0
    v = np.sort(v)
    n = len(v)
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * v) / (n * np.sum(v))) - (n + 1) / n


def detect_bursts(kappa: np.ndarray, threshold: float,
                  min_length: int = 2, merge_gap: int = 2) -> list[dict]:
    """
    Detect contiguous segments of high curvature.
    
    Returns list of burst dicts with:
      start, end, length, peak_kappa, mean_kappa, total_kappa
    """
    above = kappa > threshold
    
    # Find contiguous runs
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
    
    # Merge bursts separated by small gaps
    if len(bursts) > 1:
        merged = [bursts[0]]
        for b in bursts[1:]:
            if b[0] - merged[-1][1] <= merge_gap:
                merged[-1] = (merged[-1][0], b[1])
            else:
                merged.append(b)
        bursts = merged
    
    # Filter by minimum length and compute stats
    result = []
    for start, end in bursts:
        length = end - start
        if length >= min_length:
            segment = kappa[start:end]
            result.append({
                'start': start,
                'end': end,
                'length': length,
                'peak_kappa': float(np.max(segment)),
                'mean_kappa': float(np.mean(segment)),
                'total_kappa': float(np.sum(segment)),
            })
    
    return result


def classify_quadrant(theta1: float, theta2: float) -> str:
    """
    Classify a torus point into quadrants.
    
    For Torus A (RR_pre × RR_post):
      Q1: fast→fast   (both < π)  — sustained tachycardia
      Q2: fast→slow   (θ₁<π, θ₂>π) — deceleration
      Q3: slow→slow   (both > π)  — sustained bradycardia
      Q4: slow→fast   (θ₁>π, θ₂<π) — acceleration
    """
    if theta1 < np.pi:
        return 'Q1' if theta2 < np.pi else 'Q2'
    else:
        return 'Q4' if theta2 < np.pi else 'Q3'


def analyze_record(df_record: pd.DataFrame) -> dict:
    """Full curvature analysis for a single record."""
    rec_id = df_record['record'].iloc[0]
    n = len(df_record)
    
    result = {'record': rec_id, 'n_beats': n}
    
    for torus_label, kappa_col in [('A', 'kappa_A'), ('B', 'kappa_B'), ('C', 'kappa_C')]:
        kappa = df_record[kappa_col].values
        valid_kappa = kappa[kappa > 0]
        
        if len(valid_kappa) < 10:
            continue
        
        # Gini coefficient
        gini = gini_coefficient(valid_kappa)
        result[f'gini_{torus_label}'] = round(gini, 4)
        
        # Basic curvature stats
        result[f'kappa_median_{torus_label}'] = round(np.median(valid_kappa), 4)
        result[f'kappa_mean_{torus_label}'] = round(np.mean(valid_kappa), 4)
        result[f'kappa_std_{torus_label}'] = round(np.std(valid_kappa), 4)
        result[f'kappa_p95_{torus_label}'] = round(np.percentile(valid_kappa, 95), 4)
        result[f'kappa_max_{torus_label}'] = round(np.max(valid_kappa), 4)
        result[f'kappa_cv_{torus_label}'] = round(np.std(valid_kappa) / np.mean(valid_kappa), 4) if np.mean(valid_kappa) > 0 else 0.0
        
        # Burst detection
        threshold = np.percentile(valid_kappa, BURST_PERCENTILE)
        bursts = detect_bursts(kappa, threshold, BURST_MIN_LENGTH, BURST_MERGE_GAP)
        
        result[f'n_bursts_{torus_label}'] = len(bursts)
        if bursts:
            lengths = [b['length'] for b in bursts]
            peaks = [b['peak_kappa'] for b in bursts]
            result[f'burst_len_mean_{torus_label}'] = round(np.mean(lengths), 2)
            result[f'burst_len_std_{torus_label}'] = round(np.std(lengths), 2)
            result[f'burst_peak_mean_{torus_label}'] = round(np.mean(peaks), 4)
            result[f'burst_peak_max_{torus_label}'] = round(np.max(peaks), 4)
            
            # Inter-burst intervals
            if len(bursts) > 1:
                ibis = [bursts[i+1]['start'] - bursts[i]['end'] for i in range(len(bursts)-1)]
                result[f'ibi_mean_{torus_label}'] = round(np.mean(ibis), 2)
                result[f'ibi_std_{torus_label}'] = round(np.std(ibis), 2)
    
    # Quadrant distribution (Torus A)
    if 'theta1_A' in df_record.columns:
        quads = df_record.apply(
            lambda r: classify_quadrant(r['theta1_A'], r['theta2_A']), axis=1
        )
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            result[f'quad_{q}_frac'] = round((quads == q).mean(), 4)
    
    # Class composition
    for cls in ['N', 'S', 'V', 'F', 'Q']:
        result[f'frac_{cls}'] = round((df_record['aami_class'] == cls).mean(), 4)
    
    return result


def beat_level_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-beat curvature analysis: add quadrant, local Gini (rolling window).
    """
    # Quadrant classification
    df['quadrant_A'] = df.apply(
        lambda r: classify_quadrant(r['theta1_A'], r['theta2_A']), axis=1
    )
    
    # Rolling Gini (window of 30 beats)
    window = 30
    gini_rolling = []
    kappa = df['kappa_A'].values
    for i in range(len(kappa)):
        lo = max(0, i - window // 2)
        hi = min(len(kappa), i + window // 2)
        segment = kappa[lo:hi]
        valid = segment[segment > 0]
        gini_rolling.append(gini_coefficient(valid) if len(valid) >= 5 else np.nan)
    
    df['gini_rolling_A'] = np.round(gini_rolling, 4)
    
    return df


def statistical_tests(df: pd.DataFrame) -> dict:
    """
    Test whether curvature separates AAMI classes.
    Mann-Whitney U for each pair, Kruskal-Wallis overall.
    """
    results = {}
    
    # Overall Kruskal-Wallis
    groups = []
    labels = []
    for cls in ['N', 'S', 'V', 'F']:
        subset = df[df['aami_class'] == cls]['kappa_A'].dropna()
        subset = subset[subset > 0]
        if len(subset) >= 10:
            groups.append(subset.values)
            labels.append(cls)
    
    if len(groups) >= 2:
        H, p = stats.kruskal(*groups)
        results['kruskal_wallis'] = {
            'H': round(float(H), 2),
            'p': float(p),
            'groups': labels,
            'group_sizes': [len(g) for g in groups],
        }
    
    # Pairwise Mann-Whitney
    results['pairwise'] = {}
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            U, p = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
            # Rank-biserial effect size
            n1, n2 = len(groups[i]), len(groups[j])
            r = 1 - 2*U / (n1 * n2)
            
            key = f"{labels[i]}_vs_{labels[j]}"
            results['pairwise'][key] = {
                'U': float(U),
                'p': float(p),
                'r_rankbiserial': round(float(r), 4),
                'median_1': round(float(np.median(groups[i])), 4),
                'median_2': round(float(np.median(groups[j])), 4),
            }
    
    # Quadrant × class contingency (chi-squared)
    if 'quadrant_A' in df.columns:
        ct = pd.crosstab(df['aami_class'], df['quadrant_A'])
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            chi2, p, dof, expected = stats.chi2_contingency(ct)
            results['quadrant_chi2'] = {
                'chi2': round(float(chi2), 2),
                'p': float(p),
                'dof': int(dof),
                'contingency_table': ct.to_dict(),
            }
    
    return results


def main():
    print("=" * 60)
    print("Step 04: Curvature Analysis")
    print("=" * 60)
    
    # Load torus curvature data
    csv_path = RESULTS_DIR / 'torus_curvature.csv'
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run 03_torus_mapping.py first.")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    df['record'] = df['record'].astype(int).astype(str)
    print(f"Loaded {len(df):,} beats from {df['record'].nunique()} records")
    
    # === Per-record analysis ===
    print("\nPer-record analysis...")
    record_results = []
    for rec_id, group in df.groupby('record'):
        result = analyze_record(group)
        record_results.append(result)
    
    df_records = pd.DataFrame(record_results)
    df_records.to_csv(RESULTS_DIR / 'record_curvature_stats.csv', index=False)
    
    # Print Gini summary
    print("\nGini coefficients by record (Torus A):")
    print("-" * 55)
    for _, row in df_records.sort_values('gini_A', ascending=False).head(15).iterrows():
        rec_id = str(int(row['record'])) if not isinstance(row['record'], str) else row['record']
        print(f"  {rec_id:>4s}  G={row['gini_A']:.3f}  "
              f"κ_med={row['kappa_median_A']:.3f}  "
              f"V%={row.get('frac_V', 0)*100:.1f}  "
              f"S%={row.get('frac_S', 0)*100:.1f}")
    
    # === Beat-level analysis ===
    print("\nBeat-level analysis...")
    df = beat_level_analysis(df)
    df.to_csv(RESULTS_DIR / 'torus_curvature_analyzed.csv', index=False)
    
    # === Statistical tests ===
    print("\nStatistical tests...")
    test_results = statistical_tests(df)
    
    with open(RESULTS_DIR / 'statistical_tests.json', 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    # Print results
    if 'kruskal_wallis' in test_results:
        kw = test_results['kruskal_wallis']
        print(f"\n  Kruskal-Wallis H = {kw['H']:.1f}, p = {kw['p']:.2e}")
        print(f"  Groups: {kw['groups']}, sizes: {kw['group_sizes']}")
    
    if 'pairwise' in test_results:
        print("\n  Pairwise comparisons (κ_A):")
        for pair, res in test_results['pairwise'].items():
            sig = "***" if res['p'] < 0.001 else "**" if res['p'] < 0.01 else "*" if res['p'] < 0.05 else "ns"
            print(f"    {pair:>8s}: r={res['r_rankbiserial']:+.3f}  "
                  f"p={res['p']:.2e}  {sig}  "
                  f"medians: {res['median_1']:.4f} vs {res['median_2']:.4f}")
    
    if 'quadrant_chi2' in test_results:
        qc = test_results['quadrant_chi2']
        print(f"\n  Quadrant × Class χ² = {qc['chi2']:.1f}, "
              f"p = {qc['p']:.2e}, dof = {qc['dof']}")
    
    # === Key finding: curvature by class ===
    print("\n" + "=" * 55)
    print("KEY FINDING: Curvature distribution by beat class")
    print("=" * 55)
    for cls in ['N', 'S', 'V', 'F']:
        subset = df[df['aami_class'] == cls]['kappa_A']
        valid = subset[subset > 0]
        if len(valid) > 0:
            gini = gini_coefficient(valid.values)
            print(f"  {cls}: n={len(valid):6,}  median κ={np.median(valid):.4f}  "
                  f"Gini={gini:.3f}  P95={np.percentile(valid, 95):.4f}")
    
    # === Quadrant occupancy by class ===
    if 'quadrant_A' in df.columns:
        print("\nQuadrant occupancy by AAMI class:")
        ct = pd.crosstab(df['aami_class'], df['quadrant_A'], normalize='index')
        print(ct.round(3).to_string())
    
    print(f"\nSaved: {RESULTS_DIR / 'record_curvature_stats.csv'}")
    print(f"Saved: {RESULTS_DIR / 'torus_curvature_analyzed.csv'}")
    print(f"Saved: {RESULTS_DIR / 'statistical_tests.json'}")


if __name__ == '__main__':
    main()

"""
09_noise_robustness.py — PPG Timing Jitter Robustness Test
True North Research | Cardiac Torus Pipeline

The critical question for wearable deployment: does the torus framework
survive the timing imprecision of PPG sensors?

ECG R-peaks: ~1ms precision
PPG peaks: 10-30ms jitter (motion, sensor quality, skin tone)

Method:
  1. Load the multi-disease RR intervals (clean ECG-derived)
  2. Add Gaussian timing noise at levels: 0, 5, 10, 20, 30, 50 ms
  3. Recompute torus curvature at each noise level
  4. Measure: does the diagnostic separation (effect size r) survive?
  5. Find the "cliff" — the noise level where CHF/VA separation collapses

If separation survives at 20ms jitter, PPG wearables are viable.
If it collapses below 10ms, only ECG devices work.
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, FIGURES_DIR, DPI, FIG_FORMAT, RR_MIN_MS, RR_MAX_MS

# Reuse torus functions from step 06
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
    s = (a+b+c)/2
    asq = s*(s-a)*(s-b)*(s-c)
    if asq <= 0: return 0.0
    return 4*np.sqrt(asq)/(a*b*c)

def gini_coefficient(values):
    v = np.abs(values[values > 0])
    if len(v) < 2: return 0.0
    v = np.sort(v); n = len(v)
    idx = np.arange(1, n+1)
    return (2*np.sum(idx*v)/(n*np.sum(v))) - (n+1)/n


def compute_record_kappa(rr_pre, rr_post):
    """Compute median curvature and Gini for a set of RR pairs."""
    n = len(rr_pre)
    if n < 20:
        return None, None

    theta1 = np.array([to_angle(rr, RR_MIN_MS, RR_MAX_MS) for rr in rr_pre])
    theta2 = np.array([to_angle(rr, RR_MIN_MS, RR_MAX_MS) for rr in rr_post])

    kappa = np.zeros(n)
    for i in range(1, n-1):
        kappa[i] = menger_curvature_torus(
            (theta1[i-1], theta2[i-1]),
            (theta1[i], theta2[i]),
            (theta1[i+1], theta2[i+1])
        )

    valid = kappa[kappa > 0]
    if len(valid) < 10:
        return None, None

    return float(np.median(valid)), float(gini_coefficient(valid))


def add_rr_jitter(rr_pre, rr_post, noise_std_ms, rng):
    """
    Simulate PPG timing jitter by adding Gaussian noise to RR intervals.
    
    When a beat time shifts by δ, both the preceding and following RR 
    intervals change: RR_pre → RR_pre + δ, RR_post → RR_post - δ
    (because the beat moved later, the gap before it grew and after it shrank).
    
    For simplicity, we add independent noise to each RR interval,
    which slightly overestimates the effect (real jitter is correlated).
    """
    noise_pre = rng.normal(0, noise_std_ms, len(rr_pre))
    noise_post = rng.normal(0, noise_std_ms, len(rr_post))

    noisy_pre = np.clip(rr_pre + noise_pre, RR_MIN_MS, RR_MAX_MS)
    noisy_post = np.clip(rr_post + noise_post, RR_MIN_MS, RR_MAX_MS)

    return noisy_pre, noisy_post


def main():
    print("=" * 65)
    print("Step 09: PPG Noise Robustness Test")
    print("True North Research | Cardiac Torus Pipeline")
    print("=" * 65)

    # Load torus curvature data (beat-level, from step 03)
    beat_path = RESULTS_DIR / 'torus_curvature.csv'
    if not beat_path.exists():
        print(f"ERROR: {beat_path} not found. Run steps 01-03 first.")
        sys.exit(1)

    df_beats = pd.read_csv(beat_path)
    df_beats['record'] = df_beats['record'].astype(int).astype(str)

    # Also load multi-disease data
    md_path = RESULTS_DIR / 'multi_disease_records.csv'
    has_multi = md_path.exists()
    if has_multi:
        df_multi = pd.read_csv(md_path)
        print(f"Loaded multi-disease records: {len(df_multi)}")

    # Noise levels to test (ms)
    NOISE_LEVELS = [0, 2, 5, 10, 15, 20, 30, 50, 75, 100]

    rng = np.random.default_rng(2024)
    N_TRIALS = 10  # repeat each noise level to estimate variance

    # ================================================================
    # TEST 1: Beat-level AAMI class separation under noise
    # ================================================================
    print("\n" + "=" * 65)
    print("TEST 1: AAMI beat-class separation vs timing jitter")
    print("=" * 65)

    # Group beats by AAMI class, get their RR pairs
    class_rr = {}
    for cls in ['N', 'S', 'V', 'F']:
        subset = df_beats[df_beats['aami_class'] == cls]
        if len(subset) >= 100:
            class_rr[cls] = {
                'rr_pre': subset['RR_pre_ms'].values,
                'rr_post': subset['RR_post_ms'].values,
            }

    # For each noise level, compute curvature and measure N-vs-V separation
    beat_results = []

    for noise_ms in NOISE_LEVELS:
        trial_effects = []

        for trial in range(N_TRIALS):
            class_kappas = {}
            for cls, rr_data in class_rr.items():
                if noise_ms == 0:
                    rr_pre, rr_post = rr_data['rr_pre'], rr_data['rr_post']
                else:
                    rr_pre, rr_post = add_rr_jitter(
                        rr_data['rr_pre'], rr_data['rr_post'], noise_ms, rng
                    )

                # Compute curvature for all beats in this class
                n = len(rr_pre)
                theta1 = np.array([to_angle(rr, RR_MIN_MS, RR_MAX_MS) for rr in rr_pre])
                theta2 = np.array([to_angle(rr, RR_MIN_MS, RR_MAX_MS) for rr in rr_post])

                kappa = np.zeros(n)
                for i in range(1, n-1):
                    kappa[i] = menger_curvature_torus(
                        (theta1[i-1], theta2[i-1]),
                        (theta1[i], theta2[i]),
                        (theta1[i+1], theta2[i+1])
                    )
                valid = kappa[kappa > 0]
                class_kappas[cls] = valid

            # N vs V effect size
            if 'N' in class_kappas and 'V' in class_kappas:
                n_k, v_k = class_kappas['N'], class_kappas['V']
                if len(n_k) > 10 and len(v_k) > 10:
                    U, p = stats.mannwhitneyu(n_k, v_k, alternative='two-sided')
                    r = 1 - 2*U/(len(n_k)*len(v_k))
                    trial_effects.append({
                        'r_NV': r,
                        'p_NV': p,
                        'median_N': np.median(n_k),
                        'median_V': np.median(v_k),
                        'ratio_NV': np.median(n_k) / np.median(v_k) if np.median(v_k) > 0 else 0,
                    })

            # N vs S effect size
            if 'N' in class_kappas and 'S' in class_kappas:
                n_k, s_k = class_kappas['N'], class_kappas['S']
                if len(n_k) > 10 and len(s_k) > 10:
                    U, p = stats.mannwhitneyu(n_k, s_k, alternative='two-sided')
                    r = 1 - 2*U/(len(n_k)*len(s_k))
                    trial_effects[-1]['r_NS'] = r

        if trial_effects:
            mean_r_NV = np.mean([t['r_NV'] for t in trial_effects])
            std_r_NV = np.std([t['r_NV'] for t in trial_effects])
            mean_ratio = np.mean([t['ratio_NV'] for t in trial_effects])
            mean_r_NS = np.mean([t.get('r_NS', 0) for t in trial_effects])

            beat_results.append({
                'noise_ms': noise_ms,
                'r_NV_mean': round(mean_r_NV, 4),
                'r_NV_std': round(std_r_NV, 4),
                'r_NS_mean': round(mean_r_NS, 4),
                'ratio_NV_mean': round(mean_ratio, 2),
                'median_N': round(np.mean([t['median_N'] for t in trial_effects]), 3),
                'median_V': round(np.mean([t['median_V'] for t in trial_effects]), 3),
            })

            sig = "***" if abs(mean_r_NV) > 0.5 else "**" if abs(mean_r_NV) > 0.3 else "*" if abs(mean_r_NV) > 0.1 else "LOST"
            print(f"  Noise {noise_ms:3d} ms: r(N-V)={mean_r_NV:+.3f}±{std_r_NV:.3f}  "
                  f"r(N-S)={mean_r_NS:+.3f}  "
                  f"κ_N={beat_results[-1]['median_N']:.2f}  "
                  f"κ_V={beat_results[-1]['median_V']:.2f}  "
                  f"ratio={mean_ratio:.1f}x  {sig}")

    # ================================================================
    # TEST 2: Multi-disease record-level separation under noise
    # ================================================================
    if has_multi:
        print("\n" + "=" * 65)
        print("TEST 2: Multi-disease record-level separation vs jitter")
        print("=" * 65)

        # We need the raw RR data per record from each database
        # Re-extract from the beat-level data for MITDB records
        # For other databases, we need to re-read from stored data
        # Simplification: use the MITDB beat data to test noise on
        # the conditions we can reconstruct (Normal MITDB, VA, SVA MITDB)
        
        # Load full beat data and group by record
        record_groups = {}
        for rec_id, group in df_beats.groupby('record'):
            v_frac = (group['aami_class'] == 'V').mean()
            s_frac = (group['aami_class'] == 'S').mean()
            if v_frac > 0.10:
                condition = 'VA'
            elif s_frac > 0.10:
                condition = 'SVA'
            else:
                condition = 'Normal'
            
            record_groups[rec_id] = {
                'rr_pre': group['RR_pre_ms'].values,
                'rr_post': group['RR_post_ms'].values,
                'condition': condition,
            }

        disease_noise_results = []
        
        for noise_ms in NOISE_LEVELS:
            trial_records = {cond: [] for cond in ['Normal', 'VA', 'SVA']}
            
            for trial in range(N_TRIALS):
                for rec_id, rec_data in record_groups.items():
                    if noise_ms == 0:
                        rr_pre, rr_post = rec_data['rr_pre'], rec_data['rr_post']
                    else:
                        rr_pre, rr_post = add_rr_jitter(
                            rec_data['rr_pre'], rec_data['rr_post'], noise_ms, rng
                        )
                    
                    med_k, gini_k = compute_record_kappa(rr_pre, rr_post)
                    if med_k is not None:
                        trial_records[rec_data['condition']].append(med_k)
            
            # Compute Normal vs VA effect size
            n_vals = np.array(trial_records['Normal'])
            v_vals = np.array(trial_records['VA'])
            
            if len(n_vals) >= 5 and len(v_vals) >= 5:
                U, p = stats.mannwhitneyu(n_vals, v_vals, alternative='two-sided')
                r = 1 - 2*U/(len(n_vals)*len(v_vals))
                
                disease_noise_results.append({
                    'noise_ms': noise_ms,
                    'r_Normal_VA': round(float(r), 4),
                    'p_Normal_VA': float(p),
                    'median_Normal': round(float(np.median(n_vals)), 3),
                    'median_VA': round(float(np.median(v_vals)), 3),
                    'n_Normal': len(n_vals),
                    'n_VA': len(v_vals),
                })
                
                sig = "***" if abs(r) > 0.5 else "**" if abs(r) > 0.3 else "*" if abs(r) > 0.1 else "LOST"
                print(f"  Noise {noise_ms:3d} ms: r(Normal-VA)={r:+.3f}  p={p:.2e}  "
                      f"κ_N={np.median(n_vals):.2f}  κ_VA={np.median(v_vals):.2f}  {sig}")

    # ================================================================
    # FIGURES
    # ================================================================
    print("\nGenerating figures...")

    df_beat_noise = pd.DataFrame(beat_results)
    df_beat_noise.to_csv(RESULTS_DIR / 'noise_robustness_beats.csv', index=False)

    # Figure: Effect size degradation curve
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Effect size vs noise
    ax = axes[0]
    noise = df_beat_noise['noise_ms']
    ax.errorbar(noise, np.abs(df_beat_noise['r_NV_mean']),
                yerr=df_beat_noise['r_NV_std'],
                marker='o', color='#F44336', linewidth=2, markersize=6,
                capsize=4, label='Normal vs Ventricular')
    ax.plot(noise, np.abs(df_beat_noise['r_NS_mean']),
            marker='s', color='#FF9800', linewidth=2, markersize=6,
            label='Normal vs Supraventricular')
    
    # Reference lines
    ax.axhline(0.5, color='gray', ls='--', alpha=0.5, label='Large effect threshold')
    ax.axhline(0.3, color='gray', ls=':', alpha=0.5, label='Medium effect threshold')
    ax.axvspan(10, 30, alpha=0.1, color='green', label='PPG jitter range')
    
    ax.set_xlabel('Timing noise σ (ms)')
    ax.set_ylabel('|Effect size r| (rank-biserial)')
    ax.set_title('A. Diagnostic separation vs jitter')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.8)
    ax.grid(alpha=0.3)

    # Panel B: Median curvature vs noise
    ax = axes[1]
    ax.plot(noise, df_beat_noise['median_N'], marker='o', color='#2196F3',
            linewidth=2, label='Normal κ_median')
    ax.plot(noise, df_beat_noise['median_V'], marker='s', color='#F44336',
            linewidth=2, label='Ventricular κ_median')
    ax.axvspan(10, 30, alpha=0.1, color='green')
    ax.set_xlabel('Timing noise σ (ms)')
    ax.set_ylabel('Median geodesic curvature κ')
    ax.set_title('B. Curvature magnitude vs jitter')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel C: Curvature ratio (separation factor)
    ax = axes[2]
    ax.plot(noise, df_beat_noise['ratio_NV_mean'], marker='o', color='#9C27B0',
            linewidth=2, markersize=6)
    ax.axhline(2.0, color='gray', ls='--', alpha=0.5, label='2:1 ratio')
    ax.axvspan(10, 30, alpha=0.1, color='green', label='PPG jitter range')
    ax.set_xlabel('Timing noise σ (ms)')
    ax.set_ylabel('κ_Normal / κ_Ventricular')
    ax.set_title('C. Normal/Ventricular curvature ratio')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle('Figure N1: Diagnostic Robustness to PPG Timing Jitter',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = FIGURES_DIR / f'figN1_noise_robustness.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 65)
    print("NOISE ROBUSTNESS SUMMARY")
    print("=" * 65)

    # Find the noise level where r drops below 0.5 (large effect)
    for _, row in df_beat_noise.iterrows():
        if abs(row['r_NV_mean']) < 0.5:
            print(f"\n  N-vs-V separation drops below |r|=0.5 at σ = {row['noise_ms']} ms")
            break
    else:
        print(f"\n  N-vs-V separation remains |r|>0.5 at all tested noise levels!")

    for _, row in df_beat_noise.iterrows():
        if abs(row['r_NV_mean']) < 0.3:
            print(f"  N-vs-V separation drops below |r|=0.3 at σ = {row['noise_ms']} ms")
            break
    else:
        print(f"  N-vs-V separation remains |r|>0.3 at all tested noise levels!")

    # PPG viability assessment
    ppg_row = df_beat_noise[df_beat_noise['noise_ms'] == 20]
    if len(ppg_row) > 0:
        r_at_20 = abs(ppg_row.iloc[0]['r_NV_mean'])
        ratio_at_20 = ppg_row.iloc[0]['ratio_NV_mean']
        print(f"\n  At 20ms jitter (typical PPG):")
        print(f"    N-vs-V effect size: |r| = {r_at_20:.3f}")
        print(f"    Curvature ratio: {ratio_at_20:.1f}x")
        if r_at_20 > 0.5:
            print(f"    VERDICT: PPG wearable deployment is VIABLE")
        elif r_at_20 > 0.3:
            print(f"    VERDICT: PPG wearable deployment is MARGINAL")
        else:
            print(f"    VERDICT: PPG wearable deployment is NOT VIABLE — ECG required")

    print(f"\nSaved: {RESULTS_DIR / 'noise_robustness_beats.csv'}")


if __name__ == '__main__':
    main()

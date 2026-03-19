"""
17_dance_refinement.py — Paper IV Quick Experiments
Cardiac Torus Pipeline

EXPERIMENT 1: Split Mosh Pit into AF vs SVA
  Rerun dance matching with 5 prototypes instead of 4

EXPERIMENT 2: Feature Ablation
  κ alone vs κ+Gini vs κ+Gini+spread
  Show which dimensions resolve the Waltz/Lock-Step overlap
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR

# =====================================================================
# DANCE PROTOTYPES
# =====================================================================

# Original 4-dance library (from Paper IV v1)
LIBRARY_V1 = {
    'Waltz (NSR)':      {'kappa': 10.0, 'gini': 0.30, 'spread': 2.8},
    'Lock-Step (CHF)':  {'kappa': 25.0, 'gini': 0.22, 'spread': 1.2},
    'Mosh Pit (AF)':    {'kappa': 3.3,  'gini': 0.20, 'spread': 3.5},
    'Stumble (PVC/VA)': {'kappa': 1.0,  'gini': 0.45, 'spread': 3.2},
}

# Refined 5-dance library: split Mosh Pit into AF + SVA
LIBRARY_V2 = {
    'Waltz (NSR)':      {'kappa': 10.0, 'gini': 0.30, 'spread': 2.8},
    'Lock-Step (CHF)':  {'kappa': 25.0, 'gini': 0.22, 'spread': 1.2},
    'Mosh Pit (AF)':    {'kappa': 3.3,  'gini': 0.35, 'spread': 3.5},
    'Sway (SVA)':       {'kappa': 7.6,  'gini': 0.45, 'spread': 3.0},
    'Stumble (PVC/VA)': {'kappa': 1.0,  'gini': 0.45, 'spread': 3.2},
}

# Condition-to-dance mapping (v2 with SVA split)
COND_TO_DANCE_V2 = {
    'Normal': 'Waltz (NSR)',
    'Normal (MITDB)': 'Waltz (NSR)',
    'Normal (NSR1)': 'Waltz (NSR)',
    'Normal (NSR2)': 'Waltz (NSR)',
    'Normal Sinus Rhythm': 'Waltz (NSR)',
    'CHF': 'Lock-Step (CHF)',
    'Congestive Heart Failure': 'Lock-Step (CHF)',
    'CHF (NYHA 3-4)': 'Lock-Step (CHF)',
    'CHF (NYHA 1-3)': 'Lock-Step (CHF)',
    'AF': 'Mosh Pit (AF)',
    'Atrial Fibrillation': 'Mosh Pit (AF)',
    'SVA': 'Sway (SVA)',
    'Supraventricular Arrhythmia': 'Sway (SVA)',
    'Supraventricular (MITDB)': 'Sway (SVA)',
    'VA': 'Stumble (PVC/VA)',
    'Ventricular Arrhythmia': 'Stumble (PVC/VA)',
}

# v1 mapping (Mosh Pit = AF + SVA combined)
COND_TO_DANCE_V1 = dict(COND_TO_DANCE_V2)
COND_TO_DANCE_V1['SVA'] = 'Mosh Pit (AF)'
COND_TO_DANCE_V1['Supraventricular Arrhythmia'] = 'Mosh Pit (AF)'
COND_TO_DANCE_V1['Supraventricular (MITDB)'] = 'Mosh Pit (AF)'


def classify_dance(kappa, gini, spread, library, features='all', reject_threshold=2.0):
    """
    Nearest-neighbor in normalized feature space.
    features: 'k' = κ only, 'kg' = κ+Gini, 'all' = κ+Gini+spread
    """
    k_scale = 25.0
    g_scale = 0.30
    s_scale = 3.0

    best_dance = 'Unclassified'
    best_dist = float('inf')

    for name, proto in library.items():
        if features == 'k':
            d = abs(kappa - proto['kappa']) / k_scale
        elif features == 'kg':
            d = np.sqrt(
                ((kappa - proto['kappa'])/k_scale)**2 +
                ((gini - proto['gini'])/g_scale)**2
            )
        else:  # all
            d = np.sqrt(
                ((kappa - proto['kappa'])/k_scale)**2 +
                ((gini - proto['gini'])/g_scale)**2 +
                ((spread - proto['spread'])/s_scale)**2
            )
        if d < best_dist:
            best_dist = d
            best_dance = name

    if best_dist > reject_threshold:
        best_dance = 'Unclassified'

    return best_dance, best_dist


def load_data():
    """Load and merge multi-disease + CHF records."""
    multi_csv = RESULTS_DIR / 'multi_disease_records.csv'
    chf_csv = RESULTS_DIR / 'chf_replication_records.csv'

    if not multi_csv.exists():
        print(f"ERROR: {multi_csv} not found")
        return None

    df = pd.read_csv(multi_csv)
    print(f"Loaded {len(df)} multi-disease records")

    if chf_csv.exists():
        df_chf = pd.read_csv(chf_csv)
        print(f"Loaded {len(df_chf)} CHF replication records")
        # Extract just what we need
        chf_rows = []
        for _, row in df_chf.iterrows():
            chf_rows.append({
                'kappa_median': row.get('kappa_median', None),
                'gini': row.get('gini', None),
                'torus_spread': row.get('torus_spread', np.nan),
                'condition': row.get('condition', ''),
            })
        df_chf_clean = pd.DataFrame(chf_rows)
        df = pd.concat([df, df_chf_clean], ignore_index=True)
        print(f"Combined: {len(df)} total records")

    return df


def run_confusion(df, library, cond_map, features='all', label=''):
    """Run classification and print confusion matrix."""
    dances = list(library.keys()) + ['Unclassified']
    results = []

    for _, row in df.iterrows():
        k = row.get('kappa_median', None)
        g = row.get('gini', None)
        s = row.get('torus_spread', 2.5)
        cond = str(row.get('condition', ''))

        if pd.isna(k) or pd.isna(g): continue
        if cond not in cond_map: continue
        if pd.isna(s): s = 2.5

        true_dance = cond_map[cond]
        pred_dance, dist = classify_dance(float(k), float(g), float(s),
                                          library, features=features)
        results.append({
            'condition': cond,
            'true_dance': true_dance,
            'predicted_dance': pred_dance,
            'distance': dist,
            'kappa': float(k),
            'gini': float(g),
            'spread': float(s),
        })

    if not results:
        print(f"  No records classified")
        return 0, 0, {}

    df_r = pd.DataFrame(results)
    true_dances = sorted(df_r['true_dance'].unique())

    print(f"\n  {label}")
    print(f"  Records: {len(df_r)}")

    # Header
    header = f"  {'True Dance':25s}"
    for d in dances:
        short = d.split('(')[0].strip()[:8]
        header += f" {short:>8s}"
    header += f" {'Acc':>8s}"
    print(header)
    print("  " + "-" * len(header))

    total_correct = 0
    total = 0
    per_dance_acc = {}

    for td in true_dances:
        subset = df_r[df_r['true_dance'] == td]
        row_str = f"  {td:25s}"
        correct = 0
        for pd_name in dances:
            count = len(subset[subset['predicted_dance'] == pd_name])
            if pd_name == td:
                correct = count
            row_str += f" {count:8d}"
        acc = correct / len(subset) if len(subset) > 0 else 0
        row_str += f" {acc:7.1%}"
        print(row_str)
        total_correct += correct
        total += len(subset)
        per_dance_acc[td] = (correct, len(subset), acc)

    overall_acc = total_correct / total if total > 0 else 0
    print(f"\n  Overall: {total_correct}/{total} = {overall_acc:.1%}")

    return total_correct, total, per_dance_acc


def main():
    print("=" * 65)
    print("Paper IV: Dance Vocabulary Refinement Experiments")
    print("=" * 65)

    df = load_data()
    if df is None:
        return

    # Print unique conditions
    print(f"\nUnique conditions: {sorted(df['condition'].dropna().unique())}")

    # ==========================================================
    # EXPERIMENT 1: Mosh Pit Split
    # ==========================================================
    print("\n" + "=" * 65)
    print("EXPERIMENT 1: Mosh Pit Split (4 dances → 5 dances)")
    print("=" * 65)

    print("\n--- BEFORE (4 dances, AF+SVA combined): ---")
    c1, t1, acc1 = run_confusion(df, LIBRARY_V1, COND_TO_DANCE_V1,
                                  features='all', label='V1: 4 DANCES')

    print("\n--- AFTER (5 dances, AF and SVA separated): ---")
    c2, t2, acc2 = run_confusion(df, LIBRARY_V2, COND_TO_DANCE_V2,
                                  features='all', label='V2: 5 DANCES (Mosh Pit split)')

    if t1 > 0 and t2 > 0:
        print(f"\n  IMPROVEMENT: {c1}/{t1} ({100*c1/t1:.1f}%) → {c2}/{t2} ({100*c2/t2:.1f}%)")
        print(f"  Gain: +{100*(c2/t2 - c1/t1):.1f} percentage points")

    # ==========================================================
    # EXPERIMENT 2: Feature Ablation
    # ==========================================================
    print("\n" + "=" * 65)
    print("EXPERIMENT 2: Feature Ablation (κ vs κ+G vs κ+G+spread)")
    print("=" * 65)

    # Use v2 library (5 dances) for all ablation runs
    print("\n--- κ only: ---")
    ck, tk, acck = run_confusion(df, LIBRARY_V2, COND_TO_DANCE_V2,
                                  features='k', label='κ ONLY')

    print("\n--- κ + Gini: ---")
    ckg, tkg, acckg = run_confusion(df, LIBRARY_V2, COND_TO_DANCE_V2,
                                     features='kg', label='κ + GINI')

    print("\n--- κ + Gini + Spread: ---")
    ckgs, tkgs, acckgs = run_confusion(df, LIBRARY_V2, COND_TO_DANCE_V2,
                                        features='all', label='κ + GINI + SPREAD')

    # Summary table
    print("\n" + "=" * 65)
    print("ABLATION SUMMARY (5-dance library)")
    print("=" * 65)
    print(f"  {'Features':25s} {'Overall':>10s}")
    print(f"  {'-'*40}")
    if tk > 0: print(f"  {'κ only':25s} {100*ck/tk:9.1f}%")
    if tkg > 0: print(f"  {'κ + Gini':25s} {100*ckg/tkg:9.1f}%")
    if tkgs > 0: print(f"  {'κ + Gini + Spread':25s} {100*ckgs/tkgs:9.1f}%")

    # Per-dance ablation
    print(f"\n  {'Dance':25s} {'κ only':>10s} {'κ+Gini':>10s} {'κ+G+Sp':>10s}")
    print(f"  {'-'*60}")
    all_dances = sorted(set(list(acck.keys()) + list(acckg.keys()) + list(acckgs.keys())))
    for d in all_dances:
        k_acc = f"{100*acck[d][2]:.0f}%" if d in acck else "---"
        kg_acc = f"{100*acckg[d][2]:.0f}%" if d in acckg else "---"
        kgs_acc = f"{100*acckgs[d][2]:.0f}%" if d in acckgs else "---"
        print(f"  {d:25s} {k_acc:>10s} {kg_acc:>10s} {kgs_acc:>10s}")

    # ==========================================================
    # EXPERIMENT 3: Empirical centroids for v2 library
    # ==========================================================
    print("\n" + "=" * 65)
    print("EMPIRICAL CENTROIDS (v2 library)")
    print("=" * 65)

    for dance_name in LIBRARY_V2:
        matching_conds = [c for c, d in COND_TO_DANCE_V2.items() if d == dance_name]
        subset = df[df['condition'].isin(matching_conds)]
        if len(subset) > 0:
            emp_k = subset['kappa_median'].median()
            emp_g = subset['gini'].median()
            emp_s = subset['torus_spread'].median()
            proto = LIBRARY_V2[dance_name]
            print(f"  {dance_name:25s}  n={len(subset):4d}  "
                  f"κ={emp_k:6.1f} (proto: {proto['kappa']:5.1f})  "
                  f"G={emp_g:.3f} (proto: {proto['gini']:.2f})  "
                  f"Sp={emp_s:.2f} (proto: {proto['spread']:.1f})")

    # ==========================================================
    # Rerun with empirical centroids
    # ==========================================================
    print("\n" + "=" * 65)
    print("EXPERIMENT 4: Empirical Centroids Classification")
    print("=" * 65)

    # Build empirical library
    LIBRARY_EMP = {}
    for dance_name in LIBRARY_V2:
        matching_conds = [c for c, d in COND_TO_DANCE_V2.items() if d == dance_name]
        subset = df[df['condition'].isin(matching_conds)]
        if len(subset) > 0:
            LIBRARY_EMP[dance_name] = {
                'kappa': float(subset['kappa_median'].median()),
                'gini': float(subset['gini'].median()),
                'spread': float(subset['torus_spread'].fillna(2.5).median()),
            }

    if LIBRARY_EMP:
        print("\n--- Empirical centroids, κ + Gini + Spread: ---")
        ce, te, acce = run_confusion(df, LIBRARY_EMP, COND_TO_DANCE_V2,
                                      features='all', label='EMPIRICAL CENTROIDS (5 dances)')

    # Save results
    print(f"\n  All results printed above.")
    print(f"  Done.")


if __name__ == '__main__':
    main()

"""
19_fetal_validation.py — Paper V Kill-or-Cure Tests
Cardiac Torus Pipeline

TEST 1: Independence — partial correlation κ vs pH controlling for std_fhr
TEST 2: Final 30 minutes — recompute torus on last 30 min only
TEST 3: ROC classification — pH < 7.20 detection (std_fhr vs κ vs combined)
TEST 4: Duration control — partial correlation controlling for recording length
TEST 5: Regression — pH ~ std_fhr + κ_mean (does κ add to the model?)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR

# Import torus functions from step 18
from importlib import import_module


def partial_spearman(x, y, z):
    """Partial Spearman correlation of x and y controlling for z."""
    rx = stats.spearmanr(x, z)[0]
    ry = stats.spearmanr(y, z)[0]
    rxy = stats.spearmanr(x, y)[0]
    
    denom = np.sqrt((1 - rx**2) * (1 - ry**2))
    if denom < 1e-10:
        return 0.0, 1.0
    
    partial_r = (rxy - rx * ry) / denom
    
    # Approximate p-value via Fisher z-transform
    n = len(x)
    if n <= 4:
        return partial_r, 1.0
    z_val = 0.5 * np.log((1 + partial_r) / (1 - partial_r + 1e-10))
    se = 1.0 / np.sqrt(n - 4)
    p = 2 * (1 - stats.norm.cdf(abs(z_val / se)))
    
    return round(partial_r, 4), p


def main():
    print("=" * 65)
    print("Paper V: Fetal Torus — Kill-or-Cure Validation Tests")
    print("=" * 65)
    
    # Load results
    csv_path = RESULTS_DIR / 'fetal_torus_results.csv'
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run 18_fetal_torus.py first.")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} fetal records")
    
    # Ensure pH is available
    df_ph = df.dropna(subset=['pH'])
    print(f"Records with pH: {len(df_ph)}")
    
    # ================================================================
    # TEST 1: INDEPENDENCE FROM std_fhr
    # ================================================================
    print("\n" + "=" * 65)
    print("TEST 1: INDEPENDENCE — Does κ add info beyond std_fhr?")
    print("=" * 65)
    
    # Need std_fhr and kappa columns
    test_cols = ['kappa_median', 'kappa_mean', 'kappa_std', 'kappa_p95',
                 'kappa_cv', 'gini', 'torus_spread', 'speed_cv']
    
    valid = df_ph.dropna(subset=['std_fhr_bpm'] + test_cols)
    print(f"  Records with complete data: {len(valid)}")
    
    print(f"\n  {'Feature':25s} {'Raw ρ':>10s} {'Partial ρ':>12s} {'p(partial)':>12s} {'Survives?':>10s}")
    print("  " + "-" * 75)
    
    survivors = []
    for col in test_cols:
        raw_rho, raw_p = stats.spearmanr(valid[col], valid['pH'])
        partial_rho, partial_p = partial_spearman(
            valid[col].values, valid['pH'].values, valid['std_fhr_bpm'].values
        )
        survives = "YES" if partial_p < 0.05 else "no"
        sig = "***" if partial_p < 0.001 else "**" if partial_p < 0.01 else "*" if partial_p < 0.05 else ""
        print(f"  {col:25s} {raw_rho:+10.4f} {partial_rho:+12.4f} {partial_p:12.2e} {survives:>6s} {sig}")
        
        if partial_p < 0.05:
            survivors.append((col, partial_rho, partial_p))
    
    print(f"\n  >>> {len(survivors)} of {len(test_cols)} torus features survive controlling for std_fhr")
    
    if survivors:
        best = min(survivors, key=lambda x: x[2])
        print(f"  >>> Best surviving feature: {best[0]} (partial ρ = {best[1]:+.4f}, p = {best[2]:.2e})")
        print(f"  >>> INDEPENDENCE TEST: PASSED — κ carries information beyond FHR variability")
    else:
        print(f"  >>> INDEPENDENCE TEST: FAILED — std_fhr absorbs all torus signal")
    
    # Also run the regression
    print(f"\n  REGRESSION: pH ~ std_fhr + κ_mean")
    try:
        from sklearn.linear_model import LinearRegression
        
        X1 = valid[['std_fhr_bpm']].values
        X2 = valid[['std_fhr_bpm', 'kappa_mean']].values
        y = valid['pH'].values
        
        m1 = LinearRegression().fit(X1, y)
        m2 = LinearRegression().fit(X2, y)
        
        r2_1 = m1.score(X1, y)
        r2_2 = m2.score(X2, y)
        
        print(f"    std_fhr alone:        R² = {r2_1:.4f}")
        print(f"    std_fhr + κ_mean:     R² = {r2_2:.4f}")
        print(f"    κ_mean adds:          ΔR² = {r2_2 - r2_1:.4f} ({100*(r2_2-r2_1)/r2_1:.1f}% improvement)")
        
        # Add more features
        feat_cols = ['std_fhr_bpm', 'kappa_mean', 'gini', 'speed_cv']
        avail = [c for c in feat_cols if c in valid.columns]
        X3 = valid[avail].values
        m3 = LinearRegression().fit(X3, y)
        r2_3 = m3.score(X3, y)
        print(f"    std_fhr + κ + G + sCV: R² = {r2_3:.4f}")
        print(f"    Full torus adds:      ΔR² = {r2_3 - r2_1:.4f} ({100*(r2_3-r2_1)/r2_1:.1f}% improvement)")
        
    except ImportError:
        print("    sklearn not available")
    
    # ================================================================
    # TEST 2: FINAL 30 MINUTES
    # ================================================================
    print("\n" + "=" * 65)
    print("TEST 2: FINAL 30 MINUTES — Last 30 min before delivery")
    print("=" * 65)
    
    try:
        import wfdb
        from pathlib import Path as P
        
        # Find CTG data directory
        ctg_dir = None
        for candidate in [Path('data/ctg'), Path('data/ctg/1.0.0')]:
            if candidate.exists():
                ctg_dir = candidate
                break
        
        if ctg_dir is None:
            print("  CTG data directory not found — skipping")
        else:
            # Import torus computation from step 18
            sys.path.insert(0, str(Path(__file__).parent))
            from importlib.util import spec_from_file_location, module_from_spec
            
            spec = spec_from_file_location("fetal", str(Path(__file__).parent / "18_fetal_torus.py"))
            fetal_mod = module_from_spec(spec)
            spec.loader.exec_module(fetal_mod)
            
            dat_files = sorted(ctg_dir.glob('*.dat'))
            if not dat_files:
                dat_files = sorted(ctg_dir.rglob('*.dat'))
            
            print(f"  Found {len(dat_files)} CTG recordings")
            print(f"  Reprocessing with final 30 min only...")
            
            results_30 = []
            
            for dat_path in dat_files:
                record_id = dat_path.stem
                
                fhr, metadata = fetal_mod.read_ctg_record(dat_path)
                if fhr is None:
                    continue
                
                # Take only last 30 minutes (30 * 60 * 4 = 7200 samples at 4 Hz)
                last_30_samples = 30 * 60 * 4
                if len(fhr) > last_30_samples:
                    fhr_30 = fhr[-last_30_samples:]
                else:
                    fhr_30 = fhr  # recording shorter than 30 min, use all
                
                intervals, interval_stats = fetal_mod.extract_fhr_intervals(fhr_30)
                if intervals is None or len(intervals) < 20:
                    continue
                
                torus = fetal_mod.compute_torus_features(intervals)
                if torus is None:
                    continue
                
                clinical = fetal_mod.parse_clinical_metadata(metadata)
                
                record = {
                    'record': record_id,
                    **interval_stats,
                    **torus,
                    **clinical,
                }
                results_30.append(record)
            
            if results_30:
                df_30 = pd.DataFrame(results_30)
                df_30_ph = df_30.dropna(subset=['pH'])
                
                print(f"  Processed: {len(results_30)} (final 30 min)")
                print(f"  With pH: {len(df_30_ph)}")
                
                # Correlations
                print(f"\n  FINAL-30 vs FULL RECORDING:")
                print(f"  {'Feature':25s} {'Full ρ':>10s} {'30-min ρ':>10s} {'Change':>10s}")
                print("  " + "-" * 60)
                
                for col in ['kappa_median', 'kappa_mean', 'gini', 'speed_cv', 'std_fhr_bpm']:
                    if col in df_30_ph.columns and col in df_ph.columns:
                        full_valid = df_ph[[col, 'pH']].dropna()
                        min30_valid = df_30_ph[[col, 'pH']].dropna()
                        
                        if len(full_valid) >= 20 and len(min30_valid) >= 20:
                            rho_full, _ = stats.spearmanr(full_valid[col], full_valid['pH'])
                            rho_30, p_30 = stats.spearmanr(min30_valid[col], min30_valid['pH'])
                            sig = "***" if p_30 < 0.001 else "**" if p_30 < 0.01 else "*" if p_30 < 0.05 else ""
                            change = rho_30 - rho_full
                            print(f"  {col:25s} {rho_full:+10.4f} {rho_30:+10.4f} {change:+10.4f} {sig}")
                
                # Independence test on 30-min data
                valid30 = df_30_ph.dropna(subset=['std_fhr_bpm', 'kappa_mean'])
                if len(valid30) >= 30:
                    pr, pp = partial_spearman(
                        valid30['kappa_mean'].values, valid30['pH'].values, valid30['std_fhr_bpm'].values
                    )
                    print(f"\n  Independence (30-min): κ_mean partial ρ = {pr:+.4f} (p = {pp:.2e})")
                    if pp < 0.05:
                        print(f"  >>> SURVIVES on final 30 minutes")
                    else:
                        print(f"  >>> Does NOT survive on final 30 minutes")
                
                df_30.to_csv(RESULTS_DIR / 'fetal_torus_30min_results.csv', index=False)
    
    except Exception as e:
        print(f"  Error in final-30 analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # ================================================================
    # TEST 3: ROC CLASSIFICATION (pH < 7.20)
    # ================================================================
    print("\n" + "=" * 65)
    print("TEST 3: ROC — Predicting pH < 7.20 (clinically acidotic)")
    print("=" * 65)
    
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, balanced_accuracy_score
        
        # Binary outcome
        df_roc = df_ph.copy()
        df_roc['acidotic'] = (df_roc['pH'] < 7.20).astype(int)
        
        n_acid = df_roc['acidotic'].sum()
        n_norm = len(df_roc) - n_acid
        print(f"  Normal (pH ≥ 7.20): {n_norm}")
        print(f"  Acidotic (pH < 7.20): {n_acid}")
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y = df_roc['acidotic'].values
        
        feature_sets = {
            'std_fhr alone': ['std_fhr_bpm'],
            'κ_mean alone': ['kappa_mean'],
            'κ_median alone': ['kappa_median'],
            'Gini alone': ['gini'],
            'speed_cv alone': ['speed_cv'],
            'All torus (κ+G+sCV)': ['kappa_mean', 'gini', 'speed_cv'],
            'std_fhr + κ_mean': ['std_fhr_bpm', 'kappa_mean'],
            'std_fhr + all torus': ['std_fhr_bpm', 'kappa_mean', 'gini', 'speed_cv'],
            'Full (std+mean_fhr+torus)': ['std_fhr_bpm', 'mean_fhr_bpm', 'kappa_mean', 'kappa_std',
                                           'gini', 'speed_cv', 'torus_spread'],
        }
        
        print(f"\n  {'Feature Set':35s} {'AUC':>8s} {'Bal.Acc':>8s}")
        print("  " + "-" * 55)
        
        for name, cols in feature_sets.items():
            avail_cols = [c for c in cols if c in df_roc.columns]
            if not avail_cols:
                continue
            
            X = df_roc[avail_cols].fillna(0).values
            
            aucs = []
            baccs = []
            for tr_idx, te_idx in skf.split(X, y):
                sc = StandardScaler()
                clf = LogisticRegression(max_iter=1000, class_weight='balanced')
                X_tr = sc.fit_transform(X[tr_idx])
                X_te = sc.transform(X[te_idx])
                clf.fit(X_tr, y[tr_idx])
                probs = clf.predict_proba(X_te)[:, 1]
                preds = clf.predict(X_te)
                aucs.append(roc_auc_score(y[te_idx], probs))
                baccs.append(balanced_accuracy_score(y[te_idx], preds))
            
            print(f"  {name:35s} {np.mean(aucs):8.3f} {np.mean(baccs):8.3f}")
        
        # Torus gain
        print(f"\n  Key comparison:")
        
    except ImportError:
        print("  sklearn not available — skipping classification")
    
    # ================================================================
    # TEST 4: DURATION CONTROL
    # ================================================================
    print("\n" + "=" * 65)
    print("TEST 4: DURATION CONTROL")
    print("=" * 65)
    
    if 'duration_min' in df_ph.columns and 'n_beats' in df_ph.columns:
        valid = df_ph.dropna(subset=['duration_min', 'kappa_mean', 'kappa_median'])
        
        # Duration vs pH
        rho_dur, p_dur = stats.spearmanr(valid['duration_min'], valid['pH'])
        print(f"  Duration vs pH: ρ = {rho_dur:+.4f} (p = {p_dur:.2e})")
        
        rho_nb, p_nb = stats.spearmanr(valid['n_beats'], valid['pH'])
        print(f"  n_beats vs pH:  ρ = {rho_nb:+.4f} (p = {p_nb:.2e})")
        
        # Partial correlations controlling for duration
        print(f"\n  Partial correlations controlling for duration:")
        print(f"  {'Feature':25s} {'Raw ρ':>10s} {'Partial ρ':>12s} {'p':>12s} {'Survives?':>10s}")
        print("  " + "-" * 75)
        
        for col in ['kappa_median', 'kappa_mean', 'gini', 'speed_cv', 'std_fhr_bpm']:
            if col in valid.columns:
                v = valid.dropna(subset=[col])
                raw_rho, _ = stats.spearmanr(v[col], v['pH'])
                pr, pp = partial_spearman(v[col].values, v['pH'].values, v['duration_min'].values)
                survives = "YES" if pp < 0.05 else "no"
                sig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else ""
                print(f"  {col:25s} {raw_rho:+10.4f} {pr:+12.4f} {pp:12.2e} {survives:>6s} {sig}")
        
        # n_beats control
        print(f"\n  Partial correlations controlling for n_beats:")
        for col in ['kappa_median', 'kappa_mean', 'gini', 'speed_cv']:
            if col in valid.columns:
                v = valid.dropna(subset=[col])
                pr, pp = partial_spearman(v[col].values, v['pH'].values, v['n_beats'].values)
                survives = "YES" if pp < 0.05 else "no"
                sig = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else ""
                print(f"  {col:25s} partial ρ = {pr:+.4f} (p = {pp:.2e}) {survives} {sig}")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 65)
    print("PAPER V VALIDATION SUMMARY")
    print("=" * 65)
    print("  Test 1 (Independence from std_fhr): Results above")
    print("  Test 2 (Final 30 minutes): Results above")
    print("  Test 3 (ROC pH < 7.20): Results above")
    print("  Test 4 (Duration control): Results above")
    print(f"\n  All results saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()

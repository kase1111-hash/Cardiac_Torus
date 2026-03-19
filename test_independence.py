"""
Quick test: Is torus curvature just a nonlinear transform of HRV?

If κ ~ SDNN + RMSSD + DFA_α1 + SD1 + SD2 explains all variance,
the torus adds nothing new. If there's significant residual, 
curvature captures independent geometric information.
"""
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('results/hrv_vs_torus_features.csv')
print(f"Records: {len(df)}")

hrv_cols = ['SDNN', 'RMSSD', 'pNN50', 'SD1', 'SD2', 'SD1_SD2', 'CV_RR', 'DFA_alpha1']
torus_cols = ['kappa_median', 'gini_kappa', 'torus_spread', 'torus_speed_cv', 'Q2_frac']

clean = df.dropna(subset=hrv_cols + torus_cols)
print(f"Complete records: {len(clean)}")

# Test 1: Multiple regression — κ predicted by all HRV metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X_hrv = clean[hrv_cols].values
for tcol in torus_cols:
    y = clean[tcol].values
    reg = LinearRegression().fit(X_hrv, y)
    y_pred = reg.predict(X_hrv)
    r2 = r2_score(y, y_pred)
    residual = y - y_pred
    
    # Test if residual still separates CHF from Normal
    cond = clean['condition']
    chf_mask = cond.str.contains('CHF')
    norm_mask = cond.str.contains('Normal')
    
    resid_chf = residual[chf_mask]
    resid_norm = residual[norm_mask]
    
    if len(resid_chf) > 3 and len(resid_norm) > 3:
        U, p = stats.mannwhitneyu(resid_chf, resid_norm, alternative='two-sided')
        r = 1 - 2*U/(len(resid_chf)*len(resid_norm))
        print(f"\n{tcol}:")
        print(f"  R² from HRV: {r2:.3f}  (HRV explains {r2*100:.1f}% of variance)")
        print(f"  Residual CHF vs Normal: r={r:+.3f}, p={p:.3e}")
        if abs(r) > 0.2:
            print(f"  >>> RESIDUAL STILL SEPARATES — independent information")
        else:
            print(f"  >>> Residual does NOT separate — may be reducible")
    else:
        print(f"\n{tcol}: R²={r2:.3f} (insufficient CHF/Normal for residual test)")

# Test 2: Partial correlation — κ vs condition, controlling for SDNN
print("\n" + "="*60)
print("PARTIAL CORRELATIONS (controlling for SDNN + DFA_alpha1)")
print("="*60)

# Encode condition as numeric: Normal=0, CHF=1
binary = clean[clean['condition'].str.contains('CHF|Normal')].copy()
binary['is_chf'] = binary['condition'].str.contains('CHF').astype(float)
print(f"Binary CHF/Normal records: {len(binary)}")

for tcol in ['kappa_median', 'gini_kappa', 'torus_spread']:
    # Raw correlation
    rho_raw, p_raw = stats.spearmanr(binary[tcol], binary['is_chf'])
    
    # Partial: regress out SDNN and DFA from both κ and is_chf
    X_ctrl = binary[['SDNN', 'DFA_alpha1']].values
    
    reg_k = LinearRegression().fit(X_ctrl, binary[tcol].values)
    resid_k = binary[tcol].values - reg_k.predict(X_ctrl)
    
    reg_y = LinearRegression().fit(X_ctrl, binary['is_chf'].values)
    resid_y = binary['is_chf'].values - reg_y.predict(X_ctrl)
    
    rho_partial, p_partial = stats.spearmanr(resid_k, resid_y)
    
    print(f"\n{tcol}:")
    print(f"  Raw ρ with CHF:     {rho_raw:+.3f} (p={p_raw:.3e})")
    print(f"  Partial ρ (ctrl SDNN+DFA): {rho_partial:+.3f} (p={p_partial:.3e})")
    if abs(rho_partial) > 0.15 and p_partial < 0.05:
        print(f"  >>> SURVIVES CONTROL — independent of SDNN+DFA")
    else:
        print(f"  >>> Does not survive control")

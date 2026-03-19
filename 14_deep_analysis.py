"""
14_deep_analysis.py — Three Deep Analyses on Existing Data
True North Research | Cardiac Torus Pipeline

ANALYSIS A: Combined Classifier
  Does torus + HRV beat either alone? Logistic regression and 
  random forest on the combined feature set from Step 12.

ANALYSIS B: Circadian Curvature
  Does κ follow a day-night cycle? Hourly curvature from 10-24h 
  recordings (NSR, CHF, AF databases). Tests whether the torus 
  tracks autonomic state across the circadian cycle.

ANALYSIS C: Pre-Arrhythmia Detection
  Does curvature change BEFORE a PVC fires? Rolling κ in the 
  beats preceding each ventricular ectopic event from MIT-BIH.
  If there's a pre-ectopic signature, that's predictive, not 
  just classificatory.

All three use data already on disk — no downloads needed.
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from config import (RESULTS_DIR, FIGURES_DIR, DPI, FIG_FORMAT,
                     RR_MIN_MS, RR_MAX_MS)

try:
    import wfdb
except ImportError:
    print("Install wfdb: pip install wfdb")
    sys.exit(1)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    print("WARNING: scikit-learn not installed. Install with: pip install scikit-learn")
    print("  Analysis A (combined classifier) will be skipped.")
    HAS_SKLEARN = False

DATA_BASE_DIR = Path(__file__).parent / "data"

# Torus functions
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

def compute_kappa_from_rr(rr):
    """Quick κ computation from an RR interval array."""
    valid = rr[(rr >= RR_MIN_MS) & (rr <= RR_MAX_MS)]
    if len(valid) < 20:
        return np.array([]), np.array([]), np.array([])
    rr_pre = valid[:-1]; rr_post = valid[1:]
    t1 = np.array([to_angle(r, RR_MIN_MS, RR_MAX_MS) for r in rr_pre])
    t2 = np.array([to_angle(r, RR_MIN_MS, RR_MAX_MS) for r in rr_post])
    n = len(t1)
    kappa = np.zeros(n)
    for i in range(1, n-1):
        kappa[i] = menger_curvature_torus((t1[i-1],t2[i-1]),(t1[i],t2[i]),(t1[i+1],t2[i+1]))
    return kappa, t1, t2


# ================================================================
# ANALYSIS A: Combined Classifier
# ================================================================

def analysis_a_combined_classifier():
    print("\n" + "=" * 70)
    print("ANALYSIS A: Combined Classifier — Torus + HRV")
    print("=" * 70)

    if not HAS_SKLEARN:
        print("  SKIPPED: scikit-learn not installed")
        return

    feat_path = RESULTS_DIR / 'hrv_vs_torus_features.csv'
    if not feat_path.exists():
        print(f"  SKIPPED: {feat_path} not found. Run step 12 first.")
        return

    df = pd.read_csv(feat_path)
    print(f"  Loaded {len(df)} records with features")

    # Define feature sets
    hrv_cols = ['SDNN', 'RMSSD', 'pNN50', 'SD1', 'SD2', 'SD1_SD2', 'CV_RR', 'DFA_alpha1']
    torus_cols = ['kappa_median', 'gini_kappa', 'torus_spread', 'torus_speed_cv', 'Q2_frac']
    all_cols = hrv_cols + torus_cols

    # Simplify conditions for classification
    condition_map = {
        'Normal (NSR1)': 'Normal',
        'Normal (NSR2)': 'Normal',
        'CHF (NYHA 3-4)': 'CHF',
        'CHF (NYHA 1-3)': 'CHF',
        'Atrial Fibrillation': 'AF',
        'SVA': 'SVA',
    }
    df['label'] = df['condition'].map(condition_map)
    df = df.dropna(subset=['label'])

    # Drop rows with any NaN in features
    df_clean = df.dropna(subset=all_cols).copy()
    print(f"  Records with complete features: {len(df_clean)}")
    print(f"  Class distribution:")
    for label, count in df_clean['label'].value_counts().items():
        print(f"    {label}: {count}")

    # Need at least 2 classes with enough samples
    class_counts = df_clean['label'].value_counts()
    valid_classes = class_counts[class_counts >= 5].index.tolist()
    df_clean = df_clean[df_clean['label'].isin(valid_classes)]

    if len(valid_classes) < 2:
        print("  Not enough classes with sufficient samples")
        return

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df_clean['label'])
    class_names = le.classes_

    # Run 3 feature sets × 2 classifiers
    feature_sets = {
        'HRV only': hrv_cols,
        'Torus only': torus_cols,
        'HRV + Torus': all_cols,
    }

    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=2000, C=1.0),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    print(f"\n  {'Feature Set':20s} {'Classifier':25s} {'Accuracy':>10s} {'Bal. Acc':>10s}")
    print("  " + "-" * 70)

    for feat_name, feat_cols in feature_sets.items():
        X = df_clean[feat_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for clf_name, clf in classifiers.items():
            try:
                acc_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
                bal_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='balanced_accuracy')

                mean_acc = acc_scores.mean()
                std_acc = acc_scores.std()
                mean_bal = bal_scores.mean()
                std_bal = bal_scores.std()

                print(f"  {feat_name:20s} {clf_name:25s} "
                      f"{mean_acc:.3f}\u00B1{std_acc:.3f} "
                      f"{mean_bal:.3f}\u00B1{std_bal:.3f}")

                results.append({
                    'feature_set': feat_name,
                    'classifier': clf_name,
                    'accuracy_mean': round(mean_acc, 4),
                    'accuracy_std': round(std_acc, 4),
                    'balanced_accuracy_mean': round(mean_bal, 4),
                    'balanced_accuracy_std': round(std_bal, 4),
                    'n_features': len(feat_cols),
                    'n_samples': len(y),
                })
            except Exception as e:
                print(f"  {feat_name:20s} {clf_name:25s} ERROR: {e}")

    # Feature importance from best combined model
    print(f"\n  Feature Importance (Random Forest, HRV + Torus):")
    X_all = StandardScaler().fit_transform(df_clean[all_cols].values)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X_all, y)
    importances = sorted(zip(all_cols, rf.feature_importances_), key=lambda x: x[1], reverse=True)
    for name, imp in importances:
        tag = " (TORUS)" if name in torus_cols else " (HRV)"
        bar = "\u2588" * int(imp * 50)
        print(f"    {name:20s} {imp:.3f} {bar}{tag}")

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_DIR / 'combined_classifier_results.csv', index=False)

    # Improvement calculation
    hrv_best = max([r['balanced_accuracy_mean'] for r in results if r['feature_set'] == 'HRV only'], default=0)
    torus_best = max([r['balanced_accuracy_mean'] for r in results if r['feature_set'] == 'Torus only'], default=0)
    combined_best = max([r['balanced_accuracy_mean'] for r in results if r['feature_set'] == 'HRV + Torus'], default=0)

    print(f"\n  VERDICT:")
    print(f"    HRV only best balanced acc:     {hrv_best:.3f}")
    print(f"    Torus only best balanced acc:    {torus_best:.3f}")
    print(f"    Combined best balanced acc:      {combined_best:.3f}")
    improvement = combined_best - hrv_best
    if improvement > 0.01:
        print(f"    \u2714 COMBINED WINS: +{improvement:.3f} over HRV alone")
    elif improvement > -0.01:
        print(f"    ~ COMPARABLE: {improvement:+.3f} difference")
    else:
        print(f"    \u2718 HRV alone is sufficient: {improvement:+.3f}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Accuracy comparison
    ax = axes[0]
    x = np.arange(len(feature_sets))
    width = 0.35
    for i, clf_name in enumerate(classifiers.keys()):
        vals = [r['balanced_accuracy_mean'] for r in results if r['classifier'] == clf_name]
        errs = [r['balanced_accuracy_std'] for r in results if r['classifier'] == clf_name]
        ax.bar(x + i*width, vals, width, yerr=errs, capsize=4,
               label=clf_name, alpha=0.8, edgecolor='black', linewidth=0.3)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(list(feature_sets.keys()), fontsize=9)
    ax.set_ylabel('Balanced Accuracy (5-fold CV)')
    ax.set_title('A. Classification accuracy by feature set')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    # Panel B: Feature importance
    ax = axes[1]
    names = [n for n, _ in importances]
    imps = [v for _, v in importances]
    colors = ['#F44336' if n in torus_cols else '#2196F3' for n in names]
    ax.barh(range(len(names)), imps, color=colors, alpha=0.8,
            edgecolor='black', linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Feature importance')
    ax.set_title('B. RF importance (red=torus, blue=HRV)')
    ax.invert_yaxis()

    fig.suptitle('Analysis A: Combined Classifier — Torus + HRV',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = FIGURES_DIR / f'figA_combined_classifier.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# ANALYSIS B: Circadian Curvature
# ================================================================

def analysis_b_circadian():
    print("\n" + "=" * 70)
    print("ANALYSIS B: Circadian Curvature Rhythm")
    print("=" * 70)

    DATABASES = {
        'nsrdb': {'physionet_id': 'nsrdb', 'condition': 'Normal', 'ann_ext': 'atr'},
        'nsr2db': {'physionet_id': 'nsr2db', 'condition': 'Normal', 'ann_ext': 'ecg'},
        'chfdb': {'physionet_id': 'chfdb', 'condition': 'CHF', 'ann_ext': 'ecg'},
        'chf2db': {'physionet_id': 'chf2db', 'condition': 'CHF', 'ann_ext': 'ecg'},
        'afdb': {'physionet_id': 'afdb', 'condition': 'AF', 'ann_ext': 'qrs'},
    }

    hourly_data = defaultdict(list)  # condition → list of (hour, median_kappa)
    record_curves = []

    for db_key, db_info in DATABASES.items():
        db_dir = DATA_BASE_DIR / db_key
        if not db_dir.exists():
            continue

        try:
            record_list = wfdb.get_record_list(db_info['physionet_id'],
                                                records_dir=str(db_dir))
        except:
            record_list = [f.stem for f in db_dir.glob('*.hea')]

        for rec_name in record_list:
            rec_path = str(db_dir / rec_name)

            ann = None
            for ext in [db_info['ann_ext'], 'ecg', 'atr', 'qrs']:
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

            beat_times_sec = ann.sample / fs
            rr = np.diff(ann.sample) / fs * 1000.0

            total_hours = (beat_times_sec[-1] - beat_times_sec[0]) / 3600
            if total_hours < 4:
                continue  # Need at least 4 hours

            # Slice into 1-hour windows
            hour_kappas = []
            for h in range(int(total_hours)):
                start_sec = h * 3600
                end_sec = (h + 1) * 3600

                # Get beats in this hour
                mask = (beat_times_sec[:-1] >= start_sec) & (beat_times_sec[:-1] < end_sec)
                hour_rr = rr[mask[:-1] if len(mask) > len(rr) else mask[:len(rr)]]

                if len(hour_rr) < 100:
                    continue

                kappa, _, _ = compute_kappa_from_rr(hour_rr)
                valid = kappa[kappa > 0]
                if len(valid) < 50:
                    continue

                med_k = float(np.median(valid))
                gini_k = gini_coefficient(valid)

                hourly_data[db_info['condition']].append({
                    'record': rec_name,
                    'hour': h,
                    'kappa_median': med_k,
                    'gini': gini_k,
                    'n_beats': len(hour_rr),
                })
                hour_kappas.append((h, med_k, gini_k))

            if len(hour_kappas) >= 4:
                record_curves.append({
                    'record': rec_name,
                    'condition': db_info['condition'],
                    'hours': [hk[0] for hk in hour_kappas],
                    'kappas': [hk[1] for hk in hour_kappas],
                    'ginis': [hk[2] for hk in hour_kappas],
                })

    print(f"  Records with hourly data: {len(record_curves)}")
    for cond in ['Normal', 'CHF', 'AF']:
        n = sum(1 for rc in record_curves if rc['condition'] == cond)
        nh = len(hourly_data[cond])
        print(f"    {cond}: {n} records, {nh} hourly windows")

    if not record_curves:
        print("  No hourly data available.")
        return

    # Compute circadian amplitude per record
    print(f"\n  Circadian amplitude (max κ - min κ over recording):")
    circadian_results = []
    for rc in record_curves:
        kappas = np.array(rc['kappas'])
        amplitude = float(np.max(kappas) - np.min(kappas))
        cv = float(np.std(kappas) / np.mean(kappas)) if np.mean(kappas) > 0 else 0
        circadian_results.append({
            'record': rc['record'],
            'condition': rc['condition'],
            'amplitude': round(amplitude, 3),
            'kappa_cv_hourly': round(cv, 4),
            'mean_kappa': round(float(np.mean(kappas)), 3),
            'n_hours': len(kappas),
        })

    df_circ = pd.DataFrame(circadian_results)
    df_circ.to_csv(RESULTS_DIR / 'circadian_curvature.csv', index=False)

    for cond in ['Normal', 'CHF', 'AF']:
        sub = df_circ[df_circ['condition'] == cond]
        if len(sub) > 0:
            print(f"    {cond}: amplitude={sub['amplitude'].median():.2f}  "
                  f"hourly CV={sub['kappa_cv_hourly'].median():.3f}  "
                  f"mean κ={sub['mean_kappa'].median():.2f}")

    # Test: does CHF have lower circadian amplitude?
    for cond in ['CHF', 'AF']:
        normal_amp = df_circ[df_circ['condition'] == 'Normal']['kappa_cv_hourly']
        disease_amp = df_circ[df_circ['condition'] == cond]['kappa_cv_hourly']
        if len(normal_amp) >= 3 and len(disease_amp) >= 3:
            U, p = stats.mannwhitneyu(normal_amp, disease_amp, alternative='two-sided')
            r = 1 - 2*U/(len(normal_amp)*len(disease_amp))
            print(f"    Normal vs {cond} circadian CV: r={r:+.3f}, p={p:.3e}")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    COND_COLORS = {'Normal': '#2196F3', 'CHF': '#FF9800', 'AF': '#F44336'}

    # Panel A: Hourly κ traces (overlay multiple records per condition)
    ax = axes[0]
    for cond in ['Normal', 'CHF', 'AF']:
        for rc in record_curves:
            if rc['condition'] == cond:
                ax.plot(rc['hours'], rc['kappas'], color=COND_COLORS[cond],
                        alpha=0.15, linewidth=0.8)
        # Mean curve
        cond_records = [rc for rc in record_curves if rc['condition'] == cond]
        if cond_records:
            max_h = max(len(rc['hours']) for rc in cond_records)
            for h in range(min(24, max_h)):
                vals = [rc['kappas'][h] for rc in cond_records if h < len(rc['kappas'])]
                if vals:
                    ax.scatter(h, np.median(vals), color=COND_COLORS[cond],
                               s=30, zorder=5, edgecolors='black', linewidths=0.3)

    ax.set_xlabel('Hour of recording')
    ax.set_ylabel('Median κ')
    ax.set_title('A. Hourly curvature traces')
    ax.grid(alpha=0.3)

    # Panel B: Circadian amplitude by condition
    ax = axes[1]
    plot_data = []
    plot_labels = []
    plot_colors = []
    for cond in ['Normal', 'CHF', 'AF']:
        vals = df_circ[df_circ['condition'] == cond]['kappa_cv_hourly']
        if len(vals) >= 3:
            plot_data.append(vals.values)
            plot_labels.append(f"{cond}\n(n={len(vals)})")
            plot_colors.append(COND_COLORS[cond])

    if plot_data:
        bp = ax.boxplot(plot_data, widths=0.6, patch_artist=True, showfliers=True,
                        flierprops=dict(marker='.', markersize=4, alpha=0.5),
                        medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], plot_colors):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(plot_labels)+1))
        ax.set_xticklabels(plot_labels)
    ax.set_ylabel('Hourly κ CV (circadian amplitude)')
    ax.set_title('B. Circadian amplitude by condition')
    ax.grid(axis='y', alpha=0.3)

    # Panel C: Hourly Gini
    ax = axes[2]
    for cond in ['Normal', 'CHF', 'AF']:
        for rc in record_curves:
            if rc['condition'] == cond:
                ax.plot(rc['hours'], rc['ginis'], color=COND_COLORS[cond],
                        alpha=0.15, linewidth=0.8)
    ax.set_xlabel('Hour of recording')
    ax.set_ylabel('Curvature Gini')
    ax.set_title('C. Hourly Gini traces')
    ax.grid(alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=COND_COLORS[c], lw=2, label=c) for c in ['Normal', 'CHF', 'AF']]
    axes[0].legend(handles=legend_elements, fontsize=9)

    fig.suptitle('Analysis B: Circadian Curvature Rhythm',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = FIGURES_DIR / f'figB_circadian.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# ANALYSIS C: Pre-Arrhythmia Detection
# ================================================================

def analysis_c_pre_arrhythmia():
    print("\n" + "=" * 70)
    print("ANALYSIS C: Pre-Arrhythmia Curvature Signature")
    print("=" * 70)

    # Load beat-level data with annotations
    beat_path = RESULTS_DIR / 'torus_curvature_analyzed.csv'
    if not beat_path.exists():
        beat_path = RESULTS_DIR / 'torus_curvature.csv'
    if not beat_path.exists():
        print(f"  SKIPPED: no beat-level data found")
        return

    df = pd.read_csv(beat_path)
    df['record'] = df['record'].astype(int).astype(str)
    print(f"  Loaded {len(df):,} beats")

    # For each V beat, extract the curvature of the N preceding beats
    WINDOW_BEFORE = 20  # beats before the V event
    WINDOW_AFTER = 10   # beats after (for comparison)

    pre_v_curves = []   # curvature profiles before V beats
    pre_n_curves = []   # curvature profiles before random N beats (control)

    for rec_id, group in df.groupby('record'):
        group = group.sort_values('beat_idx').reset_index(drop=True)
        kappa = group['kappa_A'].values
        classes = group['aami_class'].values

        if len(kappa) < WINDOW_BEFORE + WINDOW_AFTER + 5:
            continue

        # Find V beats with enough preceding N beats
        for i in range(WINDOW_BEFORE, len(kappa) - WINDOW_AFTER):
            if classes[i] == 'V':
                # Check that preceding beats are mostly N
                pre_classes = classes[i-WINDOW_BEFORE:i]
                if np.sum(pre_classes == 'N') >= WINDOW_BEFORE * 0.8:
                    pre_kappa = kappa[i-WINDOW_BEFORE:i+WINDOW_AFTER+1]
                    if len(pre_kappa) == WINDOW_BEFORE + WINDOW_AFTER + 1:
                        pre_v_curves.append(pre_kappa)

        # Control: random N beats with preceding N beats
        n_indices = np.where(classes == 'N')[0]
        n_indices = n_indices[(n_indices >= WINDOW_BEFORE) &
                              (n_indices < len(kappa) - WINDOW_AFTER)]

        if len(n_indices) > 0:
            # Sample up to 50 per record
            sample_size = min(50, len(n_indices))
            rng = np.random.default_rng(hash(rec_id) % 2**31)
            sampled = rng.choice(n_indices, sample_size, replace=False)

            for i in sampled:
                pre_classes = classes[i-WINDOW_BEFORE:i]
                if np.sum(pre_classes == 'N') >= WINDOW_BEFORE * 0.8:
                    pre_kappa = kappa[i-WINDOW_BEFORE:i+WINDOW_AFTER+1]
                    if len(pre_kappa) == WINDOW_BEFORE + WINDOW_AFTER + 1:
                        pre_n_curves.append(pre_kappa)

    pre_v = np.array(pre_v_curves) if pre_v_curves else np.array([])
    pre_n = np.array(pre_n_curves) if pre_n_curves else np.array([])

    print(f"  Pre-V event profiles: {len(pre_v)}")
    print(f"  Pre-N control profiles: {len(pre_n)}")

    if len(pre_v) < 10 or len(pre_n) < 10:
        print("  Not enough profiles for analysis")
        return

    # Compute median profiles
    beat_positions = np.arange(-WINDOW_BEFORE, WINDOW_AFTER + 1)
    median_pre_v = np.median(pre_v, axis=0)
    median_pre_n = np.median(pre_n, axis=0)
    p25_v = np.percentile(pre_v, 25, axis=0)
    p75_v = np.percentile(pre_v, 75, axis=0)
    p25_n = np.percentile(pre_n, 25, axis=0)
    p75_n = np.percentile(pre_n, 75, axis=0)

    # Statistical test at each beat position
    p_values = []
    effect_sizes = []
    for j in range(len(beat_positions)):
        v_vals = pre_v[:, j]
        n_vals = pre_n[:, j]
        valid_v = v_vals[v_vals > 0]
        valid_n = n_vals[n_vals > 0]
        if len(valid_v) >= 10 and len(valid_n) >= 10:
            U, p = stats.mannwhitneyu(valid_v, valid_n, alternative='two-sided')
            r = 1 - 2*U/(len(valid_v)*len(valid_n))
            p_values.append(p)
            effect_sizes.append(r)
        else:
            p_values.append(1.0)
            effect_sizes.append(0.0)

    # Find when divergence starts
    print(f"\n  Beat-by-beat effect sizes (negative = pre-V has lower κ):")
    print(f"  {'Beat':>6s} {'r':>8s} {'p':>12s} {'Sig':>5s}")
    for j in range(max(0, WINDOW_BEFORE-10), WINDOW_BEFORE + 5):
        bp = beat_positions[j]
        r = effect_sizes[j]
        p = p_values[j]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        marker = " <<<" if bp == 0 else ""
        print(f"  {bp:>+6d} {r:>+8.3f} {p:>12.2e} {sig:>5s}{marker}")

    # Aggregate: mean κ in beats -10 to -1 vs -20 to -11
    near_window = (-10, 0)  # 10 beats before V
    far_window = (-20, -10) # 20-11 beats before V

    near_v = pre_v[:, WINDOW_BEFORE+near_window[0]:WINDOW_BEFORE+near_window[1]]
    far_v = pre_v[:, WINDOW_BEFORE+far_window[0]:WINDOW_BEFORE+far_window[1]]
    near_n = pre_n[:, WINDOW_BEFORE+near_window[0]:WINDOW_BEFORE+near_window[1]]
    far_n = pre_n[:, WINDOW_BEFORE+far_window[0]:WINDOW_BEFORE+far_window[1]]

    mean_near_v = np.mean(near_v[near_v > 0]) if np.any(near_v > 0) else 0
    mean_far_v = np.mean(far_v[far_v > 0]) if np.any(far_v > 0) else 0
    mean_near_n = np.mean(near_n[near_n > 0]) if np.any(near_n > 0) else 0
    mean_far_n = np.mean(far_n[far_n > 0]) if np.any(far_n > 0) else 0

    print(f"\n  Aggregate curvature windows:")
    print(f"    Pre-V beats -20 to -11: κ = {mean_far_v:.3f}")
    print(f"    Pre-V beats -10 to -1:  κ = {mean_near_v:.3f}")
    print(f"    Pre-N beats -20 to -11: κ = {mean_far_n:.3f}")
    print(f"    Pre-N beats -10 to -1:  κ = {mean_near_n:.3f}")

    if mean_far_v > 0:
        change_v = (mean_near_v - mean_far_v) / mean_far_v * 100
        change_n = (mean_near_n - mean_far_n) / mean_far_n * 100 if mean_far_n > 0 else 0
        print(f"    Pre-V near/far change: {change_v:+.1f}%")
        print(f"    Pre-N near/far change: {change_n:+.1f}%")

    # Save
    results = {
        'n_pre_v': len(pre_v),
        'n_pre_n': len(pre_n),
        'effect_sizes': [round(float(r), 4) for r in effect_sizes],
        'p_values': [float(p) for p in p_values],
        'beat_positions': beat_positions.tolist(),
        'mean_near_v': round(mean_near_v, 4),
        'mean_far_v': round(mean_far_v, 4),
        'mean_near_n': round(mean_near_n, 4),
        'mean_far_n': round(mean_far_n, 4),
    }
    with open(RESULTS_DIR / 'pre_arrhythmia_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Median curvature profile
    ax = axes[0]
    ax.fill_between(beat_positions, p25_v, p75_v, color='#F44336', alpha=0.15)
    ax.fill_between(beat_positions, p25_n, p75_n, color='#2196F3', alpha=0.15)
    ax.plot(beat_positions, median_pre_v, color='#F44336', linewidth=2,
            label=f'Pre-V (n={len(pre_v)})')
    ax.plot(beat_positions, median_pre_n, color='#2196F3', linewidth=2,
            label=f'Pre-N control (n={len(pre_n)})')
    ax.axvline(0, color='black', ls='--', alpha=0.5, label='Event beat')
    ax.set_xlabel('Beats relative to event')
    ax.set_ylabel('Median κ')
    ax.set_title('A. Curvature profile around ectopic events')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel B: Effect size per beat
    ax = axes[1]
    colors = ['#F44336' if p < 0.05 else '#BBBBBB' for p in p_values]
    ax.bar(beat_positions, effect_sizes, color=colors, alpha=0.8,
           edgecolor='black', linewidth=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', ls='--', alpha=0.5)
    ax.set_xlabel('Beats relative to event')
    ax.set_ylabel('Effect size r (V vs N)')
    ax.set_title('B. Per-beat effect size (red = p<0.05)')
    ax.grid(alpha=0.3)

    # Panel C: κ ratio (pre-V / pre-N) as function of distance
    ax = axes[2]
    ratio = median_pre_v / np.maximum(median_pre_n, 0.01)
    ax.plot(beat_positions, ratio, color='#9C27B0', linewidth=2)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5, label='No difference')
    ax.axvline(0, color='black', ls='--', alpha=0.5)
    ax.set_xlabel('Beats relative to event')
    ax.set_ylabel('κ ratio (pre-V / pre-N)')
    ax.set_title('C. Curvature ratio approaching ectopy')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle('Analysis C: Pre-Arrhythmia Curvature Signature',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = FIGURES_DIR / f'figC_pre_arrhythmia.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 70)
    print("Step 14: Deep Analysis — Three Quick Wins")
    print("Cardiac Torus Pipeline")
    print("=" * 70)

    import time
    t0 = time.time()

    # Analysis A
    t_a = time.time()
    analysis_a_combined_classifier()
    print(f"  [Analysis A completed in {time.time()-t_a:.1f}s]")

    # Analysis B
    t_b = time.time()
    analysis_b_circadian()
    print(f"  [Analysis B completed in {time.time()-t_b:.1f}s]")

    # Analysis C
    t_c = time.time()
    analysis_c_pre_arrhythmia()
    print(f"  [Analysis C completed in {time.time()-t_c:.1f}s]")

    print(f"\n{'='*70}")
    print(f"All three analyses complete in {time.time()-t0:.1f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

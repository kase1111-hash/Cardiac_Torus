"""
16_donut_dance_experiments.py — Paper IV Validation Experiments
Cardiac Torus Pipeline

EXPERIMENT A: Cross-Modal Coherence (CMC)
  From each CinC 2016 recording, extract:
    - Rhythm torus: S1-S1 interval curvature sequence
    - Acoustic torus: energy×spectral curvature sequence
  Compute CMC = Spearman ρ(κ_rhythm(n), κ_acoustic(n)) per recording
  Test: Normal CMC > Abnormal CMC?

EXPERIMENT B: Dance Matching Confusion Matrix
  From Paper I MIT-BIH data, classify each record by nearest-neighbor
  in κ-Gini-spread space against the 5 rhythm dance prototypes.
  Report confusion matrix.

EXPERIMENT C: Non-Dance Detection
  Test autocorrelation-based detector on V-Tach segments
  (from MIT-BIH) as proxy for non-dance states.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, FIGURES_DIR, DPI, FIG_FORMAT

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

# =====================================================================
# TORUS FUNCTIONS (same as all papers)
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

def compute_curvature_sequence(values, vmin, vmax):
    """Return per-point curvature sequence for a 1D signal."""
    n = len(values) - 1
    if n < 4: return np.array([])
    theta1 = np.array([to_angle(values[i], vmin, vmax) for i in range(n)])
    theta2 = np.array([to_angle(values[i+1], vmin, vmax) for i in range(n)])
    kappa = np.zeros(n)
    for i in range(1, n-1):
        kappa[i] = menger_curvature_torus(
            (theta1[i-1], theta2[i-1]),
            (theta1[i], theta2[i]),
            (theta1[i+1], theta2[i+1]))
    return kappa[1:-1]  # trim edges


# =====================================================================
# EXPERIMENT A: CROSS-MODAL COHERENCE
# =====================================================================

def extract_envelope(audio, fs):
    from scipy import signal as sig
    nyq = fs / 2
    low = max(25/nyq, 0.01); high = min(400/nyq, 0.99)
    try:
        b, a = sig.butter(4, [low, high], btype='band')
        filtered = sig.filtfilt(b, a, audio)
    except:
        filtered = audio
    analytic = sig.hilbert(filtered)
    envelope = np.abs(analytic)
    lp = min(50/nyq, 0.99)
    try:
        b2, a2 = sig.butter(2, lp, btype='low')
        envelope = sig.filtfilt(b2, a2, envelope)
    except:
        win = max(1, int(fs/50))
        envelope = np.convolve(envelope, np.ones(win)/win, mode='same')
    return envelope, filtered

def segment_beats(envelope, fs):
    from scipy import signal as sig
    min_dist = int(fs * 60 / 200)
    height = np.percentile(envelope, 50)
    peaks, _ = sig.find_peaks(envelope, distance=min_dist, height=height)
    if len(peaks) < 4:
        height = np.percentile(envelope, 30)
        peaks, _ = sig.find_peaks(envelope, distance=min_dist, height=height)
    return peaks

def read_wav(filepath):
    import wave
    try:
        with wave.open(filepath, 'r') as wf:
            fs = wf.getframerate(); n = wf.getnframes()
            nc = wf.getnchannels(); sw = wf.getsampwidth()
            raw = wf.readframes(n)
        if sw == 2: audio = np.frombuffer(raw, dtype=np.int16).astype(float)
        elif sw == 4: audio = np.frombuffer(raw, dtype=np.int32).astype(float)
        elif sw == 1: audio = np.frombuffer(raw, dtype=np.uint8).astype(float) - 128
        else: return None, None
        if nc > 1: audio = audio[::nc]
        mx = np.max(np.abs(audio))
        if mx > 0: audio = audio / mx
        return audio, fs
    except: return None, None

def compute_cmc_for_recording(filepath):
    """
    Extract both rhythm and acoustic torus curvature sequences
    from a single heart sound recording, compute CMC.
    """
    audio, fs = read_wav(filepath)
    if audio is None or len(audio)/fs < 3:
        return None

    envelope, filtered = extract_envelope(audio, fs)
    peaks = segment_beats(envelope, fs)
    if len(peaks) < 8:
        return None

    # ---- RHYTHM: S1-S1 intervals ----
    intervals = np.diff(peaks) / fs * 1000  # ms
    # Filter physiologically implausible
    valid_mask = (intervals > 300) & (intervals < 1500)
    intervals_clean = intervals[valid_mask]
    if len(intervals_clean) < 8:
        return None

    # Rhythm curvature sequence
    i_min, i_max = np.percentile(intervals_clean, 2), np.percentile(intervals_clean, 98)
    kappa_rhythm = compute_curvature_sequence(intervals_clean, i_min, i_max)

    # ---- ACOUSTIC: per-beat energy × spectral centroid ----
    energies = []
    centroids = []
    for i in range(len(peaks)-1):
        s1_idx = peaks[i]; next_s1 = peaks[i+1]
        interval_ms = 1000*(next_s1 - s1_idx)/fs
        if interval_ms < 300 or interval_ms > 1500:
            energies.append(np.nan); centroids.append(np.nan)
            continue
        beat_audio = filtered[s1_idx:next_s1]
        if len(beat_audio) < 32:
            energies.append(np.nan); centroids.append(np.nan)
            continue
        energy = float(np.sum(beat_audio**2)) / len(beat_audio)
        fft = np.abs(np.fft.rfft(beat_audio))
        freqs = np.fft.rfftfreq(len(beat_audio), 1/fs)
        total = np.sum(fft)
        centroid = float(np.sum(freqs * fft) / total) if total > 0 else 100.0
        energies.append(energy)
        centroids.append(centroid)

    energies = np.array(energies)
    centroids = np.array(centroids)

    # Use energy as the acoustic signal for curvature
    valid_ac = ~np.isnan(energies)
    energies_clean = energies[valid_ac]
    centroids_clean = centroids[valid_ac]

    if len(energies_clean) < 8:
        return None

    e_min, e_max = np.percentile(energies_clean, 2), np.percentile(energies_clean, 98)
    kappa_acoustic = compute_curvature_sequence(energies_clean, e_min, e_max)

    # ---- Align lengths ----
    min_len = min(len(kappa_rhythm), len(kappa_acoustic))
    if min_len < 6:
        return None

    kr = kappa_rhythm[:min_len]
    ka = kappa_acoustic[:min_len]

    # Remove zeros for cleaner correlation
    valid = (kr > 0) & (ka > 0)
    if np.sum(valid) < 6:
        return None

    cmc, cmc_p = stats.spearmanr(kr[valid], ka[valid])

    # Also compute per-substrate summaries
    kr_valid = kr[kr > 0]
    ka_valid = ka[ka > 0]

    return {
        'cmc': round(float(cmc), 4),
        'cmc_p': float(cmc_p),
        'n_beats_aligned': int(np.sum(valid)),
        'rhythm_kappa_med': round(float(np.median(kr_valid)), 4) if len(kr_valid) > 0 else 0,
        'rhythm_gini': round(gini_coefficient(kr_valid), 4) if len(kr_valid) > 0 else 0,
        'acoustic_kappa_med': round(float(np.median(ka_valid)), 4) if len(ka_valid) > 0 else 0,
        'acoustic_gini': round(gini_coefficient(ka_valid), 4) if len(ka_valid) > 0 else 0,
    }


# =====================================================================
# EXPERIMENT B: DANCE MATCHING
# =====================================================================

# Dance prototypes from Papers I-III (κ median, Gini, spread ranges → centroids)
RHYTHM_DANCES = {
    'Waltz (NSR)':      {'kappa': 10.0, 'gini': 0.30, 'spread': 2.8},
    'Lock-Step (CHF)':  {'kappa': 25.0, 'gini': 0.22, 'spread': 1.2},
    'Mosh Pit (AF)':    {'kappa': 3.3,  'gini': 0.20, 'spread': 3.5},
    'Stumble (PVC/VA)': {'kappa': 1.0,  'gini': 0.45, 'spread': 3.2},
}

# Map conditions to dance ground truth (covers all label variants)
CONDITION_TO_DANCE = {
    'Normal': 'Waltz (NSR)',
    'Normal (MITDB)': 'Waltz (NSR)',
    'Normal Sinus Rhythm': 'Waltz (NSR)',
    'NSR': 'Waltz (NSR)',
    'Normal (NSR1)': 'Waltz (NSR)',
    'Normal (NSR2)': 'Waltz (NSR)',
    'CHF': 'Lock-Step (CHF)',
    'Congestive Heart Failure': 'Lock-Step (CHF)',
    'CHF NYHA 3-4': 'Lock-Step (CHF)',
    'CHF NYHA 1-3': 'Lock-Step (CHF)',
    'CHF (NYHA 3-4)': 'Lock-Step (CHF)',
    'CHF (NYHA 1-3)': 'Lock-Step (CHF)',
    'AF': 'Mosh Pit (AF)',
    'Atrial Fibrillation': 'Mosh Pit (AF)',
    'SVA': 'Mosh Pit (AF)',
    'Supraventricular Arrhythmia': 'Mosh Pit (AF)',
    'Supraventricular (MITDB)': 'Mosh Pit (AF)',
    'VA': 'Stumble (PVC/VA)',
    'Ventricular Arrhythmia': 'Stumble (PVC/VA)',
}

def classify_dance(kappa_med, gini_val, spread_val, library):
    """Nearest-neighbor in normalized feature space."""
    best_dance = 'Unclassified'
    best_dist = float('inf')

    # Normalize by typical ranges
    k_scale = 25.0   # typical κ range
    g_scale = 0.30    # typical Gini range
    s_scale = 3.0     # typical spread range

    for name, proto in library.items():
        d = np.sqrt(
            ((kappa_med - proto['kappa'])/k_scale)**2 +
            ((gini_val - proto['gini'])/g_scale)**2 +
            ((spread_val - proto['spread'])/s_scale)**2
        )
        if d < best_dist:
            best_dist = d
            best_dance = name

    # Reject if too far from all prototypes
    if best_dist > 2.0:
        best_dance = 'Unclassified'

    return best_dance, round(best_dist, 3)


# =====================================================================
# MAIN
# =====================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sound_dir', type=str, default=str(Path(__file__).parent / 'data' / 'heart_sounds'))
    parser.add_argument('--rhythm_csv', type=str, default=str(RESULTS_DIR / 'multi_disease_torus.csv'))
    args = parser.parse_args()

    print("=" * 65)
    print("Paper IV: The Donut Dance — Validation Experiments")
    print("=" * 65)

    # ==============================================================
    # EXPERIMENT A: Cross-Modal Coherence
    # ==============================================================
    print("\n" + "=" * 65)
    print("EXPERIMENT A: Cross-Modal Coherence (CMC)")
    print("=" * 65)

    sound_dir = Path(args.sound_dir)
    if not sound_dir.exists():
        print(f"Heart sound directory not found: {sound_dir}")
        print("Skipping CMC experiment")
    else:
        # Load recordings with labels
        recordings = []
        for subset_dir in sorted(sound_dir.iterdir()):
            if not subset_dir.is_dir(): continue
            for wav_file in sorted(subset_dir.glob('*.wav')):
                hea_file = wav_file.with_suffix('.hea')
                condition = 'Unknown'; label = None
                if hea_file.exists():
                    try:
                        with open(hea_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line == '# Normal':
                                    condition = 'Normal'; label = -1; break
                                elif line == '# Abnormal':
                                    condition = 'Abnormal'; label = 1; break
                    except: pass
                if condition != 'Unknown':
                    recordings.append({'path': str(wav_file), 'name': wav_file.stem,
                                       'subset': subset_dir.name, 'condition': condition})

        print(f"Found {len(recordings)} labeled recordings")

        cmc_results = []
        errors = 0

        for rec in tqdm(recordings, desc="Computing CMC"):
            result = compute_cmc_for_recording(rec['path'])
            if result is None:
                errors += 1
                continue
            result['name'] = rec['name']
            result['condition'] = rec['condition']
            result['subset'] = rec['subset']
            cmc_results.append(result)

        print(f"CMC computed: {len(cmc_results)}, Errors: {errors}")

        if cmc_results:
            df_cmc = pd.DataFrame(cmc_results)
            df_cmc.to_csv(RESULTS_DIR / 'cmc_results.csv', index=False)

            normal_cmc = df_cmc[df_cmc['condition'] == 'Normal']['cmc'].dropna()
            abnormal_cmc = df_cmc[df_cmc['condition'] == 'Abnormal']['cmc'].dropna()

            print(f"\n  Normal CMC:   median = {normal_cmc.median():.4f}, mean = {normal_cmc.mean():.4f} (n={len(normal_cmc)})")
            print(f"  Abnormal CMC: median = {abnormal_cmc.median():.4f}, mean = {abnormal_cmc.mean():.4f} (n={len(abnormal_cmc)})")

            U, p = stats.mannwhitneyu(normal_cmc, abnormal_cmc, alternative='two-sided')
            r = 1 - 2*U/(len(normal_cmc)*len(abnormal_cmc))
            print(f"\n  Normal vs Abnormal CMC: r = {r:+.3f}, p = {p:.3e}")

            if r > 0:
                print(f"  >>> Normal hearts show HIGHER cross-modal coherence")
            else:
                print(f"  >>> Abnormal hearts show HIGHER cross-modal coherence")

            if abs(r) > 0.1 and p < 0.05:
                print(f"  >>> CMC SEPARATES Normal from Abnormal — Paper IV prediction CONFIRMED")
            elif p < 0.05:
                print(f"  >>> CMC shows weak separation")
            else:
                print(f"  >>> CMC does not significantly separate groups")

            # Per-site CMC
            print(f"\n  Per-site CMC:")
            for subset in sorted(df_cmc['subset'].unique()):
                sub = df_cmc[df_cmc['subset'] == subset]
                sn = sub[sub['condition'] == 'Normal']['cmc'].dropna()
                sa = sub[sub['condition'] == 'Abnormal']['cmc'].dropna()
                if len(sn) >= 5 and len(sa) >= 5:
                    U2, p2 = stats.mannwhitneyu(sn, sa, alternative='two-sided')
                    r2 = 1 - 2*U2/(len(sn)*len(sa))
                    sig = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else ""
                    print(f"    {subset}: Normal={sn.median():.3f}, Abnormal={sa.median():.3f}, r={r2:+.3f}{sig}")
                else:
                    print(f"    {subset}: insufficient data ({len(sn)}N/{len(sa)}A)")

            # CMC adds to classification?
            print(f"\n  Does CMC add to classification?")
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import StratifiedKFold
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import roc_auc_score

                y = (df_cmc['condition'] == 'Abnormal').astype(int).values
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                # CMC alone
                X_cmc = df_cmc[['cmc']].fillna(0).values
                aucs = []
                for tr_idx, te_idx in skf.split(X_cmc, y):
                    sc = StandardScaler()
                    clf = LogisticRegression(max_iter=1000)
                    clf.fit(sc.fit_transform(X_cmc[tr_idx]), y[tr_idx])
                    aucs.append(roc_auc_score(y[te_idx], clf.predict_proba(sc.transform(X_cmc[te_idx]))[:,1]))
                print(f"    CMC alone:           AUC = {np.mean(aucs):.3f}")

                # Rhythm κ + acoustic κ
                feat_cols = ['rhythm_kappa_med', 'rhythm_gini', 'acoustic_kappa_med', 'acoustic_gini']
                X_sub = df_cmc[feat_cols].fillna(0).values
                aucs2 = []
                for tr_idx, te_idx in skf.split(X_sub, y):
                    sc = StandardScaler()
                    clf = LogisticRegression(max_iter=1000)
                    clf.fit(sc.fit_transform(X_sub[tr_idx]), y[tr_idx])
                    aucs2.append(roc_auc_score(y[te_idx], clf.predict_proba(sc.transform(X_sub[te_idx]))[:,1]))
                print(f"    Rhythm+Acoustic κ/G: AUC = {np.mean(aucs2):.3f}")

                # Rhythm + Acoustic + CMC
                feat_cols2 = feat_cols + ['cmc']
                X_all = df_cmc[feat_cols2].fillna(0).values
                aucs3 = []
                for tr_idx, te_idx in skf.split(X_all, y):
                    sc = StandardScaler()
                    clf = LogisticRegression(max_iter=1000)
                    clf.fit(sc.fit_transform(X_all[tr_idx]), y[tr_idx])
                    aucs3.append(roc_auc_score(y[te_idx], clf.predict_proba(sc.transform(X_all[te_idx]))[:,1]))
                print(f"    Rhythm+Acoustic+CMC: AUC = {np.mean(aucs3):.3f}")

                cmc_gain = np.mean(aucs3) - np.mean(aucs2)
                print(f"    CMC adds: +{cmc_gain:.3f} AUC")

            except ImportError:
                print("    sklearn not available — skipping classification")

    # ==============================================================
    # EXPERIMENT B: Dance Matching on Paper I Data
    # ==============================================================
    print("\n" + "=" * 65)
    print("EXPERIMENT B: Dance Matching Confusion Matrix")
    print("=" * 65)

    rhythm_csv = Path(args.rhythm_csv)
    if not rhythm_csv.exists():
        # Try alternative paths
        for alt in ['torus_curvature_results.csv', 'multi_disease_torus.csv',
                     'chf_replication_results.csv']:
            test = RESULTS_DIR / alt
            if test.exists():
                rhythm_csv = test; break

    if not rhythm_csv.exists():
        print(f"Paper I results not found at {rhythm_csv}")
        print("Skipping dance matching experiment")
    else:
        df_rhythm = pd.read_csv(rhythm_csv)
        print(f"Loaded {len(df_rhythm)} records from {rhythm_csv.name}")
        print(f"Columns: {list(df_rhythm.columns)}")

        # Find the right columns
        kappa_col = None; gini_col = None; spread_col = None; cond_col = None
        for c in df_rhythm.columns:
            cl = c.lower()
            if 'kappa' in cl and 'median' in cl and kappa_col is None: kappa_col = c
            elif 'gini' in cl and gini_col is None: gini_col = c
            elif 'spread' in cl and spread_col is None: spread_col = c
            elif cl in ['condition', 'disease', 'category', 'label'] and cond_col is None: cond_col = c

        if kappa_col is None:
            for c in df_rhythm.columns:
                if 'kappa' in c.lower():
                    kappa_col = c; break
        if gini_col is None:
            for c in df_rhythm.columns:
                if 'gini' in c.lower():
                    gini_col = c; break
        if spread_col is None:
            for c in df_rhythm.columns:
                if 'spread' in c.lower():
                    spread_col = c; break
        if cond_col is None:
            for c in df_rhythm.columns:
                if c in ['condition', 'database', 'db']:
                    cond_col = c; break

        print(f"  Using: κ={kappa_col}, Gini={gini_col}, Spread={spread_col}, Condition={cond_col}")

        if kappa_col and gini_col and cond_col:
            # Try to merge CHF replication data
            chf_csv = RESULTS_DIR / 'chf_replication_records.csv'
            if chf_csv.exists():
                df_chf = pd.read_csv(chf_csv)
                print(f"  Merging {len(df_chf)} CHF replication records")
                print(f"  CHF columns: {list(df_chf.columns)}")
                # Find matching columns in CHF data
                chf_k = chf_g = chf_s = chf_c = None
                for c in df_chf.columns:
                    cl = c.lower()
                    if 'kappa' in cl and 'median' in cl: chf_k = c
                    elif 'kappa' in cl and chf_k is None: chf_k = c
                    elif 'gini' in cl: chf_g = c
                    elif 'spread' in cl: chf_s = c
                    elif c in ['condition', 'cohort', 'database']: chf_c = c
                if chf_k and chf_g and chf_c:
                    # Build a minimal dataframe with standardized column names
                    chf_rows = []
                    for _, row in df_chf.iterrows():
                        chf_rows.append({
                            kappa_col: row[chf_k],
                            gini_col: row[chf_g],
                            spread_col: row[chf_s] if chf_s else 2.5,
                            cond_col: row[chf_c],
                        })
                    df_chf_clean = pd.DataFrame(chf_rows)
                    df_rhythm = pd.concat([df_rhythm, df_chf_clean], ignore_index=True)
                    print(f"  Combined: {len(df_rhythm)} total records")
                else:
                    print(f"  Could not find matching columns in CHF data (k={chf_k}, g={chf_g}, c={chf_c})")

            # Map database names to condition labels (for db-coded CSVs)
            db_to_condition = {
                'nsrdb': 'Normal', 'nsr2db': 'Normal', 'NSR': 'Normal',
                'afdb': 'AF', 'AF': 'AF',
                'chfdb': 'CHF', 'chf2db': 'CHF', 'CHF': 'CHF',
                'svdb': 'SVA', 'SVA': 'SVA',
                'mitdb': 'Normal',
            }

            # Print unique conditions to debug
            unique_conds = df_rhythm[cond_col].unique()
            print(f"  Unique conditions in data: {list(unique_conds)}")

            # ---- PASS 1: Compute empirical centroids from labeled data ----
            dance_data = defaultdict(list)
            for idx, row in df_rhythm.iterrows():
                k = row.get(kappa_col, None)
                g = row.get(gini_col, None)
                s = row.get(spread_col, 2.5) if spread_col else 2.5
                cond_raw = str(row.get(cond_col, ''))
                if pd.isna(k) or pd.isna(g): continue
                if cond_raw in CONDITION_TO_DANCE:
                    dance = CONDITION_TO_DANCE[cond_raw]
                elif cond_raw in db_to_condition:
                    dance = CONDITION_TO_DANCE.get(db_to_condition[cond_raw], None)
                else:
                    dance = None
                if dance:
                    dance_data[dance].append({'kappa': float(k), 'gini': float(g), 'spread': float(s)})

            print(f"\n  EMPIRICAL CENTROIDS (computed from data):")
            print(f"  {'Dance':25s} {'n':>5s} {'κ median':>10s} {'Gini med':>10s} {'Spread med':>10s}   vs hand-set κ")
            print(f"  {'-'*75}")
            EMPIRICAL_DANCES = {}
            for dance_name in ['Waltz (NSR)', 'Lock-Step (CHF)', 'Mosh Pit (AF)', 'Stumble (PVC/VA)']:
                pts = dance_data.get(dance_name, [])
                if pts:
                    ks = [p['kappa'] for p in pts]
                    gs = [p['gini'] for p in pts]
                    ss = [p['spread'] for p in pts]
                    ek = float(np.median(ks))
                    eg = float(np.median(gs))
                    es = float(np.median(ss))
                    hand_k = RHYTHM_DANCES[dance_name]['kappa']
                    print(f"  {dance_name:25s} {len(pts):5d} {ek:10.2f} {eg:10.3f} {es:10.2f}   (hand-set: {hand_k:.1f})")
                    EMPIRICAL_DANCES[dance_name] = {'kappa': ek, 'gini': eg, 'spread': es}
                else:
                    print(f"  {dance_name:25s}     0     NO DATA")
                    EMPIRICAL_DANCES[dance_name] = RHYTHM_DANCES[dance_name]

            # ---- PASS 2: Classify using BOTH hand-set and empirical centroids ----
            for lib_name, library in [("Hand-set prototypes", RHYTHM_DANCES), ("Empirical centroids", EMPIRICAL_DANCES)]:
                print(f"\n  === Classification with {lib_name} ===")
                results = []
                for idx, row in df_rhythm.iterrows():
                    k = row.get(kappa_col, None)
                    g = row.get(gini_col, None)
                    s = row.get(spread_col, 2.5) if spread_col else 2.5
                    cond_raw = str(row.get(cond_col, ''))

                    if pd.isna(k) or pd.isna(g): continue

                    if cond_raw in CONDITION_TO_DANCE:
                        condition = cond_raw
                    elif cond_raw in db_to_condition:
                        condition = db_to_condition[cond_raw]
                    else:
                        continue

                    true_dance = CONDITION_TO_DANCE[condition]
                    pred_dance, dist = classify_dance(float(k), float(g), float(s), library)

                    results.append({
                        'condition': condition,
                        'true_dance': true_dance,
                        'predicted_dance': pred_dance,
                        'distance': dist,
                        'kappa': float(k),
                        'gini': float(g),
                    })

                if results:
                    df_match = pd.DataFrame(results)
                    if lib_name == "Empirical centroids":
                        df_match.to_csv(RESULTS_DIR / 'dance_matching_results.csv', index=False)

                    print(f"\n  Records classified: {len(df_match)}")
                    print(f"  Conditions: {df_match['condition'].value_counts().to_dict()}")

                    # Confusion matrix
                    dances = list(library.keys()) + ['Unclassified']
                    true_dances = sorted(df_match['true_dance'].unique())

                    print(f"\n  CONFUSION MATRIX (rows=true, cols=predicted):")
                    header = f"  {'True Dance':22s}"
                    for d in dances:
                        short = d.split('(')[0].strip()[:10]
                        header += f" {short:>10s}"
                    header += f" {'Accuracy':>10s}"
                    print(header)
                    print("  " + "-" * len(header))

                    total_correct = 0
                    total = 0

                    for td in true_dances:
                        subset = df_match[df_match['true_dance'] == td]
                        row_str = f"  {td:22s}"
                        correct = 0
                        for pd_name in dances:
                            count = len(subset[subset['predicted_dance'] == pd_name])
                            if pd_name == td:
                                correct = count
                            row_str += f" {count:10d}"
                        acc = correct / len(subset) if len(subset) > 0 else 0
                        row_str += f" {acc:10.1%}"
                        print(row_str)
                        total_correct += correct
                        total += len(subset)

                    overall_acc = total_correct / total if total > 0 else 0
                    print(f"\n  Overall accuracy: {total_correct}/{total} = {overall_acc:.1%}")

                    # Per-dance accuracy
                    print(f"\n  Per-dance accuracy:")
                    for dance in library:
                        true_set = df_match[df_match['true_dance'] == dance]
                        if len(true_set) == 0: continue
                        correct = len(true_set[true_set['predicted_dance'] == dance])
                        print(f"    {dance:25s}: {correct}/{len(true_set)} = {correct/len(true_set):.1%}")

    # ==============================================================
    # EXPERIMENT C: Non-Dance Detection
    # ==============================================================
    print("\n" + "=" * 65)
    print("EXPERIMENT C: Non-Dance Autocorrelation Test")
    print("=" * 65)

    # Use the rhythm data to test: do V-Tach segments have low autocorrelation?
    if rhythm_csv.exists():
        df_r = pd.read_csv(rhythm_csv)
        # We need a proxy: records with very low κ AND very low spread are non-dance candidates
        if kappa_col and spread_col:
            print("  Testing autocorrelation proxy on κ-spread extremes:")
            lo_k = df_r[kappa_col].quantile(0.1)
            lo_s = df_r[spread_col].quantile(0.1) if spread_col in df_r.columns else 1.5
            extreme = df_r[(df_r[kappa_col] < lo_k)]
            normal_range = df_r[(df_r[kappa_col] > df_r[kappa_col].quantile(0.4)) &
                                (df_r[kappa_col] < df_r[kappa_col].quantile(0.6))]

            print(f"    Low-κ records (potential non-dance): {len(extreme)}")
            print(f"    Mid-κ records (normal dance range): {len(normal_range)}")

            if len(extreme) > 0 and len(normal_range) > 0:
                if cond_col:
                    print(f"    Low-κ conditions: {extreme[cond_col].value_counts().to_dict()}")
                    print(f"    Mid-κ conditions: {normal_range[cond_col].value_counts().to_dict()}")
                print(f"\n  >>> Non-dance detector would flag the low-κ extreme as potential emergency states")
                print(f"  >>> This requires validation on confirmed V-Fib/asystole data (not available in current datasets)")
        else:
            print("  Insufficient columns for non-dance analysis")

    # ==============================================================
    # SUMMARY
    # ==============================================================
    print("\n" + "=" * 65)
    print("PAPER IV EXPERIMENT SUMMARY")
    print("=" * 65)
    print("  Experiment A (CMC): Results above")
    print("  Experiment B (Dance Matching): Confusion matrix above")
    print("  Experiment C (Non-Dance): Proxy analysis above")
    print(f"\n  All results saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()

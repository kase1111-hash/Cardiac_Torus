"""
15_heart_sounds.py — Heart Sound Torus Analysis
Cardiac Torus Pipeline — Paper III

THE ACOUSTIC SUBSTRATE:
  Paper I: electrical (RR intervals from ECG/PPG)
  Paper II: mechanical-visual (pixel motion from echo)
  Paper III: mechanical-acoustic (sound pressure from stethoscope)

Maps consecutive heart sound features onto T² and computes
geodesic curvature. Tests whether murmurs, gallops, and
abnormal sounds produce distinct geometric signatures.

DATA SOURCE:
  PhysioNet/CinC 2016 Challenge — Heart Sound Database
  ~3,000 recordings from multiple sites
  Labels: normal (-1) vs abnormal (1)
  WAV format, variable sample rates

SIGNAL EXTRACTION:
  1. Envelope extraction (Hilbert transform)
  2. Beat segmentation (peak detection on envelope)  
  3. Per-beat features: S1 amplitude, S2 amplitude, 
     S1-S2 interval, systolic energy, diastolic energy
  4. Consecutive beat-pairs → torus mapping
  5. Curvature analysis

REQUIREMENTS:
  pip install wfdb numpy scipy pandas matplotlib tqdm
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats, signal as sig
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, FIGURES_DIR, DPI, FIG_FORMAT, RR_MIN_MS, RR_MAX_MS

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    import wfdb
except ImportError:
    print("Install wfdb: pip install wfdb")
    sys.exit(1)

DATA_DIR = Path(__file__).parent / "data" / "heart_sounds"

# =====================================================================
# TORUS FUNCTIONS
# =====================================================================
def to_angle(value, vmin, vmax):
    clipped = np.clip(value, vmin, vmax)
    if vmax - vmin < 1e-10:
        return np.pi
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


# =====================================================================
# HEART SOUND PROCESSING
# =====================================================================

def extract_envelope(audio, fs, cutoff=50):
    """Extract amplitude envelope using Hilbert transform + lowpass."""
    # Bandpass filter to heart sound range (25-400 Hz)
    nyq = fs / 2
    low = max(25 / nyq, 0.01)
    high = min(400 / nyq, 0.99)
    
    try:
        b, a = sig.butter(4, [low, high], btype='band')
        filtered = sig.filtfilt(b, a, audio)
    except:
        filtered = audio
    
    # Hilbert envelope
    analytic = sig.hilbert(filtered)
    envelope = np.abs(analytic)
    
    # Smooth envelope
    lp_cutoff = min(cutoff / nyq, 0.99)
    try:
        b2, a2 = sig.butter(2, lp_cutoff, btype='low')
        envelope = sig.filtfilt(b2, a2, envelope)
    except:
        # Fallback: moving average
        win = max(1, int(fs / cutoff))
        envelope = np.convolve(envelope, np.ones(win)/win, mode='same')
    
    return envelope, filtered


def segment_beats(envelope, fs, min_bpm=40, max_bpm=200):
    """Find heart beats from envelope peaks."""
    min_distance = int(fs * 60 / max_bpm)
    max_distance = int(fs * 60 / min_bpm)
    
    # Find peaks (S1 candidates)
    height = np.percentile(envelope, 60)
    peaks, props = sig.find_peaks(envelope, 
                                   distance=min_distance,
                                   height=height)
    
    if len(peaks) < 4:
        # Try with lower threshold
        height = np.percentile(envelope, 40)
        peaks, props = sig.find_peaks(envelope, distance=min_distance, height=height)
    
    return peaks


def extract_beat_features(audio, envelope, peaks, fs):
    """
    Extract per-beat features for torus mapping.
    
    For each beat (S1 peak), compute:
      - S1 amplitude (peak height)
      - Systolic energy (energy between this S1 and next S2)
      - S2 amplitude (secondary peak)
      - Diastolic energy (energy between S2 and next S1)
      - Beat interval (time to next S1)
      - S1-S2 interval
    """
    features = []
    
    for i in range(len(peaks) - 1):
        s1_idx = peaks[i]
        next_s1_idx = peaks[i + 1]
        
        beat_interval_ms = 1000.0 * (next_s1_idx - s1_idx) / fs
        
        # Skip physiologically implausible intervals
        if beat_interval_ms < 300 or beat_interval_ms > 1500:
            continue
        
        s1_amp = float(envelope[s1_idx])
        
        # Find S2 as secondary peak in the middle third of the interval
        beat_len = next_s1_idx - s1_idx
        s2_search_start = s1_idx + int(beat_len * 0.25)
        s2_search_end = s1_idx + int(beat_len * 0.6)
        
        if s2_search_end > s2_search_start + 5:
            s2_region = envelope[s2_search_start:s2_search_end]
            s2_local_idx = np.argmax(s2_region)
            s2_idx = s2_search_start + s2_local_idx
            s2_amp = float(envelope[s2_idx])
            s1_s2_interval = 1000.0 * (s2_idx - s1_idx) / fs
        else:
            s2_amp = s1_amp * 0.5
            s1_s2_interval = beat_interval_ms * 0.35
            s2_idx = s1_idx + int(beat_len * 0.35)
        
        # Energy in systolic and diastolic periods
        systolic_segment = envelope[s1_idx:s2_idx]
        diastolic_segment = envelope[s2_idx:next_s1_idx]
        
        systolic_energy = float(np.sum(systolic_segment**2)) / max(1, len(systolic_segment))
        diastolic_energy = float(np.sum(diastolic_segment**2)) / max(1, len(diastolic_segment))
        
        # S1/S2 ratio
        s1_s2_ratio = s1_amp / s2_amp if s2_amp > 1e-10 else 10.0
        
        # Spectral centroid of the beat
        beat_audio = audio[s1_idx:next_s1_idx]
        if len(beat_audio) > 32:
            fft = np.abs(np.fft.rfft(beat_audio))
            freqs = np.fft.rfftfreq(len(beat_audio), 1/fs)
            if np.sum(fft) > 0:
                spectral_centroid = float(np.sum(freqs * fft) / np.sum(fft))
            else:
                spectral_centroid = 100.0
        else:
            spectral_centroid = 100.0
        
        features.append({
            'beat_idx': i,
            's1_amp': round(s1_amp, 6),
            's2_amp': round(s2_amp, 6),
            's1_s2_ratio': round(s1_s2_ratio, 4),
            's1_s2_interval': round(s1_s2_interval, 2),
            'beat_interval': round(beat_interval_ms, 2),
            'systolic_energy': round(systolic_energy, 6),
            'diastolic_energy': round(diastolic_energy, 6),
            'spectral_centroid': round(spectral_centroid, 2),
        })
    
    return features


def compute_sound_torus(features, pair_type='interval_s1amp'):
    """
    Map consecutive beat features onto T² and compute curvature.
    
    Pair types:
      'interval_s1amp': (beat_interval, S1_amplitude) — rhythm + loudness
      'interval_ratio': (beat_interval, S1/S2 ratio) — rhythm + valve balance
      'systolic_diastolic': (systolic_energy, diastolic_energy) — energy balance
      'spectral_interval': (spectral_centroid, beat_interval) — timbre + rhythm
    """
    if len(features) < 10:
        return None
    
    # Extract the pair values
    if pair_type == 'interval_s1amp':
        vals1 = [f['beat_interval'] for f in features]
        vals2 = [f['s1_amp'] for f in features]
    elif pair_type == 'interval_ratio':
        vals1 = [f['beat_interval'] for f in features]
        vals2 = [f['s1_s2_ratio'] for f in features]
    elif pair_type == 'systolic_diastolic':
        vals1 = [f['systolic_energy'] for f in features]
        vals2 = [f['diastolic_energy'] for f in features]
    elif pair_type == 'spectral_interval':
        vals1 = [f['spectral_centroid'] for f in features]
        vals2 = [f['beat_interval'] for f in features]
    else:
        return None
    
    v1 = np.array(vals1)
    v2 = np.array(vals2)
    
    # Percentile normalization
    v1_min, v1_max = np.percentile(v1, 2), np.percentile(v1, 98)
    v2_min, v2_max = np.percentile(v2, 2), np.percentile(v2, 98)
    
    if v1_max - v1_min < 1e-10 or v2_max - v2_min < 1e-10:
        return None
    
    # Build consecutive pairs
    n = len(v1) - 1
    if n < 8:
        return None
    
    # Torus A: (feature_t, feature_{t+1}) for the primary signal
    theta1 = np.array([to_angle(v1[i], v1_min, v1_max) for i in range(n)])
    theta2 = np.array([to_angle(v1[i+1], v1_min, v1_max) for i in range(n)])
    
    # Also compute Torus B: (signal1, signal2) cross-feature
    phi1 = np.array([to_angle(v1[i], v1_min, v1_max) for i in range(n)])
    phi2 = np.array([to_angle(v2[i], v2_min, v2_max) for i in range(n)])
    
    # Curvature on Torus A (consecutive same-feature)
    kappa_a = np.zeros(n)
    for i in range(1, n-1):
        kappa_a[i] = menger_curvature_torus(
            (theta1[i-1], theta2[i-1]),
            (theta1[i], theta2[i]),
            (theta1[i+1], theta2[i+1]))
    
    # Curvature on Torus B (cross-feature)
    kappa_b = np.zeros(n)
    for i in range(1, n-1):
        kappa_b[i] = menger_curvature_torus(
            (phi1[i-1], phi2[i-1]),
            (phi1[i], phi2[i]),
            (phi1[i+1], phi2[i+1]))
    
    valid_a = kappa_a[kappa_a > 0]
    valid_b = kappa_b[kappa_b > 0]
    
    if len(valid_a) < 5:
        return None
    
    result = {
        'kappa_A_median': round(float(np.median(valid_a)), 4),
        'kappa_A_mean': round(float(np.mean(valid_a)), 4),
        'gini_A': round(gini_coefficient(valid_a), 4),
        'n_beats': n,
    }
    
    if len(valid_b) >= 5:
        result['kappa_B_median'] = round(float(np.median(valid_b)), 4)
        result['kappa_B_mean'] = round(float(np.mean(valid_b)), 4)
        result['gini_B'] = round(gini_coefficient(valid_b), 4)
    
    # Spread and speed
    spread = np.sqrt(np.std(theta1)**2 + np.std(theta2)**2)
    result['spread_A'] = round(float(spread), 4)
    
    return result


# =====================================================================
# DATA DOWNLOAD
# =====================================================================

def download_heart_sounds(data_dir):
    """Download PhysioNet heart sound databases."""
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # The CinC 2016 challenge data has multiple training sets
    databases = [
        ('training-a', 'challenge-2016/training-a'),
        ('training-b', 'challenge-2016/training-b'),
        ('training-c', 'challenge-2016/training-c'),
        ('training-d', 'challenge-2016/training-d'),
        ('training-e', 'challenge-2016/training-e'),
        ('training-f', 'challenge-2016/training-f'),
    ]
    
    downloaded = 0
    for name, physionet_path in databases:
        db_dir = data_dir / name
        if db_dir.exists() and any(db_dir.glob('*.wav')):
            n_wav = len(list(db_dir.glob('*.wav')))
            print(f"  {name}: {n_wav} files (already downloaded)")
            downloaded += n_wav
            continue
        
        db_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading {name}...")
        
        try:
            wfdb.dl_database(physionet_path, str(db_dir))
            n_wav = len(list(db_dir.glob('*.wav')))
            print(f"  {name}: {n_wav} files downloaded")
            downloaded += n_wav
        except Exception as e:
            print(f"  {name}: download failed ({e})")
            # Try alternative: direct download of individual records
            try:
                record_list = wfdb.get_record_list(physionet_path)
                for rec in record_list[:5]:  # test with first 5
                    wfdb.dl_database(physionet_path, str(db_dir), records=[rec])
                n_wav = len(list(db_dir.glob('*.wav')))
                print(f"  {name}: {n_wav} files (partial)")
                downloaded += n_wav
            except:
                print(f"  {name}: skipped")
    
    return downloaded


def load_recordings(data_dir):
    """Load heart sound recordings with labels from .hea files."""
    recordings = []
    
    for subset_dir in sorted(data_dir.iterdir()):
        if not subset_dir.is_dir():
            continue
        
        for wav_file in sorted(subset_dir.glob('*.wav')):
            rec_name = wav_file.stem
            condition = 'Unknown'
            label = None
            
            # Read label from .hea file (comment line: # Normal or # Abnormal)
            hea_file = wav_file.with_suffix('.hea')
            if hea_file.exists():
                try:
                    with open(hea_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line == '# Normal':
                                condition = 'Normal'; label = -1; break
                            elif line == '# Abnormal':
                                condition = 'Abnormal'; label = 1; break
                except:
                    pass
            
            recordings.append({
                'path': str(wav_file),
                'name': rec_name,
                'subset': subset_dir.name,
                'label': label,
                'condition': condition,
            })
    
    return recordings


def read_wav(filepath):
    """Read a WAV file and return audio array + sample rate."""
    try:
        import wave
        with wave.open(filepath, 'r') as wf:
            fs = wf.getframerate()
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw = wf.readframes(n_frames)
        
        if sampwidth == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(float)
        elif sampwidth == 4:
            audio = np.frombuffer(raw, dtype=np.int32).astype(float)
        elif sampwidth == 1:
            audio = np.frombuffer(raw, dtype=np.uint8).astype(float) - 128
        else:
            return None, None
        
        if n_channels > 1:
            audio = audio[::n_channels]  # take first channel
        
        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        return audio, fs
    except Exception as e:
        return None, None


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=str(DATA_DIR))
    parser.add_argument('--max_recordings', type=int, default=None)
    parser.add_argument('--skip_download', action='store_true')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    print("=" * 65)
    print("Step 15: Heart Sound Torus Analysis")
    print("Cardiac Torus Pipeline \u2014 Paper III")
    print("The Acoustic Substrate")
    print("=" * 65)
    
    # Download if needed
    if not args.skip_download:
        print("\nChecking/downloading heart sound data...")
        n_files = download_heart_sounds(data_dir)
        print(f"Total audio files: {n_files}")
    
    # Load recordings
    print("\nLoading recordings...")
    recordings = load_recordings(data_dir)
    print(f"Found {len(recordings)} recordings")
    
    # Filter to labeled only
    labeled = [r for r in recordings if r['condition'] != 'Unknown']
    print(f"Labeled recordings: {len(labeled)}")
    for cond in ['Normal', 'Abnormal']:
        n = sum(1 for r in labeled if r['condition'] == cond)
        print(f"  {cond}: {n}")
    
    if args.max_recordings:
        labeled = labeled[:args.max_recordings]
    
    # Process recordings
    print(f"\nProcessing {len(labeled)} recordings...")
    all_results = []
    errors = 0
    
    for rec in tqdm(labeled, desc="Heart sounds"):
        audio, fs = read_wav(rec['path'])
        if audio is None or fs is None:
            errors += 1
            continue
        
        # Skip very short recordings
        duration = len(audio) / fs
        if duration < 3:
            errors += 1
            continue
        
        try:
            # Extract envelope
            envelope, filtered = extract_envelope(audio, fs)
            
            # Segment beats
            peaks = segment_beats(envelope, fs)
            
            if len(peaks) < 5:
                errors += 1
                continue
            
            # Extract per-beat features
            features = extract_beat_features(filtered, envelope, peaks, fs)
            
            if len(features) < 5:
                errors += 1
                continue
            
            # Compute torus features for multiple pair types
            result = {
                'name': rec['name'],
                'subset': rec['subset'],
                'condition': rec['condition'],
                'label': rec['label'],
                'duration': round(duration, 2),
                'n_beats_raw': len(peaks),
                'n_beats_valid': len(features),
                'fs': fs,
                'mean_hr': round(60000 / np.mean([f['beat_interval'] for f in features]), 1),
                'mean_s1_amp': round(np.mean([f['s1_amp'] for f in features]), 6),
                'mean_s2_amp': round(np.mean([f['s2_amp'] for f in features]), 6),
                'mean_s1s2_ratio': round(np.mean([f['s1_s2_ratio'] for f in features]), 4),
                'mean_spectral_centroid': round(np.mean([f['spectral_centroid'] for f in features]), 2),
            }
            
            # Torus analysis for each pair type
            for ptype in ['interval_s1amp', 'interval_ratio', 'systolic_diastolic', 'spectral_interval']:
                torus = compute_sound_torus(features, ptype)
                if torus:
                    for k, v in torus.items():
                        result[f'{ptype}_{k}'] = v
            
            all_results.append(result)
        
        except Exception as e:
            errors += 1
    
    print(f"\nProcessed: {len(all_results)}, Errors: {errors}")
    
    if not all_results:
        print("No recordings processed!")
        sys.exit(1)
    
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / 'heart_sound_torus_results.csv', index=False)
    
    # ---- ANALYSIS ----
    print("\n" + "=" * 65)
    print("HEART SOUND TORUS ANALYSIS")
    print("=" * 65)
    print(f"Recordings: {len(df)}")
    print(f"  Normal: {len(df[df['condition']=='Normal'])}")
    print(f"  Abnormal: {len(df[df['condition']=='Abnormal'])}")
    
    # Compare Normal vs Abnormal for each torus metric
    normal = df[df['condition'] == 'Normal']
    abnormal = df[df['condition'] == 'Abnormal']
    
    print(f"\n{'Metric':45s} {'Normal':>10s} {'Abnormal':>10s} {'r':>8s} {'p':>12s}")
    print("-" * 90)
    
    comparison_results = {}
    
    torus_cols = [c for c in df.columns if 'kappa' in c or 'gini' in c or 'spread' in c]
    basic_cols = ['mean_hr', 'mean_s1_amp', 'mean_s2_amp', 'mean_s1s2_ratio', 'mean_spectral_centroid']
    
    for col in basic_cols + torus_cols:
        if col not in df.columns:
            continue
        
        n_vals = normal[col].dropna()
        a_vals = abnormal[col].dropna()
        
        if len(n_vals) < 5 or len(a_vals) < 5:
            continue
        
        U, p = stats.mannwhitneyu(n_vals, a_vals, alternative='two-sided')
        r = 1 - 2*U/(len(n_vals)*len(a_vals))
        
        sig_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        tag = " TORUS" if col in torus_cols else ""
        
        print(f"  {col:43s} {np.median(n_vals):10.4f} {np.median(a_vals):10.4f} {r:+8.3f} {p:12.3e} {sig_str}{tag}")
        
        comparison_results[col] = {
            'normal_median': round(float(np.median(n_vals)), 4),
            'abnormal_median': round(float(np.median(a_vals)), 4),
            'r': round(float(r), 4),
            'p': float(p),
        }
    
    with open(RESULTS_DIR / 'heart_sound_comparisons.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # ---- FIGURE ----
    print("\nGenerating figures...")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Panel A: κ comparison (interval_s1amp Torus A)
    plot_metrics = [
        ('interval_s1amp_kappa_A_median', '\u03BA median\n(interval torus)'),
        ('interval_s1amp_gini_A', 'Gini\n(interval torus)'),
        ('interval_ratio_kappa_B_median', '\u03BA median\n(S1/S2 ratio torus)'),
        ('interval_ratio_gini_B', 'Gini\n(S1/S2 ratio torus)'),
        ('systolic_diastolic_kappa_B_median', '\u03BA median\n(energy torus)'),
        ('mean_s1s2_ratio', 'S1/S2 ratio\n(raw feature)'),
    ]
    
    colors = {'Normal': '#4FC3F7', 'Abnormal': '#FF7043'}
    
    for idx, (col, label) in enumerate(plot_metrics):
        row, c2 = divmod(idx, 3)
        ax = axes[row, c2]
        
        if col not in df.columns:
            ax.set_title(f'{label}: n/a')
            continue
        
        data_n = normal[col].dropna()
        data_a = abnormal[col].dropna()
        
        if len(data_n) > 0 and len(data_a) > 0:
            bp = ax.boxplot([data_n, data_a], widths=0.6, patch_artist=True,
                           showfliers=False,
                           medianprops=dict(color='black', linewidth=2))
            bp['boxes'][0].set_facecolor(colors['Normal']); bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][1].set_facecolor(colors['Abnormal']); bp['boxes'][1].set_alpha(0.7)
            ax.set_xticklabels([f'Normal\n(n={len(data_n)})', f'Abnormal\n(n={len(data_a)})'])
            
            U, p = stats.mannwhitneyu(data_n, data_a, alternative='two-sided')
            r = 1 - 2*U/(len(data_n)*len(data_a))
            sig_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.set_title(f'{label}\nr={r:+.3f} {sig_str}', fontsize=10)
        
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Heart Sound Torus: Normal vs Abnormal',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    path = FIGURES_DIR / f'figS1_heart_sounds.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    
    # Summary
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    
    # Find best torus metric
    best_r = 0
    best_metric = None
    for col in torus_cols:
        if col in comparison_results:
            if abs(comparison_results[col]['r']) > abs(best_r):
                best_r = comparison_results[col]['r']
                best_metric = col
    
    if best_metric:
        print(f"\n  Best torus metric: {best_metric}")
        print(f"  Effect size: r = {best_r:+.3f}")
        print(f"  Normal median: {comparison_results[best_metric]['normal_median']:.4f}")
        print(f"  Abnormal median: {comparison_results[best_metric]['abnormal_median']:.4f}")
        
        if abs(best_r) > 0.2:
            print(f"\n  >>> SIGNAL DETECTED: Torus separates normal from abnormal heart sounds")
        elif abs(best_r) > 0.1:
            print(f"\n  >>> WEAK SIGNAL: Modest separation, needs investigation")
        else:
            print(f"\n  >>> NO CLEAR SIGNAL at this level of analysis")
    
    print(f"\nAll results: {RESULTS_DIR / 'heart_sound_torus_results.csv'}")


if __name__ == '__main__':
    main()

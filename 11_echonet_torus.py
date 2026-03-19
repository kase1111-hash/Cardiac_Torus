"""
11_echonet_torus.py — EchoNet-Dynamic Torus Analysis
True North Research | Cardiac Torus Pipeline

Validates the torus-curvature framework on 10,030 real echocardiograms
from Stanford's EchoNet-Dynamic dataset.

APPROACH:
  For each apical-4-chamber video:
  1. Load AVI frames as grayscale (112×112)
  2. Extract cardiac motion signal via two complementary methods:
     a) LV cavity brightness: mean pixel intensity in the central ROI
        where the LV cavity is (tracks filling/emptying)
     b) Virtual M-mode: pixel intensity along a vertical scanline through
        the mitral valve region, tracked per frame
  3. Map consecutive motion values onto phase-space torus:
     θ₁ = normalize(signal[t])
     θ₂ = normalize(signal[t+1])
  4. Compute geodesic curvature on T²
  5. Correlate curvature features with ejection fraction (EF)

HYPOTHESIS:
  Low EF (heart failure) → weak contraction → small signal excursion →
  tight orbit on torus → HIGH curvature (same as CHF in RR torus)
  
  Normal EF → vigorous contraction → large excursion →
  wider orbit → moderate curvature
  
  If κ correlates negatively with EF, the valve/wall motion torus
  framework is validated on real data — 10,000 patients.

REQUIREMENTS:
  pip install opencv-python numpy pandas scipy matplotlib tqdm

USAGE:
  python 11_echonet_torus.py --data_dir /path/to/EchoNet-Dynamic
  
  Or edit ECHONET_DIR below to point to your download location.
"""

import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, FIGURES_DIR, DPI, FIG_FORMAT

try:
    import cv2
except ImportError:
    print("Install opencv: pip install opencv-python")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        total = kwargs.get('total', None)
        desc = kwargs.get('desc', '')
        for i, item in enumerate(iterable):
            if total and (i % max(1, total//20) == 0):
                print(f"  {desc} {i}/{total} ({100*i/total:.0f}%)")
            yield item


# =====================================================================
# CONFIGURATION
# =====================================================================

# Set this to your EchoNet-Dynamic download location, or use --data_dir
ECHONET_DIR = Path(r"G:\EchoNet-Dynamic")  # Default for Windows

# ROI for LV cavity brightness extraction (in 112×112 frame)
# Apical 4-chamber: LV cavity is roughly in the upper-center-left
# These are approximate and work across most A4C orientations
LV_ROI = {
    'y_start': 20, 'y_end': 70,   # vertical range (apex to base)
    'x_start': 35, 'x_end': 80,   # horizontal range (septum to lateral wall)
}

# Virtual M-mode scanline position
MMODE_X = 56  # vertical scanline at center of 112-wide frame

# Processing limits
MAX_VIDEOS = None   # Set to e.g. 1000 for testing, None for all
SKIP_ERRORS = True  # Continue on individual video failures


# =====================================================================
# TORUS FUNCTIONS (reused)
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


# =====================================================================
# VIDEO PROCESSING
# =====================================================================

def extract_motion_signals(video_path: str) -> dict | None:
    """
    Extract cardiac motion signals from an echocardiogram video.
    
    Returns dict with:
      lv_brightness: mean pixel intensity in LV ROI per frame
      mmode_profile: pixel intensity along vertical scanline per frame
      valve_position: tracked bright-structure position along scanline
      n_frames: total frames
      fps: frame rate
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if n_frames < 20 or fps <= 0:
        cap.release()
        return None
    
    lv_brightness = []
    valve_positions = []
    frame_diffs = []
    prev_frame = None
    
    roi = LV_ROI
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Method 1: LV cavity mean brightness
        roi_pixels = gray[roi['y_start']:roi['y_end'], roi['x_start']:roi['x_end']]
        lv_brightness.append(float(np.mean(roi_pixels)))
        
        # Method 2: Virtual M-mode — track bright structure along scanline
        scanline = gray[:, MMODE_X].astype(float)
        # Find position of the brightest region (valve/wall echo)
        # Use center-of-mass of top 20% brightest pixels
        threshold = np.percentile(scanline, 80)
        bright_mask = scanline >= threshold
        bright_indices = np.where(bright_mask)[0]
        if len(bright_indices) > 0:
            weights = scanline[bright_mask]
            valve_pos = np.average(bright_indices, weights=weights)
        else:
            valve_pos = 56.0  # center fallback
        valve_positions.append(valve_pos)
        
        # Method 3: Frame-to-frame motion energy
        if prev_frame is not None:
            diff = np.mean(np.abs(gray.astype(float) - prev_frame.astype(float)))
            frame_diffs.append(diff)
        prev_frame = gray.copy()
    
    cap.release()
    
    if len(lv_brightness) < 20:
        return None
    
    return {
        'lv_brightness': np.array(lv_brightness),
        'valve_position': np.array(valve_positions),
        'frame_diffs': np.array(frame_diffs) if frame_diffs else np.zeros(1),
        'n_frames': len(lv_brightness),
        'fps': fps,
    }


def compute_motion_torus_features(signal: np.ndarray, signal_name: str) -> dict | None:
    """
    Map a 1D motion signal onto the consecutive-sample torus and compute curvature.
    
    Torus mapping: (signal[t], signal[t+1]) → (θ₁, θ₂)
    This is the wall-motion analog of the (RR_pre, RR_post) beat-pair torus.
    """
    if len(signal) < 30:
        return None
    
    # Smooth slightly to reduce ultrasound speckle noise
    kernel_size = 3
    signal_smooth = np.convolve(signal, np.ones(kernel_size)/kernel_size, mode='valid')
    
    # Consecutive pairs
    s_pre = signal_smooth[:-1]
    s_post = signal_smooth[1:]
    
    n = len(s_pre)
    if n < 20:
        return None
    
    # Map to torus using signal range
    sig_min = np.percentile(signal_smooth, 2)
    sig_max = np.percentile(signal_smooth, 98)
    if sig_max - sig_min < 1e-6:
        return None  # no variation = dead signal
    
    theta1 = np.array([to_angle(s, sig_min, sig_max) for s in s_pre])
    theta2 = np.array([to_angle(s, sig_min, sig_max) for s in s_post])
    
    # Compute curvature
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
    
    # Velocity and speed on torus
    dt1 = np.diff(theta1)
    dt1 = np.where(dt1 > np.pi, dt1 - 2*np.pi, dt1)
    dt1 = np.where(dt1 < -np.pi, dt1 + 2*np.pi, dt1)
    dt2 = np.diff(theta2)
    dt2 = np.where(dt2 > np.pi, dt2 - 2*np.pi, dt2)
    dt2 = np.where(dt2 < -np.pi, dt2 + 2*np.pi, dt2)
    speed = np.sqrt(dt1**2 + dt2**2)
    
    spread = np.sqrt(np.std(theta1)**2 + np.std(theta2)**2)
    
    prefix = signal_name
    return {
        f'{prefix}_kappa_median': round(float(np.median(valid_kappa)), 4),
        f'{prefix}_kappa_mean': round(float(np.mean(valid_kappa)), 4),
        f'{prefix}_kappa_std': round(float(np.std(valid_kappa)), 4),
        f'{prefix}_kappa_p95': round(float(np.percentile(valid_kappa, 95)), 4),
        f'{prefix}_kappa_cv': round(float(np.std(valid_kappa)/np.mean(valid_kappa)), 4),
        f'{prefix}_gini': round(gini, 4),
        f'{prefix}_spread': round(float(spread), 4),
        f'{prefix}_speed_mean': round(float(np.mean(speed)), 4),
        f'{prefix}_speed_cv': round(float(np.std(speed)/np.mean(speed)), 4) if np.mean(speed) > 0 else 0,
        f'{prefix}_signal_range': round(float(sig_max - sig_min), 4),
        f'{prefix}_signal_cv': round(float(np.std(signal_smooth)/np.mean(signal_smooth)), 4) if np.mean(signal_smooth) > 0 else 0,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='EchoNet-Dynamic Torus Analysis')
    parser.add_argument('--data_dir', type=str, default=str(ECHONET_DIR),
                        help='Path to EchoNet-Dynamic directory')
    parser.add_argument('--max_videos', type=int, default=MAX_VIDEOS,
                        help='Max videos to process (None = all)')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    print("=" * 65)
    print("Step 11: EchoNet-Dynamic Torus Analysis")
    print("True North Research | Cardiac Torus Pipeline")
    print("=" * 65)
    
    # ---- Load FileList.csv ----
    filelist_path = data_dir / "FileList.csv"
    if not filelist_path.exists():
        print(f"ERROR: {filelist_path} not found")
        print(f"Set --data_dir to your EchoNet-Dynamic download location")
        sys.exit(1)
    
    df_meta = pd.read_csv(filelist_path)
    print(f"\nLoaded metadata for {len(df_meta)} videos")
    print(f"Columns: {list(df_meta.columns)}")
    
    # Standardize column names (handle case variations)
    df_meta.columns = [c.strip() for c in df_meta.columns]
    
    # Find EF column
    ef_col = None
    for candidate in ['EF', 'ef', 'EjectionFraction', 'Ejection Fraction']:
        if candidate in df_meta.columns:
            ef_col = candidate
            break
    
    if ef_col is None:
        print(f"WARNING: No EF column found. Available columns: {list(df_meta.columns)}")
        print("Will compute torus features without EF correlation.")
    else:
        print(f"EF column: '{ef_col}'")
        print(f"EF range: {df_meta[ef_col].min():.1f} - {df_meta[ef_col].max():.1f}%")
        print(f"EF mean: {df_meta[ef_col].mean():.1f}%")
    
    # Find filename column
    fn_col = None
    for candidate in ['FileName', 'filename', 'Video', 'video']:
        if candidate in df_meta.columns:
            fn_col = candidate
            break
    
    if fn_col is None:
        print(f"ERROR: No filename column found. Columns: {list(df_meta.columns)}")
        sys.exit(1)
    
    # Video directory
    video_dir = data_dir / "Videos"
    if not video_dir.exists():
        # Try alternative names
        for alt in ["videos", "a4c-video-dir"]:
            if (data_dir / alt).exists():
                video_dir = data_dir / alt
                break
        else:
            print(f"ERROR: Videos directory not found in {data_dir}")
            sys.exit(1)
    
    print(f"Video directory: {video_dir}")
    
    # ---- Process videos ----
    if args.max_videos:
        df_process = df_meta.head(args.max_videos)
    else:
        df_process = df_meta
    
    print(f"\nProcessing {len(df_process)} videos...")
    
    all_results = []
    errors = 0
    
    for idx, row in tqdm(df_process.iterrows(), total=len(df_process), desc="Videos"):
        filename = row[fn_col]
        # Ensure .avi extension
        if not filename.endswith('.avi'):
            filename = filename + '.avi'
        
        video_path = video_dir / filename
        if not video_path.exists():
            errors += 1
            continue
        
        try:
            # Extract motion signals
            signals = extract_motion_signals(str(video_path))
            if signals is None:
                errors += 1
                continue
            
            result = {
                'filename': row[fn_col],
                'n_frames': signals['n_frames'],
                'fps': signals['fps'],
            }
            
            # Add EF if available
            if ef_col:
                result['ef'] = float(row[ef_col])
            
            # Add other metadata columns
            for col in ['ESV', 'EDV', 'Split', 'NumberOfFrames', 'FPS']:
                if col in row.index:
                    result[col.lower()] = row[col]
            
            # Compute torus features for each signal type
            lv_features = compute_motion_torus_features(
                signals['lv_brightness'], 'lv')
            if lv_features:
                result.update(lv_features)
            
            valve_features = compute_motion_torus_features(
                signals['valve_position'], 'valve')
            if valve_features:
                result.update(valve_features)
            
            if len(signals['frame_diffs']) > 20:
                motion_features = compute_motion_torus_features(
                    signals['frame_diffs'], 'motion')
                if motion_features:
                    result.update(motion_features)
            
            all_results.append(result)
            
        except Exception as e:
            if not SKIP_ERRORS:
                raise
            errors += 1
    
    print(f"\nProcessed: {len(all_results)}, Errors: {errors}")
    
    if len(all_results) == 0:
        print("ERROR: No videos processed successfully!")
        sys.exit(1)
    
    # ---- Save results ----
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / 'echonet_torus_results.csv', index=False)
    print(f"Saved: {RESULTS_DIR / 'echonet_torus_results.csv'}")
    
    # ---- Analysis ----
    print("\n" + "=" * 65)
    print("ECHONET TORUS ANALYSIS")
    print("=" * 65)
    
    if ef_col and 'ef' in df.columns:
        print(f"\nVideos analyzed: {len(df)}")
        print(f"EF range: {df['ef'].min():.1f} - {df['ef'].max():.1f}%")
        
        # EF categories (clinical thresholds)
        df['ef_category'] = pd.cut(df['ef'],
            bins=[0, 30, 40, 55, 100],
            labels=['Severely Reduced (<30%)', 'Reduced (30-40%)',
                    'Mildly Reduced (40-55%)', 'Normal (>55%)'])
        
        print(f"\nEF categories:")
        for cat, group in df.groupby('ef_category', observed=True):
            print(f"  {cat}: n={len(group)}")
        
        # Correlations: κ vs EF
        print(f"\n{'='*65}")
        print("KEY TEST: Curvature vs Ejection Fraction")
        print(f"{'='*65}")
        
        correlation_results = {}
        
        for signal in ['lv', 'valve', 'motion']:
            kappa_col = f'{signal}_kappa_median'
            gini_col = f'{signal}_gini'
            spread_col = f'{signal}_spread'
            
            if kappa_col not in df.columns:
                continue
            
            valid = df[[kappa_col, 'ef']].dropna()
            if len(valid) < 20:
                continue
            
            rho, p = stats.spearmanr(valid[kappa_col], valid['ef'])
            print(f"\n  {signal.upper()} signal:")
            print(f"    κ_median vs EF: Spearman ρ = {rho:+.3f}, p = {p:.2e}")
            
            if gini_col in df.columns:
                valid_g = df[[gini_col, 'ef']].dropna()
                rho_g, p_g = stats.spearmanr(valid_g[gini_col], valid_g['ef'])
                print(f"    Gini vs EF:     Spearman ρ = {rho_g:+.3f}, p = {p_g:.2e}")
            
            if spread_col in df.columns:
                valid_s = df[[spread_col, 'ef']].dropna()
                rho_s, p_s = stats.spearmanr(valid_s[spread_col], valid_s['ef'])
                print(f"    Spread vs EF:   Spearman ρ = {rho_s:+.3f}, p = {p_s:.2e}")
            
            correlation_results[signal] = {
                'rho_kappa_ef': round(float(rho), 4),
                'p_kappa_ef': float(p),
                'n': len(valid),
            }
            
            # Group comparison: Low EF (<35%) vs Normal EF (>55%)
            low_ef = df[(df['ef'] < 35) & df[kappa_col].notna()][kappa_col]
            norm_ef = df[(df['ef'] > 55) & df[kappa_col].notna()][kappa_col]
            
            if len(low_ef) >= 10 and len(norm_ef) >= 10:
                U, p_u = stats.mannwhitneyu(low_ef, norm_ef, alternative='two-sided')
                r = 1 - 2*U/(len(low_ef)*len(norm_ef))
                print(f"    Low EF(<35%) vs Normal(>55%): r = {r:+.3f}, p = {p_u:.2e}")
                print(f"      κ_low={np.median(low_ef):.3f}, κ_norm={np.median(norm_ef):.3f}")
                
                correlation_results[signal].update({
                    'r_lowEF_normEF': round(float(r), 4),
                    'p_lowEF_normEF': float(p_u),
                    'kappa_lowEF': round(float(np.median(low_ef)), 4),
                    'kappa_normEF': round(float(np.median(norm_ef)), 4),
                    'n_lowEF': len(low_ef),
                    'n_normEF': len(norm_ef),
                })
        
        with open(RESULTS_DIR / 'echonet_correlations.json', 'w') as f:
            json.dump(correlation_results, f, indent=2)
        
        # ---- Figures ----
        print("\nGenerating figures...")
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        for col_idx, signal in enumerate(['lv', 'valve', 'motion']):
            kappa_col = f'{signal}_kappa_median'
            gini_col = f'{signal}_gini'
            
            if kappa_col not in df.columns:
                continue
            
            valid = df[['ef', kappa_col, gini_col]].dropna() if gini_col in df.columns else df[['ef', kappa_col]].dropna()
            
            if len(valid) < 20:
                continue
            
            # Row 1: κ vs EF scatter
            ax = axes[0, col_idx]
            ax.scatter(valid['ef'], valid[kappa_col], s=2, alpha=0.15,
                       color='#2196F3', rasterized=True)
            
            # Binned means
            bins = np.arange(10, 85, 5)
            bin_centers = []
            bin_means = []
            bin_stds = []
            for b in range(len(bins)-1):
                mask = (valid['ef'] >= bins[b]) & (valid['ef'] < bins[b+1])
                vals = valid.loc[mask, kappa_col]
                if len(vals) >= 5:
                    bin_centers.append((bins[b] + bins[b+1])/2)
                    bin_means.append(np.median(vals))
                    bin_stds.append(stats.iqr(vals)/2)
            
            if bin_centers:
                ax.errorbar(bin_centers, bin_means, yerr=bin_stds,
                           color='#F44336', linewidth=2, markersize=5,
                           marker='o', capsize=3, label='Binned median ± IQR/2')
            
            rho = stats.spearmanr(valid['ef'], valid[kappa_col])[0]
            ax.set_xlabel('Ejection Fraction (%)')
            ax.set_ylabel(f'Median κ ({signal})')
            ax.set_title(f'{signal.upper()} signal: ρ = {rho:+.3f}')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
            
            # Row 2: κ by EF category (boxplot)
            ax = axes[1, col_idx]
            categories = ['Severely Reduced (<30%)', 'Reduced (30-40%)',
                         'Mildly Reduced (40-55%)', 'Normal (>55%)']
            cat_data = []
            cat_labels = []
            cat_colors = ['#F44336', '#FF9800', '#FFC107', '#2196F3']
            
            for cat in categories:
                vals = df[df['ef_category'] == cat][kappa_col].dropna()
                if len(vals) >= 5:
                    cat_data.append(vals.values)
                    short = cat.split('(')[0].strip()
                    cat_labels.append(f"{short}\n(n={len(vals)})")
            
            if cat_data:
                bp = ax.boxplot(cat_data, widths=0.6, patch_artist=True,
                               showfliers=False,
                               medianprops=dict(color='black', linewidth=2))
                for i, (patch, color) in enumerate(zip(bp['boxes'], cat_colors[:len(cat_data)])):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax.set_xticks(range(1, len(cat_labels)+1))
                ax.set_xticklabels(cat_labels, fontsize=8)
                ax.set_ylabel(f'Median κ ({signal})')
                ax.set_title(f'{signal.upper()}: κ by EF category')
                ax.grid(axis='y', alpha=0.2)
        
        fig.suptitle(f'Figure E1: EchoNet-Dynamic — Torus Curvature vs Ejection Fraction\n'
                     f'(n = {len(df):,} echocardiograms)',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        path = FIGURES_DIR / f'figE1_echonet_curvature_ef.{FIG_FORMAT}'
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")
        
        # Figure 2: Disease landscape — EF category in κ-Gini space
        if 'lv_gini' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 7))
            cat_colors_map = {
                'Severely Reduced (<30%)': '#F44336',
                'Reduced (30-40%)': '#FF9800',
                'Mildly Reduced (40-55%)': '#FFC107',
                'Normal (>55%)': '#2196F3',
            }
            for cat in categories:
                g = df[df['ef_category'] == cat].dropna(subset=['lv_kappa_median', 'lv_gini'])
                if len(g) > 0:
                    ax.scatter(g['lv_kappa_median'], g['lv_gini'],
                              c=cat_colors_map.get(cat, '#607D8B'), s=8, alpha=0.3,
                              label=f"{cat.split('(')[0].strip()} (n={len(g)})",
                              rasterized=True)
            
            ax.set_xlabel('Median κ (LV brightness)')
            ax.set_ylabel('Curvature Gini')
            ax.set_title(f'Figure E2: EchoNet Disease Landscape on κ-Gini Plane\n'
                        f'(n = {len(df):,})', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9, markerscale=4)
            ax.grid(alpha=0.2)
            
            path = FIGURES_DIR / f'figE2_echonet_landscape.{FIG_FORMAT}'
            fig.savefig(path, dpi=DPI, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {path}")
    
    # ---- Summary ----
    print(f"\n{'='*65}")
    print("ECHONET SUMMARY")
    print(f"{'='*65}")
    print(f"Videos processed: {len(df)}")
    
    if 'ef' in df.columns and 'lv_kappa_median' in df.columns:
        rho, p = stats.spearmanr(
            df['lv_kappa_median'].dropna(),
            df.loc[df['lv_kappa_median'].notna(), 'ef']
        )
        print(f"LV brightness κ vs EF: ρ = {rho:+.3f} (p = {p:.2e})")
        
        if rho < -0.1 and p < 0.05:
            print(f"\n✓ VALIDATION SUCCESSFUL: Higher κ = lower EF = worse cardiac function")
            print(f"  The torus framework detects reduced ejection fraction from")
            print(f"  echocardiogram wall motion — consistent with CHF finding in RR torus.")
        elif rho > 0.1 and p < 0.05:
            print(f"\n~ UNEXPECTED DIRECTION: Higher κ = higher EF")
            print(f"  The relationship is opposite to the CHF prediction.")
            print(f"  This may reflect different dynamics in wall motion vs RR intervals.")
        else:
            print(f"\n? INCONCLUSIVE: No significant correlation (ρ = {rho:.3f})")
    
    print(f"\nAll results: {RESULTS_DIR / 'echonet_torus_results.csv'}")
    print(f"Correlations: {RESULTS_DIR / 'echonet_correlations.json'}")


if __name__ == '__main__':
    main()

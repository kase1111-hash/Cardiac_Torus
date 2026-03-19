"""
13_echonet_lvh.py — EchoNet-LVH (Parasternal Long Axis) Torus Analysis
True North Research | Cardiac Torus Pipeline

The PLAX view is the OPTIMAL view for valve motion analysis:
  - Mitral valve leaflets are perpendicular to the ultrasound beam
  - Aortic valve clearly visible at the top of the image
  - Interventricular septum (IVS) and posterior wall (LVPW) measurable
  - Virtual M-mode through the valve tips gives real leaflet dynamics

EchoNet-LVH provides:
  - 12,000 PLAX echocardiogram videos
  - IVS thickness, LVPW thickness measurements
  - LV internal diameter (diastole/systole)
  - Can derive: fractional shortening, relative wall thickness

This script extracts:
  1. Mitral valve dynamics via optimally-positioned virtual M-mode
  2. Aortic valve dynamics via upper scanline
  3. LV wall motion from septum and posterior wall regions
  4. Torus curvature for each signal
  5. Correlation with wall thickness and chamber dimensions

REQUIREMENTS:
  pip install opencv-python numpy pandas scipy matplotlib tqdm

USAGE:
  python 13_echonet_lvh.py --data_dir /path/to/EchoNet-LVH
"""

import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

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
    def tqdm(iterable, **kwargs):
        total = kwargs.get('total', None)
        desc = kwargs.get('desc', '')
        for i, item in enumerate(iterable):
            if total and (i % max(1, total//20) == 0):
                print(f"  {desc} {i}/{total}")
            yield item


# =====================================================================
# CONFIGURATION
# =====================================================================

ECHONET_LVH_DIR = Path(r"G:\EchoNet-LVH")

# PLAX anatomy in 112×112 frame:
# Top of image = near field (chest wall, RV)
# Upper-center = aortic root and valve
# Center = mitral valve, LV outflow tract  
# Lower portion = LV cavity, posterior wall
# The septum runs diagonally from upper-left to center
# The posterior wall is at the bottom

# ROI definitions for PLAX view (these are approximate)
PLAX_ROIS = {
    'mitral_valve': {  # Where MV leaflets are in PLAX
        'y_start': 40, 'y_end': 75,
        'x_start': 45, 'x_end': 80,
    },
    'aortic_root': {  # Aortic valve region
        'y_start': 15, 'y_end': 45,
        'x_start': 40, 'x_end': 75,
    },
    'lv_cavity': {  # LV cavity (filling/emptying)
        'y_start': 50, 'y_end': 95,
        'x_start': 30, 'x_end': 85,
    },
    'septum': {  # IVS region  
        'y_start': 25, 'y_end': 60,
        'x_start': 20, 'x_end': 50,
    },
    'posterior_wall': {  # LVPW region
        'y_start': 75, 'y_end': 105,
        'x_start': 30, 'x_end': 80,
    },
}

# Virtual M-mode scanlines for PLAX
# MV tips: vertical line through the center of the MV region
MMODE_MV_X = 62   # Through mitral valve leaflet tips
MMODE_AV_X = 56   # Through aortic valve
MMODE_LV_X = 56   # Through LV at papillary muscle level

MAX_VIDEOS = None
SKIP_ERRORS = True


# =====================================================================
# TORUS FUNCTIONS
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


def compute_signal_torus(signal, prefix):
    """Compute torus features from a 1D motion signal."""
    if len(signal) < 30:
        return None

    # Smooth
    kernel = 3
    sig = np.convolve(signal, np.ones(kernel)/kernel, mode='valid')

    s_pre = sig[:-1]
    s_post = sig[1:]
    n = len(s_pre)
    if n < 20:
        return None

    sig_min = np.percentile(sig, 2)
    sig_max = np.percentile(sig, 98)
    if sig_max - sig_min < 1e-6:
        return None

    theta1 = np.array([to_angle(s, sig_min, sig_max) for s in s_pre])
    theta2 = np.array([to_angle(s, sig_min, sig_max) for s in s_post])

    kappa = np.zeros(n)
    for i in range(1, n-1):
        kappa[i] = menger_curvature_torus(
            (theta1[i-1], theta2[i-1]),
            (theta1[i], theta2[i]),
            (theta1[i+1], theta2[i+1]))

    valid = kappa[kappa > 0]
    if len(valid) < 10:
        return None

    gini = gini_coefficient(valid)
    spread = np.sqrt(np.std(theta1)**2 + np.std(theta2)**2)

    dt1 = np.diff(theta1)
    dt1 = np.where(dt1 > np.pi, dt1 - 2*np.pi, dt1)
    dt1 = np.where(dt1 < -np.pi, dt1 + 2*np.pi, dt1)
    dt2 = np.diff(theta2)
    dt2 = np.where(dt2 > np.pi, dt2 - 2*np.pi, dt2)
    dt2 = np.where(dt2 < -np.pi, dt2 + 2*np.pi, dt2)
    speed = np.sqrt(dt1**2 + dt2**2)

    return {
        f'{prefix}_kappa_median': round(float(np.median(valid)), 4),
        f'{prefix}_kappa_mean': round(float(np.mean(valid)), 4),
        f'{prefix}_kappa_p95': round(float(np.percentile(valid, 95)), 4),
        f'{prefix}_kappa_cv': round(float(np.std(valid)/np.mean(valid)), 4),
        f'{prefix}_gini': round(gini, 4),
        f'{prefix}_spread': round(float(spread), 4),
        f'{prefix}_speed_mean': round(float(np.mean(speed)), 4),
        f'{prefix}_speed_cv': round(float(np.std(speed)/np.mean(speed)), 4) if np.mean(speed) > 0 else 0,
        f'{prefix}_signal_range': round(float(sig_max - sig_min), 4),
    }


# =====================================================================
# VIDEO PROCESSING
# =====================================================================

def process_plax_video(video_path):
    """Extract multiple motion signals from a PLAX echocardiogram."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if n_frames < 20 or fps <= 0:
        cap.release()
        return None

    # Collect signals
    roi_signals = {name: [] for name in PLAX_ROIS}
    mmode_mv = []     # Virtual M-mode through mitral valve
    mmode_av = []     # Virtual M-mode through aortic valve
    frame_diffs = []
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        h, w = gray.shape

        # ROI brightness signals
        for name, roi in PLAX_ROIS.items():
            ys = min(roi['y_start'], h-1)
            ye = min(roi['y_end'], h)
            xs = min(roi['x_start'], w-1)
            xe = min(roi['x_end'], w)
            roi_signals[name].append(float(np.mean(gray[ys:ye, xs:xe])))

        # Virtual M-mode: track bright-structure position along scanline
        # Mitral valve scanline
        mv_x = min(MMODE_MV_X, w-1)
        scanline_mv = gray[:, mv_x].astype(float)
        thresh = np.percentile(scanline_mv, 80)
        bright = np.where(scanline_mv >= thresh)[0]
        if len(bright) > 0:
            mmode_mv.append(float(np.average(bright, weights=scanline_mv[bright])))
        else:
            mmode_mv.append(float(h/2))

        # Aortic valve scanline
        av_x = min(MMODE_AV_X, w-1)
        scanline_av = gray[10:50, av_x].astype(float)  # Upper portion only
        thresh_av = np.percentile(scanline_av, 80)
        bright_av = np.where(scanline_av >= thresh_av)[0]
        if len(bright_av) > 0:
            mmode_av.append(float(np.average(bright_av, weights=scanline_av[bright_av])))
        else:
            mmode_av.append(20.0)

        # Frame difference
        if prev_frame is not None:
            frame_diffs.append(float(np.mean(np.abs(gray.astype(float) - prev_frame.astype(float)))))
        prev_frame = gray.copy()

    cap.release()

    if len(roi_signals['lv_cavity']) < 20:
        return None

    return {
        'roi_signals': {k: np.array(v) for k, v in roi_signals.items()},
        'mmode_mv': np.array(mmode_mv),
        'mmode_av': np.array(mmode_av),
        'frame_diffs': np.array(frame_diffs) if frame_diffs else np.zeros(1),
        'n_frames': len(roi_signals['lv_cavity']),
        'fps': fps,
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=str(ECHONET_LVH_DIR))
    parser.add_argument('--max_videos', type=int, default=MAX_VIDEOS)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 65)
    print("Step 13: EchoNet-LVH (PLAX) Torus Analysis")
    print("Cardiac Torus Pipeline — Paper II")
    print("=" * 65)

    # Find FileList
    filelist_path = data_dir / "FileList.csv"
    if not filelist_path.exists():
        # Try subdirectories
        for sub in ['', 'EchoNet-LVH']:
            test = data_dir / sub / "FileList.csv"
            if test.exists():
                filelist_path = test
                data_dir = test.parent
                break
        else:
            print(f"ERROR: FileList.csv not found in {data_dir}")
            sys.exit(1)

    df_meta = pd.read_csv(filelist_path)
    print(f"\nLoaded metadata for {len(df_meta)} videos")
    print(f"Columns: {list(df_meta.columns)}")

    # Find key columns
    fn_col = None
    for c in ['FileName', 'filename', 'Video']:
        if c in df_meta.columns:
            fn_col = c; break

    if fn_col is None:
        print(f"ERROR: No filename column found")
        sys.exit(1)

    # Find measurement columns (wall thickness, dimensions)
    measure_cols = {}
    for c in df_meta.columns:
        cl = c.lower().strip()
        if 'ivs' in cl or 'septum' in cl:
            measure_cols['IVS'] = c
        elif 'lvpw' in cl or 'posterior' in cl or 'pw' in cl:
            measure_cols['LVPW'] = c
        elif 'lvid' in cl and ('d' in cl or 'diast' in cl):
            measure_cols['LVIDd'] = c
        elif 'lvid' in cl and ('s' in cl or 'syst' in cl):
            measure_cols['LVIDs'] = c
        elif 'ef' in cl or 'ejection' in cl:
            measure_cols['EF'] = c
        elif 'fs' == cl or 'fractional' in cl:
            measure_cols['FS'] = c

    print(f"Measurement columns found: {measure_cols}")

    # Video directory
    video_dir = data_dir / "Videos"
    if not video_dir.exists():
        for alt in ["videos", "a4c-video-dir", "plax-video-dir"]:
            if (data_dir / alt).exists():
                video_dir = data_dir / alt; break

    print(f"Video directory: {video_dir}")

    if args.max_videos:
        df_process = df_meta.head(args.max_videos)
    else:
        df_process = df_meta

    print(f"\nProcessing {len(df_process)} videos...")

    # Process videos
    all_results = []
    errors = 0

    for idx, row in tqdm(df_process.iterrows(), total=len(df_process), desc="PLAX"):
        filename = row[fn_col]
        if not filename.endswith('.avi'):
            filename = filename + '.avi'

        video_path = video_dir / filename
        if not video_path.exists():
            errors += 1
            continue

        try:
            signals = process_plax_video(str(video_path))
            if signals is None:
                errors += 1
                continue

            result = {
                'filename': row[fn_col],
                'n_frames': signals['n_frames'],
                'fps': signals['fps'],
            }

            # Add metadata measurements
            for key, col in measure_cols.items():
                if col in row.index and pd.notna(row[col]):
                    result[key.lower()] = float(row[col])

            # Compute torus features for each ROI signal
            for roi_name, signal in signals['roi_signals'].items():
                feats = compute_signal_torus(signal, roi_name)
                if feats:
                    result.update(feats)

            # Virtual M-mode signals
            mv_feats = compute_signal_torus(signals['mmode_mv'], 'mmode_mv')
            if mv_feats:
                result.update(mv_feats)

            av_feats = compute_signal_torus(signals['mmode_av'], 'mmode_av')
            if av_feats:
                result.update(av_feats)

            # Motion energy
            if len(signals['frame_diffs']) > 20:
                motion_feats = compute_signal_torus(signals['frame_diffs'], 'motion')
                if motion_feats:
                    result.update(motion_feats)

            all_results.append(result)

        except Exception as e:
            if not SKIP_ERRORS:
                raise
            errors += 1

    print(f"\nProcessed: {len(all_results)}, Errors: {errors}")

    if len(all_results) == 0:
        print("ERROR: No videos processed!")
        sys.exit(1)

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / 'echonet_lvh_torus_results.csv', index=False)
    print(f"Saved: {RESULTS_DIR / 'echonet_lvh_torus_results.csv'}")

    # ---- Analysis ----
    print("\n" + "=" * 65)
    print("ECHONET-LVH TORUS ANALYSIS")
    print("=" * 65)
    print(f"Videos analyzed: {len(df)}")

    # Correlation analysis with available measurements
    signal_prefixes = ['lv_cavity', 'mitral_valve', 'aortic_root', 'septum',
                        'posterior_wall', 'mmode_mv', 'mmode_av', 'motion']

    targets = []
    for key in ['ef', 'ivs', 'lvpw', 'lvidd', 'lvids', 'fs']:
        if key in df.columns and df[key].notna().sum() > 20:
            targets.append(key)

    if targets:
        print(f"\nMeasurement columns available: {targets}")

        print(f"\n{'Signal':20s}", end='')
        for t in targets:
            print(f" {'ρ(κ,'+t+')':>14s}", end='')
        print(f" {'ρ(G,'+targets[0]+')':>14s}" if targets else '')
        print("-" * (20 + 15 * len(targets) + 15))

        correlation_results = {}

        for prefix in signal_prefixes:
            kappa_col = f'{prefix}_kappa_median'
            gini_col = f'{prefix}_gini'

            if kappa_col not in df.columns:
                continue

            print(f"  {prefix:18s}", end='')
            corr_row = {}

            for target in targets:
                valid = df[[kappa_col, target]].dropna()
                if len(valid) >= 20:
                    rho, p = stats.spearmanr(valid[kappa_col], valid[target])
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    print(f" {rho:+.3f}{sig:3s}      ", end='')
                    corr_row[f'rho_kappa_{target}'] = round(float(rho), 4)
                    corr_row[f'p_kappa_{target}'] = float(p)
                else:
                    print(f" {'n/a':>14s}", end='')

            # Gini vs first target
            if targets and gini_col in df.columns:
                valid_g = df[[gini_col, targets[0]]].dropna()
                if len(valid_g) >= 20:
                    rho_g, p_g = stats.spearmanr(valid_g[gini_col], valid_g[targets[0]])
                    sig = "***" if p_g < 0.001 else "**" if p_g < 0.01 else "*" if p_g < 0.05 else ""
                    print(f" {rho_g:+.3f}{sig:3s}", end='')
                    corr_row[f'rho_gini_{targets[0]}'] = round(float(rho_g), 4)

            print()
            correlation_results[prefix] = corr_row

        with open(RESULTS_DIR / 'echonet_lvh_correlations.json', 'w') as f:
            json.dump(correlation_results, f, indent=2)

    # ---- Figures ----
    print("\nGenerating figures...")

    # Find the best signal and best target for the main figure
    best_signal = None
    best_rho = 0
    best_target = targets[0] if targets else None

    for prefix in signal_prefixes:
        kappa_col = f'{prefix}_kappa_median'
        if kappa_col in df.columns and best_target and best_target in df.columns:
            valid = df[[kappa_col, best_target]].dropna()
            if len(valid) >= 20:
                rho, _ = stats.spearmanr(valid[kappa_col], valid[best_target])
                if abs(rho) > abs(best_rho):
                    best_rho = rho
                    best_signal = prefix

    if best_signal and best_target:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        plot_signals = [s for s in signal_prefixes
                        if f'{s}_kappa_median' in df.columns][:6]

        for idx, prefix in enumerate(plot_signals):
            row, col = divmod(idx, 3)
            if row >= 2 or col >= 3:
                break
            ax = axes[row, col]

            kappa_col = f'{prefix}_kappa_median'
            valid = df[[kappa_col, best_target]].dropna()

            if len(valid) >= 20:
                ax.scatter(valid[best_target], valid[kappa_col],
                           s=2, alpha=0.15, color='#2196F3', rasterized=True)

                rho, p = stats.spearmanr(valid[kappa_col], valid[best_target])
                ax.set_title(f'{prefix}: ρ={rho:+.3f}', fontsize=10)
            else:
                ax.set_title(f'{prefix}: n/a', fontsize=10)

            ax.set_xlabel(best_target)
            ax.set_ylabel(f'κ median')
            ax.grid(alpha=0.2)

        fig.suptitle(f'EchoNet-LVH: Torus Curvature vs {best_target}\n(n={len(df):,})',
                     fontsize=13, fontweight='bold')
        fig.tight_layout()

        path = FIGURES_DIR / f'figL1_echonet_lvh.{FIG_FORMAT}'
        fig.savefig(path, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")

    print(f"\nAll results: {RESULTS_DIR / 'echonet_lvh_torus_results.csv'}")


if __name__ == '__main__':
    main()

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
    'mitral_valve': {  # MV leaflets in PLAX (center-lower)
        'y_start': 300, 'y_end': 550,
        'x_start': 350, 'x_end': 600,
    },
    'aortic_root': {  # Aortic valve (upper-center)
        'y_start': 100, 'y_end': 320,
        'x_start': 350, 'x_end': 600,
    },
    'lv_cavity': {  # LV cavity bulk
        'y_start': 280, 'y_end': 650,
        'x_start': 250, 'x_end': 650,
    },
    'septum': {  # IVS region (upper-left to center)
        'y_start': 200, 'y_end': 420,
        'x_start': 180, 'x_end': 400,
    },
    'posterior_wall': {  # LVPW region (lower)
        'y_start': 500, 'y_end': 700,
        'x_start': 250, 'x_end': 600,
    },
}

# Virtual M-mode scanlines for 1024-wide PLAX
MMODE_MV_X = 480   # Through mitral valve leaflet tips
MMODE_AV_X = 450   # Through aortic valve
MMODE_LV_X = 450   # Through LV

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
        scanline_av = gray[80:320, av_x].astype(float)  # Upper portion for AV
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
    parser.add_argument('--workers', type=int, default=1, help='Parallel workers (1-8)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 65)
    print("Step 13: EchoNet-LVH (PLAX) Torus Analysis")
    print("Cardiac Torus Pipeline — Paper II")
    print("=" * 65)

    # Find metadata CSV
    filelist_path = None
    for name in ['FileList.csv', 'MeasurementsList.csv', 'filelist.csv', 'measurements.csv']:
        test = data_dir / name
        if test.exists():
            filelist_path = test; break
        for sub in ['', 'EchoNet-LVH']:
            test2 = data_dir / sub / name
            if test2.exists():
                filelist_path = test2; data_dir = test2.parent; break
        if filelist_path: break

    if filelist_path is None:
        # List what's actually there
        print(f"ERROR: No metadata CSV found in {data_dir}")
        print(f"Files found: {[f.name for f in data_dir.iterdir()]}")
        sys.exit(1)

    df_meta = pd.read_csv(filelist_path)
    print(f"\nLoaded metadata from {filelist_path.name}: {len(df_meta)} rows")
    print(f"Columns: {list(df_meta.columns)}")

    # ---- Detect format: long (MeasurementsList) or wide (FileList) ----
    is_long = 'Calc' in df_meta.columns and 'CalcValue' in df_meta.columns

    if is_long:
        print(f"  Long format detected — pivoting to wide...")
        print(f"  Measurement types: {df_meta['Calc'].unique().tolist()}")

        fn_col = 'HashedFileName'

        # Pivot CalcValue to columns
        df_wide = df_meta.pivot_table(
            index=fn_col, columns='Calc', values='CalcValue', aggfunc='first'
        ).reset_index()

        # Get per-video metadata
        vid_meta = df_meta.groupby(fn_col).first()[['Frames', 'FPS', 'Width', 'Height', 'split']].reset_index()
        df_wide = df_wide.merge(vid_meta, on=fn_col, how='left')

        # Grab LVIDd measurement coords for ROI guidance
        lvid_rows = df_meta[df_meta['Calc'] == 'LVIDd'][[fn_col, 'Frame', 'X1', 'X2', 'Y1', 'Y2']].copy()
        lvid_rows.columns = [fn_col, 'meas_frame', 'lvid_x1', 'lvid_x2', 'lvid_y1', 'lvid_y2']
        df_wide = df_wide.merge(lvid_rows, on=fn_col, how='left')

        # Derive FS and RWT
        if 'LVIDd' in df_wide.columns and 'LVIDs' in df_wide.columns:
            df_wide['FS'] = (df_wide['LVIDd'] - df_wide['LVIDs']) / df_wide['LVIDd']
            print(f"  Derived: Fractional Shortening (FS)")
        if 'LVPWd' in df_wide.columns and 'LVIDd' in df_wide.columns:
            df_wide['RWT'] = 2 * df_wide['LVPWd'] / df_wide['LVIDd']
            print(f"  Derived: Relative Wall Thickness (RWT)")

        df_meta = df_wide
        print(f"  Wide: {len(df_meta)} videos x {len(df_meta.columns)} columns")
    else:
        # Standard wide format — find filename column
        fn_col = None
        for c in df_meta.columns:
            cl = c.lower().strip()
            if cl in ['filename', 'video', 'hashedfilename']:
                fn_col = c; break
            if 'file' in cl and 'name' in cl:
                fn_col = c; break
        if fn_col is None:
            fn_col = df_meta.columns[0]
        print(f"  Filename column: '{fn_col}'")

    # Map measurement columns
    measure_cols = {}
    for c in df_meta.columns:
        cl = c.strip()
        if cl == 'IVSd': measure_cols['ivsd'] = c
        elif cl == 'LVPWd': measure_cols['lvpwd'] = c
        elif cl == 'LVIDd': measure_cols['lvidd'] = c
        elif cl == 'LVIDs': measure_cols['lvids'] = c
        elif cl == 'IVSs': measure_cols['ivss'] = c
        elif cl == 'LVPWs': measure_cols['lvpws'] = c
        elif cl == 'FS': measure_cols['fs'] = c
        elif cl == 'RWT': measure_cols['rwt'] = c
    print(f"  Measurements: {list(measure_cols.keys())}")

    # Find videos across Batch directories
    video_dirs = []
    for candidate in ['Videos', 'Batch1', 'Batch2', 'Batch3', 'Batch4',
                       'Batch5', 'Batch6', 'Batch7', 'Batch8']:
        d = data_dir / candidate
        if d.exists() and d.is_dir():
            video_dirs.append(d)
    if not video_dirs:
        video_dirs = [data_dir]
    print(f"  Video dirs: {[d.name for d in video_dirs]}")

    # Build filename→path lookup
    video_lookup = {}
    for vdir in video_dirs:
        for f in vdir.iterdir():
            if f.suffix.lower() == '.avi':
                video_lookup[f.stem] = f
                video_lookup[f.name] = f
    print(f"  Videos on disk: {len(video_lookup)//2}")

    if args.max_videos:
        df_process = df_meta.head(args.max_videos)
    else:
        df_process = df_meta

    print(f"\nProcessing {len(df_process)} videos with {args.workers} worker(s)...")

    # Build job list: (video_path, row_dict)
    jobs = []
    skipped = 0
    for idx, row in df_process.iterrows():
        filename = str(row[fn_col]).strip()
        video_path = None
        for variant in [filename, filename + '.avi', filename.replace('.avi', '') + '.avi']:
            if variant in video_lookup:
                video_path = video_lookup[variant]; break
        if video_path is None:
            stem = filename.replace('.avi', '')
            if stem in video_lookup:
                video_path = video_lookup[stem]
        if video_path is None or not video_path.exists():
            skipped += 1; continue
        jobs.append((str(video_path), row.to_dict(), dict(measure_cols)))

    print(f"  Videos found: {len(jobs)}, skipped: {skipped}")

    def process_one(args_tuple):
        vpath, row_dict, mcols = args_tuple
        try:
            signals = process_plax_video(vpath)
            if signals is None:
                return None
            result = {
                'filename': row_dict[fn_col],
                'n_frames': signals['n_frames'],
                'fps': signals['fps'],
            }
            for key, col in mcols.items():
                if col in row_dict and row_dict[col] is not None and pd.notna(row_dict[col]):
                    result[key] = float(row_dict[col])
            for roi_name, signal in signals['roi_signals'].items():
                feats = compute_signal_torus(signal, roi_name)
                if feats: result.update(feats)
            mv_feats = compute_signal_torus(signals['mmode_mv'], 'mmode_mv')
            if mv_feats: result.update(mv_feats)
            av_feats = compute_signal_torus(signals['mmode_av'], 'mmode_av')
            if av_feats: result.update(av_feats)
            if len(signals['frame_diffs']) > 20:
                motion_feats = compute_signal_torus(signals['frame_diffs'], 'motion')
                if motion_feats: result.update(motion_feats)
            return result
        except Exception as e:
            return None

    all_results = []
    errors = 0

    if args.workers <= 1:
        for i, job in enumerate(tqdm(jobs, desc="PLAX")):
            r = process_one(job)
            if r: all_results.append(r)
            else: errors += 1
            # Checkpoint save every 2000 videos
            if (i+1) % 2000 == 0:
                pd.DataFrame(all_results).to_csv(RESULTS_DIR / 'echonet_lvh_torus_results_checkpoint.csv', index=False)
                print(f"\n  Checkpoint saved: {len(all_results)} results at video {i+1}")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one, j): j for j in jobs}
            for i, future in enumerate(as_completed(futures)):
                r = future.result()
                if r: all_results.append(r)
                else: errors += 1
                if (i+1) % 500 == 0 or i+1 == len(jobs):
                    print(f"  Progress: {i+1}/{len(jobs)} ({len(all_results)} ok, {errors} err)")
                if (i+1) % 2000 == 0:
                    pd.DataFrame(all_results).to_csv(RESULTS_DIR / 'echonet_lvh_torus_results_checkpoint.csv', index=False)

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
    for key in ['lvidd', 'lvids', 'ivsd', 'lvpwd', 'fs', 'rwt', 'ivss', 'lvpws', 'ef']:
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

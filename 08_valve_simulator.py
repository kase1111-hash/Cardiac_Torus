"""
08_valve_simulator.py — Synthetic Valve Motion Simulator & Torus Analysis
True North Research | Cardiac Torus Pipeline

Generates realistic M-mode mitral valve leaflet motion traces for:
  1. Normal mitral valve
  2. Mitral stenosis (restricted opening, reduced EF slope)
  3. Mitral valve prolapse (late-systolic posterior displacement)
  4. Fluttering / AF (high-frequency oscillation on leaflet motion)
  5. Flail leaflet (chaotic closure, exaggerated excursion)

For each condition, maps the valve phase portrait (position, velocity)
onto a torus T² and computes geodesic curvature — the same framework
used for RR intervals but now applied to mechanical valve motion.

The question: does the torus curvature of valve motion distinguish
these conditions the way RR torus curvature distinguishes arrhythmias?

VALVE MOTION PHYSICS (M-mode landmarks):
  D point: onset of diastolic opening
  E point: peak early opening (E-wave, passive filling)
  F point: partial closure after E-wave
  A point: reopening from atrial contraction (A-wave)
  C point: full closure at systole onset
  
  Normal E-F slope: 70-150 mm/s
  Stenosis E-F slope: 10-40 mm/s (diagnostic criterion)
  Normal D-E excursion: 18-27 mm
  Stenosis D-E excursion: <15 mm

TORUS MAPPING:
  θ₁ = 2π × normalize(leaflet_position)     [displacement]
  θ₂ = 2π × normalize(leaflet_velocity)     [rate of change]
  
  This is a phase-space torus — (x, dx/dt) wrapped onto T².
  A healthy valve traces a smooth, repeatable orbit.
  Pathology distorts the orbit in condition-specific ways.
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal, stats

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, FIGURES_DIR, DPI, FIG_FORMAT


# =====================================================================
# VALVE MOTION GENERATORS
# =====================================================================

def generate_single_cycle(t, params):
    """
    Generate one cardiac cycle of mitral valve anterior leaflet motion.
    
    The leaflet position is modeled as a piecewise function:
      Systole (C-D): closed, near baseline
      Early diastole (D-E): rapid opening to E-point
      Mid-diastole (E-F): partial closure (EF slope)
      Late diastole (F-A): reopening from atrial kick
      End diastole (A-C): rapid closure back to baseline
    
    Uses smooth interpolation (raised cosine transitions) to avoid
    discontinuities that would create spurious curvature.
    """
    cycle_len = len(t)
    pos = np.zeros(cycle_len)
    
    # Timing fractions within one cycle
    # Systole ~40% of cycle, diastole ~60%
    systole_end = params.get('systole_frac', 0.38)
    e_peak_time = systole_end + 0.12       # E-point (rapid opening)
    f_point_time = systole_end + 0.28      # F-point (partial closure)
    a_peak_time = systole_end + 0.48       # A-point (atrial kick)
    
    # Amplitudes (mm)
    baseline = params.get('baseline', 0.0)       # closed position
    e_amplitude = params.get('e_amplitude', 24.0)  # E-point excursion
    f_amplitude = params.get('f_amplitude', 14.0)  # F-point level
    a_amplitude = params.get('a_amplitude', 18.0)  # A-point excursion
    
    for i, frac in enumerate(np.linspace(0, 1, cycle_len, endpoint=False)):
        if frac < systole_end:
            # Systole: closed at baseline with slight bulge
            systole_phase = frac / systole_end
            pos[i] = baseline + 1.5 * np.sin(np.pi * systole_phase)
            
        elif frac < e_peak_time:
            # D→E: rapid opening
            phase = (frac - systole_end) / (e_peak_time - systole_end)
            # Raised cosine for smooth transition
            pos[i] = baseline + e_amplitude * 0.5 * (1 - np.cos(np.pi * phase))
            
        elif frac < f_point_time:
            # E→F: partial closure (EF slope)
            phase = (frac - e_peak_time) / (f_point_time - e_peak_time)
            pos[i] = e_amplitude + (f_amplitude - e_amplitude) * 0.5 * (1 - np.cos(np.pi * phase))
            
        elif frac < a_peak_time:
            # F→A: reopening from atrial contraction
            phase = (frac - f_point_time) / (a_peak_time - f_point_time)
            pos[i] = f_amplitude + (a_amplitude - f_amplitude) * 0.5 * (1 - np.cos(np.pi * phase))
            
        else:
            # A→C: rapid closure
            phase = (frac - a_peak_time) / (1.0 - a_peak_time)
            pos[i] = a_amplitude * 0.5 * (1 + np.cos(np.pi * phase))
    
    return pos


def generate_valve_trace(condition: str, n_cycles: int = 50,
                          samples_per_cycle: int = 300,
                          heart_rate_bpm: float = 72.0,
                          rng: np.random.Generator = None) -> dict:
    """
    Generate a multi-cycle valve motion trace for a given condition.
    
    Returns dict with:
      position: leaflet displacement array (mm)
      time: time array (seconds)
      condition: condition label
      params: generation parameters used
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    total_samples = n_cycles * samples_per_cycle
    cycle_duration = 60.0 / heart_rate_bpm  # seconds
    total_time = n_cycles * cycle_duration
    
    dt = total_time / total_samples
    fs = 1.0 / dt
    
    # Base parameters per condition
    if condition == 'Normal':
        base_params = {
            'e_amplitude': 24.0,
            'f_amplitude': 14.0,
            'a_amplitude': 18.0,
            'systole_frac': 0.38,
            'baseline': 0.0,
        }
        cycle_var = 0.03       # low cycle-to-cycle variability
        noise_std = 0.3        # low noise
        flutter_amp = 0.0
        prolapse_amp = 0.0
        hr_variability = 0.02  # normal HRV
        
    elif condition == 'Stenosis':
        base_params = {
            'e_amplitude': 12.0,   # restricted opening
            'f_amplitude': 9.0,    # reduced EF slope (key diagnostic)
            'a_amplitude': 10.0,   # reduced A-wave
            'systole_frac': 0.35,
            'baseline': 0.0,
        }
        cycle_var = 0.02       # very consistent (rigid valve)
        noise_std = 0.2
        flutter_amp = 0.0
        prolapse_amp = 0.0
        hr_variability = 0.015
        
    elif condition == 'Prolapse':
        base_params = {
            'e_amplitude': 26.0,   # slightly exaggerated opening
            'f_amplitude': 15.0,
            'a_amplitude': 20.0,
            'systole_frac': 0.38,
            'baseline': 0.0,
        }
        cycle_var = 0.04
        noise_std = 0.4
        flutter_amp = 0.0
        prolapse_amp = 5.0     # late-systolic posterior displacement
        hr_variability = 0.025
        
    elif condition == 'Flutter':
        base_params = {
            'e_amplitude': 22.0,
            'f_amplitude': 13.0,
            'a_amplitude': 16.0,
            'systole_frac': 0.38,
            'baseline': 0.0,
        }
        cycle_var = 0.08       # high variability (AF-driven)
        noise_std = 0.5
        flutter_amp = 2.5      # high-frequency leaflet oscillation
        prolapse_amp = 0.0
        hr_variability = 0.15  # very irregular (AF)
        
    elif condition == 'Flail':
        base_params = {
            'e_amplitude': 30.0,   # exaggerated excursion
            'f_amplitude': 16.0,
            'a_amplitude': 22.0,
            'systole_frac': 0.36,
            'baseline': 0.0,
        }
        cycle_var = 0.10       # chaotic
        noise_std = 0.8
        flutter_amp = 1.5
        prolapse_amp = 8.0     # flail into LA during systole
        hr_variability = 0.04
        
    else:
        raise ValueError(f"Unknown condition: {condition}")
    
    # Generate cycles with variability
    all_positions = []
    cycle_times = []
    cycle_labels = []
    
    for c in range(n_cycles):
        # Vary heart rate
        hr_factor = 1.0 + rng.normal(0, hr_variability)
        this_cycle_samples = max(100, int(samples_per_cycle * hr_factor))
        
        # Vary parameters slightly per cycle
        params = {}
        for key, val in base_params.items():
            if isinstance(val, float):
                params[key] = val * (1.0 + rng.normal(0, cycle_var))
            else:
                params[key] = val
        
        # Clamp amplitudes to positive
        for key in ['e_amplitude', 'f_amplitude', 'a_amplitude']:
            params[key] = max(2.0, params[key])
        
        t_cycle = np.linspace(0, 1, this_cycle_samples, endpoint=False)
        pos_cycle = generate_single_cycle(t_cycle, params)
        
        # Add prolapse (late systolic posterior displacement)
        if prolapse_amp > 0:
            systole_frac = params['systole_frac']
            for i, frac in enumerate(t_cycle):
                if 0.15 < frac < systole_frac:
                    prolapse_phase = (frac - 0.15) / (systole_frac - 0.15)
                    displacement = prolapse_amp * np.sin(np.pi * prolapse_phase)
                    displacement *= (1.0 + rng.normal(0, 0.1))
                    pos_cycle[i] -= displacement  # posterior = negative
        
        # Add flutter (high-frequency oscillation)
        if flutter_amp > 0:
            flutter_freq = rng.uniform(15, 30)  # Hz equivalent
            flutter = flutter_amp * np.sin(2 * np.pi * flutter_freq * t_cycle)
            flutter *= (1.0 + rng.normal(0, 0.2))
            pos_cycle += flutter
        
        # Add noise
        pos_cycle += rng.normal(0, noise_std, len(pos_cycle))
        
        all_positions.append(pos_cycle)
        cycle_labels.extend([c] * len(pos_cycle))
    
    # Concatenate all cycles
    position = np.concatenate(all_positions)
    time = np.arange(len(position)) * dt
    
    return {
        'position': position,
        'time': time,
        'condition': condition,
        'n_cycles': n_cycles,
        'samples_per_cycle': samples_per_cycle,
        'fs': fs,
        'cycle_labels': np.array(cycle_labels),
        'base_params': base_params,
    }


# =====================================================================
# PHASE-SPACE TORUS MAPPING
# =====================================================================

def compute_velocity(position, dt):
    """Central difference velocity estimate."""
    vel = np.zeros_like(position)
    vel[1:-1] = (position[2:] - position[:-2]) / (2 * dt)
    vel[0] = (position[1] - position[0]) / dt
    vel[-1] = (position[-1] - position[-2]) / dt
    return vel


def map_to_phase_torus(position, velocity, pos_range=(-10, 35), vel_range=(-400, 400)):
    """
    Map (position, velocity) to angular coordinates on T².
    
    θ₁ = 2π × (pos - pos_min) / (pos_max - pos_min)
    θ₂ = 2π × (vel - vel_min) / (vel_max - vel_min)
    """
    pos_clip = np.clip(position, pos_range[0], pos_range[1])
    vel_clip = np.clip(velocity, vel_range[0], vel_range[1])
    
    theta1 = 2 * np.pi * (pos_clip - pos_range[0]) / (pos_range[1] - pos_range[0])
    theta2 = 2 * np.pi * (vel_clip - vel_range[0]) / (vel_range[1] - vel_range[0])
    
    return theta1, theta2


def menger_curvature_torus(p1, p2, p3):
    """Menger curvature for three points on flat torus."""
    def torus_dist(a, b):
        d1 = abs(a[0] - b[0])
        d1 = min(d1, 2*np.pi - d1)
        d2 = abs(a[1] - b[1])
        d2 = min(d2, 2*np.pi - d2)
        return np.sqrt(d1**2 + d2**2)
    
    a = torus_dist(p2, p3)
    b = torus_dist(p1, p3)
    c = torus_dist(p1, p2)
    
    if a < 1e-10 or b < 1e-10 or c < 1e-10:
        return 0.0
    s = (a + b + c) / 2
    area_sq = s * (s-a) * (s-b) * (s-c)
    if area_sq <= 0:
        return 0.0
    return 4 * np.sqrt(area_sq) / (a * b * c)


def gini_coefficient(values):
    v = np.abs(values[values > 0])
    if len(v) < 2:
        return 0.0
    v = np.sort(v)
    n = len(v)
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * v) / (n * np.sum(v))) - (n + 1) / n


def analyze_valve_trace(trace: dict, subsample: int = 3) -> dict:
    """
    Full torus analysis of a valve motion trace.
    
    subsample: take every Nth sample to match ~100-120 Hz effective rate
    (raw traces at 300 samples/cycle × 72bpm ≈ 360 Hz; subsample by 3 → 120 Hz)
    """
    pos = trace['position'][::subsample]
    dt = (trace['time'][1] - trace['time'][0]) * subsample
    
    # Compute velocity
    vel = compute_velocity(pos, dt)
    
    # Map to torus
    theta1, theta2 = map_to_phase_torus(pos, vel)
    
    n = len(theta1)
    
    # Compute curvature at every point
    kappa = np.zeros(n)
    for i in range(1, n - 1):
        kappa[i] = menger_curvature_torus(
            (theta1[i-1], theta2[i-1]),
            (theta1[i], theta2[i]),
            (theta1[i+1], theta2[i+1])
        )
    
    valid_kappa = kappa[kappa > 0]
    
    if len(valid_kappa) < 10:
        return None
    
    # Per-cycle analysis
    cycle_labels = trace['cycle_labels'][::subsample][:n]
    unique_cycles = np.unique(cycle_labels)
    
    cycle_ginis = []
    cycle_median_kappas = []
    cycle_orbit_areas = []
    
    for cyc in unique_cycles:
        mask = cycle_labels == cyc
        cyc_kappa = kappa[mask]
        cyc_valid = cyc_kappa[cyc_kappa > 0]
        if len(cyc_valid) >= 5:
            cycle_ginis.append(gini_coefficient(cyc_valid))
            cycle_median_kappas.append(np.median(cyc_valid))
            
            # Orbit area: approximate by std of θ₁ × std of θ₂
            cyc_t1 = theta1[mask]
            cyc_t2 = theta2[mask]
            cycle_orbit_areas.append(np.std(cyc_t1) * np.std(cyc_t2))
    
    # Global stats
    gini = gini_coefficient(valid_kappa)
    
    result = {
        'condition': trace['condition'],
        'n_samples': n,
        'n_cycles': trace['n_cycles'],
        
        # Curvature stats
        'kappa_median': round(float(np.median(valid_kappa)), 4),
        'kappa_mean': round(float(np.mean(valid_kappa)), 4),
        'kappa_std': round(float(np.std(valid_kappa)), 4),
        'kappa_p95': round(float(np.percentile(valid_kappa, 95)), 4),
        'kappa_cv': round(float(np.std(valid_kappa) / np.mean(valid_kappa)), 4),
        'gini': round(gini, 4),
        
        # Orbit geometry
        'theta1_std': round(float(np.std(theta1)), 4),
        'theta2_std': round(float(np.std(theta2)), 4),
        'orbit_spread': round(float(np.sqrt(np.std(theta1)**2 + np.std(theta2)**2)), 4),
        
        # Cycle-to-cycle variability (key diagnostic feature)
        'cycle_gini_mean': round(float(np.mean(cycle_ginis)), 4) if cycle_ginis else 0,
        'cycle_gini_std': round(float(np.std(cycle_ginis)), 4) if cycle_ginis else 0,
        'cycle_gini_cv': round(float(np.std(cycle_ginis) / np.mean(cycle_ginis)), 4) if cycle_ginis and np.mean(cycle_ginis) > 0 else 0,
        'cycle_kappa_mean': round(float(np.mean(cycle_median_kappas)), 4) if cycle_median_kappas else 0,
        'cycle_kappa_std': round(float(np.std(cycle_median_kappas)), 4) if cycle_median_kappas else 0,
        'cycle_kappa_cv': round(float(np.std(cycle_median_kappas) / np.mean(cycle_median_kappas)), 4) if cycle_median_kappas and np.mean(cycle_median_kappas) > 0 else 0,
        'orbit_area_mean': round(float(np.mean(cycle_orbit_areas)), 4) if cycle_orbit_areas else 0,
        'orbit_area_cv': round(float(np.std(cycle_orbit_areas) / np.mean(cycle_orbit_areas)), 4) if cycle_orbit_areas and np.mean(cycle_orbit_areas) > 0 else 0,
        
        # Raw data for figures
        '_theta1': theta1,
        '_theta2': theta2,
        '_kappa': kappa,
        '_position': pos,
        '_velocity': vel,
        '_time': np.arange(n) * dt,
    }
    
    return result


# =====================================================================
# FIGURES
# =====================================================================

CONDITIONS = ['Normal', 'Stenosis', 'Prolapse', 'Flutter', 'Flail']
COND_COLORS = {
    'Normal': '#2196F3',
    'Stenosis': '#FF9800',
    'Prolapse': '#9C27B0',
    'Flutter': '#F44336',
    'Flail': '#E91E63',
}


def fig_valve_traces(all_results):
    """Raw M-mode traces for each condition — 3 cycles."""
    fig, axes = plt.subplots(len(CONDITIONS), 1, figsize=(14, 12), sharex=True)
    
    for idx, cond in enumerate(CONDITIONS):
        ax = axes[idx]
        r = all_results[cond]
        pos = r['_position']
        t = r['_time']
        
        # Show first 3 cycles
        n_show = min(len(pos), int(3 * len(pos) / r['n_cycles']))
        
        ax.plot(t[:n_show], pos[:n_show], color=COND_COLORS[cond], linewidth=0.8)
        ax.set_ylabel('Leaflet\nposition (mm)', fontsize=9)
        ax.set_title(f'{cond}', fontsize=11, fontweight='bold', loc='left',
                     color=COND_COLORS[cond])
        ax.grid(alpha=0.2)
        ax.set_ylim(-12, 35)
    
    axes[-1].set_xlabel('Time (seconds)')
    fig.suptitle('Figure V1: Synthetic Mitral Valve M-Mode Traces',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    
    path = FIGURES_DIR / f'figV1_valve_traces.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_phase_portraits(all_results):
    """Phase-space torus portraits (θ₁ vs θ₂) for each condition."""
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(18, 4))
    
    for idx, cond in enumerate(CONDITIONS):
        ax = axes[idx]
        r = all_results[cond]
        t1 = r['_theta1']
        t2 = r['_theta2']
        kappa = r['_kappa']
        
        # Show 5 cycles worth of trajectory
        n_show = min(len(t1), int(5 * len(t1) / r['n_cycles']))
        
        # Plot trajectory as connected line, colored by curvature
        for i in range(1, n_show):
            d1 = abs(t1[i] - t1[i-1])
            d2 = abs(t2[i] - t2[i-1])
            if d1 > np.pi or d2 > np.pi:
                continue
            ax.plot([t1[i-1], t1[i]], [t2[i-1], t2[i]],
                    color=COND_COLORS[cond], alpha=0.3, linewidth=0.5)
        
        ax.scatter(t1[:n_show], t2[:n_show], c=COND_COLORS[cond],
                   s=1, alpha=0.4, rasterized=True)
        
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(0, 2*np.pi)
        ax.set_aspect('equal')
        ax.set_title(f'{cond}\nκ_med={r["kappa_median"]:.2f}  G={r["gini"]:.3f}',
                     fontsize=10, color=COND_COLORS[cond])
        ax.set_xlabel('θ₁ (position)')
        if idx == 0:
            ax.set_ylabel('θ₂ (velocity)')
        ax.grid(alpha=0.15)
    
    fig.suptitle('Figure V2: Valve Phase-Space Portraits on T²',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    
    path = FIGURES_DIR / f'figV2_phase_portraits.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_curvature_comparison(all_results):
    """Curvature distributions and cycle-to-cycle variability."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel A: Curvature distributions (boxplot)
    ax = axes[0]
    data = []
    labels = []
    colors = []
    for cond in CONDITIONS:
        r = all_results[cond]
        valid = r['_kappa'][r['_kappa'] > 0]
        # Subsample for boxplot
        if len(valid) > 3000:
            valid = np.random.default_rng(42).choice(valid, 3000, replace=False)
        data.append(valid)
        labels.append(f"{cond}\n(κ={r['kappa_median']:.1f})")
        colors.append(COND_COLORS[cond])
    
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Geodesic curvature κ')
    ax.set_title('A. Curvature distribution')
    ax.set_yscale('log')
    
    # Panel B: Gini vs median curvature
    ax = axes[1]
    for cond in CONDITIONS:
        r = all_results[cond]
        ax.scatter(r['kappa_median'], r['gini'], c=COND_COLORS[cond],
                   s=150, edgecolors='black', linewidths=0.5, zorder=5,
                   label=cond)
    ax.set_xlabel('Median κ')
    ax.set_ylabel('Curvature Gini')
    ax.set_title('B. Disease position in κ-Gini space')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Panel C: Cycle-to-cycle variability
    ax = axes[2]
    metrics = ['cycle_kappa_cv', 'cycle_gini_cv', 'orbit_area_cv']
    metric_labels = ['κ CV', 'Gini CV', 'Orbit area CV']
    x = np.arange(len(metrics))
    width = 0.15
    
    for i, cond in enumerate(CONDITIONS):
        r = all_results[cond]
        vals = [r.get(m, 0) for m in metrics]
        ax.bar(x + i*width, vals, width, color=COND_COLORS[cond],
               label=cond, alpha=0.8, edgecolor='black', linewidth=0.3)
    
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylabel('Coefficient of variation')
    ax.set_title('C. Cycle-to-cycle variability')
    ax.legend(fontsize=8, ncol=2)
    
    fig.suptitle('Figure V3: Valve Curvature Analysis — Condition Comparison',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    
    path = FIGURES_DIR / f'figV3_valve_curvature.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def fig_multi_trial_separation(trial_results):
    """Scatter plot of multiple simulated patients per condition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: κ_median vs Gini
    ax = axes[0]
    for cond in CONDITIONS:
        trials = [r for r in trial_results if r['condition'] == cond]
        kappas = [r['kappa_median'] for r in trials]
        ginis = [r['gini'] for r in trials]
        ax.scatter(kappas, ginis, c=COND_COLORS[cond], s=40, alpha=0.7,
                   edgecolors='black', linewidths=0.3, label=f"{cond} (n={len(trials)})")
    
    ax.set_xlabel('Median geodesic curvature κ')
    ax.set_ylabel('Curvature Gini')
    ax.set_title('A. Multi-patient separation (κ vs Gini)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Panel B: orbit spread vs cycle-to-cycle κ CV
    ax = axes[1]
    for cond in CONDITIONS:
        trials = [r for r in trial_results if r['condition'] == cond]
        spreads = [r['orbit_spread'] for r in trials]
        kcvs = [r['cycle_kappa_cv'] for r in trials]
        ax.scatter(spreads, kcvs, c=COND_COLORS[cond], s=40, alpha=0.7,
                   edgecolors='black', linewidths=0.3, label=cond)
    
    ax.set_xlabel('Orbit spread on T²')
    ax.set_ylabel('Cycle-to-cycle curvature CV')
    ax.set_title('B. Multi-patient separation (dynamics)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    fig.suptitle('Figure V4: Diagnostic Separation — 30 Simulated Patients per Condition',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    
    path = FIGURES_DIR / f'figV4_valve_separation.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 65)
    print("Step 08: Synthetic Valve Motion Simulator")
    print("True North Research | Cardiac Torus Pipeline")
    print("=" * 65)
    
    rng = np.random.default_rng(2024)
    
    # ---- Phase 1: Generate canonical traces ----
    print("\nPhase 1: Generating canonical valve traces...")
    all_results = {}
    
    for cond in CONDITIONS:
        print(f"  {cond}...", end=' ')
        trace = generate_valve_trace(cond, n_cycles=50, rng=rng)
        result = analyze_valve_trace(trace)
        all_results[cond] = result
        print(f"κ_med={result['kappa_median']:.2f}  "
              f"Gini={result['gini']:.3f}  "
              f"spread={result['orbit_spread']:.3f}  "
              f"cycle_κ_CV={result['cycle_kappa_cv']:.3f}")
    
    # ---- Phase 2: Multi-trial simulation (30 patients per condition) ----
    print("\nPhase 2: Simulating 30 patients per condition...")
    trial_results = []
    
    for cond in CONDITIONS:
        for trial in range(30):
            seed = hash((cond, trial)) % (2**31)
            trial_rng = np.random.default_rng(seed)
            
            # Vary heart rate across patients
            hr = trial_rng.uniform(58, 92)
            
            trace = generate_valve_trace(cond, n_cycles=30,
                                          heart_rate_bpm=hr, rng=trial_rng)
            result = analyze_valve_trace(trace)
            if result is not None:
                # Remove raw arrays for storage
                clean = {k: v for k, v in result.items() if not k.startswith('_')}
                clean['trial'] = trial
                clean['heart_rate'] = round(hr, 1)
                trial_results.append(clean)
        
        n_done = len([r for r in trial_results if r['condition'] == cond])
        print(f"  {cond}: {n_done} trials completed")
    
    # ---- Phase 3: Statistical tests ----
    print("\nPhase 3: Statistical analysis...")
    
    df_trials = pd.DataFrame(trial_results)
    df_trials.to_csv(RESULTS_DIR / 'valve_simulation_trials.csv', index=False)
    
    # Kruskal-Wallis on median κ
    groups = {}
    for cond in CONDITIONS:
        vals = df_trials[df_trials['condition'] == cond]['kappa_median'].values
        if len(vals) >= 5:
            groups[cond] = vals
    
    if len(groups) >= 2:
        H, p = stats.kruskal(*groups.values())
        print(f"\n  Kruskal-Wallis (median κ): H = {H:.1f}, p = {p:.2e}")
    
    # Pairwise comparisons
    print(f"\n  {'Pair':30s} {'r':>8s} {'p':>12s} {'κ₁':>7s} {'κ₂':>7s}")
    print("  " + "-" * 70)
    
    group_names = list(groups.keys())
    for i in range(len(group_names)):
        for j in range(i+1, len(group_names)):
            g1, g2 = groups[group_names[i]], groups[group_names[j]]
            U, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            r = 1 - 2*U / (len(g1)*len(g2))
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            pair = f"{group_names[i]} vs {group_names[j]}"
            print(f"  {pair:30s} {r:+8.3f} {p:12.2e} "
                  f"{np.median(g1):7.2f} {np.median(g2):7.2f} {sig}")
    
    # ---- Phase 4: Generate figures ----
    print("\nPhase 4: Generating figures...")
    fig_valve_traces(all_results)
    fig_phase_portraits(all_results)
    fig_curvature_comparison(all_results)
    fig_multi_trial_separation(trial_results)
    
    # ---- Summary ----
    print("\n" + "=" * 65)
    print("VALVE TORUS ANALYSIS — SUMMARY")
    print("=" * 65)
    print(f"\n{'Condition':12s} {'κ_med':>7s} {'Gini':>6s} {'Spread':>7s} "
          f"{'CycleCV':>8s} {'OrbArea':>8s}")
    print("-" * 55)
    for cond in CONDITIONS:
        r = all_results[cond]
        print(f"  {cond:10s} {r['kappa_median']:7.2f} {r['gini']:6.3f} "
              f"{r['orbit_spread']:7.3f} {r['cycle_kappa_cv']:8.3f} "
              f"{r['orbit_area_mean']:8.4f}")
    
    print(f"\nPredictions for real data:")
    print(f"  Normal:   tight orbit, highest κ, lowest Gini, lowest cycle CV")
    print(f"  Stenosis: compressed orbit, high κ (rigid), very low cycle CV")
    print(f"  Prolapse: orbit with systolic excursion, moderate κ, κ spike at prolapse")
    print(f"  Flutter:  diffuse orbit, variable κ, highest cycle CV")
    print(f"  Flail:    largest orbit, lowest κ (chaotic), high Gini")
    
    print(f"\nSaved: {RESULTS_DIR / 'valve_simulation_trials.csv'}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == '__main__':
    main()

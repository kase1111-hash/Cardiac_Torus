"""
05_figures.py — Publication-quality figures
True North Research | Cardiac Torus Pipeline

Generates:
  Fig 1: The Cardiac Ramachandran Diagram — all beats on Torus A, colored by class
  Fig 2: Curvature distributions by AAMI class (violin + box)
  Fig 3: Gini coefficient by record, colored by dominant arrhythmia
  Fig 4: Curvature burst profile — example normal vs arrhythmic record
  Fig 5: Quadrant occupancy heatmap (class × quadrant)
  Fig 6: Torus trajectory comparison — 200 consecutive beats, normal vs VT
  Fig 7: Three-torus comparison panel
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, FIGURES_DIR, DPI, FIG_FORMAT

# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
})

CLASS_COLORS = {
    'N': '#2196F3',  # Blue
    'S': '#FF9800',  # Orange
    'V': '#F44336',  # Red
    'F': '#9C27B0',  # Purple
    'Q': '#607D8B',  # Gray
}

CLASS_NAMES = {
    'N': 'Normal',
    'S': 'Supraventricular',
    'V': 'Ventricular',
    'F': 'Fusion',
    'Q': 'Unknown/Paced',
}


def fig1_cardiac_ramachandran(df):
    """The Cardiac Ramachandran Diagram — the signature figure."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    
    # Panel A: All beats, colored by class
    ax = axes[0]
    for cls in ['N', 'S', 'V', 'F']:
        subset = df[df['aami_class'] == cls]
        alpha = 0.03 if cls == 'N' else 0.3
        size = 1 if cls == 'N' else 8
        zorder = 1 if cls == 'N' else 3
        ax.scatter(subset['theta1_A'], subset['theta2_A'],
                   c=CLASS_COLORS[cls], s=size, alpha=alpha,
                   label=f"{CLASS_NAMES[cls]} (n={len(subset):,})",
                   zorder=zorder, rasterized=True)
    
    ax.set_xlabel('θ₁ (RR_pre)')
    ax.set_ylabel('θ₂ (RR_post)')
    ax.set_title('A. Cardiac Ramachandran Diagram\n(Beat-Pair Torus, all records)')
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, 2*np.pi)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=8, markerscale=5)
    
    # Quadrant labels
    for (x, y, label) in [(np.pi/4, np.pi/4, 'Q1\nFast→Fast'),
                           (np.pi/4, 5*np.pi/4, 'Q2\nFast→Slow'),
                           (5*np.pi/4, 5*np.pi/4, 'Q3\nSlow→Slow'),
                           (5*np.pi/4, np.pi/4, 'Q4\nSlow→Fast')]:
        ax.text(x, y, label, ha='center', va='center',
                fontsize=7, color='gray', alpha=0.7)
    
    ax.axhline(np.pi, color='gray', ls='--', alpha=0.3, lw=0.5)
    ax.axvline(np.pi, color='gray', ls='--', alpha=0.3, lw=0.5)
    
    # Panel B: 2D histogram (density)
    ax = axes[1]
    h = ax.hist2d(df['theta1_A'], df['theta2_A'],
                   bins=80, cmap='inferno', norm=mcolors.LogNorm(),
                   range=[[0, 2*np.pi], [0, 2*np.pi]])
    plt.colorbar(h[3], ax=ax, label='Beat count (log scale)')
    ax.set_xlabel('θ₁ (RR_pre)')
    ax.set_ylabel('θ₂ (RR_post)')
    ax.set_title('B. Beat Density on T²')
    ax.set_aspect('equal')
    
    # Panel C: Curvature-colored
    ax = axes[2]
    valid = df[df['kappa_A'] > 0].copy()
    # Cap for visualization
    kappa_cap = np.percentile(valid['kappa_A'], 99)
    valid['kappa_viz'] = np.minimum(valid['kappa_A'], kappa_cap)
    
    sc = ax.scatter(valid['theta1_A'], valid['theta2_A'],
                    c=valid['kappa_viz'], s=1, alpha=0.1,
                    cmap='plasma', rasterized=True)
    plt.colorbar(sc, ax=ax, label='Geodesic curvature κ')
    ax.set_xlabel('θ₁ (RR_pre)')
    ax.set_ylabel('θ₂ (RR_post)')
    ax.set_title('C. Curvature Heatmap on T²')
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, 2*np.pi)
    ax.set_aspect('equal')
    
    fig.suptitle('Figure 1: The Cardiac Ramachandran Diagram — Heartbeat Dynamics on the Torus',
                 fontsize=13, fontweight='bold', y=1.02)
    
    path = FIGURES_DIR / f'fig1_cardiac_ramachandran.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def fig2_curvature_distributions(df):
    """Curvature distributions by AAMI class."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    classes = ['N', 'S', 'V', 'F']
    
    # Panel A: Box plots for each torus
    for idx, (torus, col) in enumerate([('A (RR×RR)', 'kappa_A'),
                                         ('B (RR×Amp)', 'kappa_B'),
                                         ('C (Ratio×Amp)', 'kappa_C')]):
        ax = axes[idx]
        data = []
        positions = []
        colors = []
        labels = []
        
        for i, cls in enumerate(classes):
            subset = df[df['aami_class'] == cls][col]
            valid = subset[subset > 0]
            if len(valid) > 0:
                # Subsample for boxplot if huge
                if len(valid) > 5000:
                    valid = valid.sample(5000, random_state=42)
                data.append(valid.values)
                positions.append(i)
                colors.append(CLASS_COLORS[cls])
                labels.append(f"{cls}\n(n={len(subset[subset>0]):,})")
        
        bp = ax.boxplot(data, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', linewidth=2))
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Geodesic curvature κ')
        ax.set_title(f'Torus {torus}')
        ax.set_yscale('log')
    
    fig.suptitle('Figure 2: Curvature Distribution by Arrhythmia Class',
                 fontsize=13, fontweight='bold')
    
    path = FIGURES_DIR / f'fig2_curvature_distributions.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def fig3_gini_by_record(df_records):
    """Gini coefficient by record."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Gini A vs fraction of ventricular beats
    ax = axes[0]
    for _, row in df_records.iterrows():
        dominant = 'V' if row.get('frac_V', 0) > 0.05 else \
                   'S' if row.get('frac_S', 0) > 0.05 else 'N'
        color = CLASS_COLORS.get(dominant, '#607D8B')
        ax.scatter(row.get('frac_V', 0) * 100, row.get('gini_A', 0),
                   c=color, s=60, edgecolors='black', linewidths=0.5, zorder=3)
        ax.annotate(row['record'], (row.get('frac_V', 0) * 100, row.get('gini_A', 0)),
                    fontsize=6, alpha=0.7)
    
    ax.set_xlabel('Ventricular beats (%)')
    ax.set_ylabel('Curvature Gini (Torus A)')
    ax.set_title('A. Gini vs Ventricular Burden')
    
    # Panel B: Gini A vs Gini B
    ax = axes[1]
    for _, row in df_records.iterrows():
        dominant = 'V' if row.get('frac_V', 0) > 0.05 else \
                   'S' if row.get('frac_S', 0) > 0.05 else 'N'
        color = CLASS_COLORS.get(dominant, '#607D8B')
        ax.scatter(row.get('gini_A', 0), row.get('gini_B', 0),
                   c=color, s=60, edgecolors='black', linewidths=0.5, zorder=3)
    
    ax.set_xlabel('Gini (Torus A: RR×RR)')
    ax.set_ylabel('Gini (Torus B: RR×Amp)')
    ax.set_title('B. Cross-Torus Gini Correlation')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax.legend()
    
    fig.suptitle('Figure 3: Curvature Gini Coefficient by Record',
                 fontsize=13, fontweight='bold')
    
    path = FIGURES_DIR / f'fig3_gini_records.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def fig4_burst_profiles(df):
    """Curvature time series — normal vs arrhythmic."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Find a mostly-normal and a mostly-arrhythmic record
    records = df.groupby('record').agg(
        v_frac=('aami_class', lambda x: (x == 'V').mean()),
        n_beats=('kappa_A', 'count'),
        kappa_mean=('kappa_A', 'mean'),
    )
    
    normal_rec = records[records['v_frac'] < 0.01].sort_values('n_beats', ascending=False)
    arrhythmic_rec = records[records['v_frac'] > 0.10].sort_values('v_frac', ascending=False)
    
    if len(normal_rec) == 0 or len(arrhythmic_rec) == 0:
        print("  [SKIP] Fig 4: couldn't find suitable records for comparison")
        plt.close(fig)
        return
    
    norm_id = normal_rec.index[0]
    arr_id = arrhythmic_rec.index[0]
    
    for row, (rec_id, label) in enumerate([(norm_id, 'Normal'), (arr_id, 'Arrhythmic')]):
        rec_data = df[df['record'] == rec_id].iloc[:500]  # First 500 beats
        
        # Curvature time series
        ax = axes[row, 0]
        colors = [CLASS_COLORS.get(c, '#607D8B') for c in rec_data['aami_class']]
        ax.scatter(range(len(rec_data)), rec_data['kappa_A'],
                   c=colors, s=3, alpha=0.6)
        
        # P90 threshold line
        valid_k = rec_data['kappa_A'][rec_data['kappa_A'] > 0]
        if len(valid_k) > 0:
            threshold = np.percentile(valid_k, 90)
            ax.axhline(threshold, color='red', ls='--', alpha=0.5, label=f'P90={threshold:.3f}')
        
        ax.set_xlabel('Beat index')
        ax.set_ylabel('κ (Torus A)')
        ax.set_title(f'{label} (Record {rec_id})')
        ax.legend(fontsize=8)
        
        # Rolling Gini
        ax = axes[row, 1]
        if 'gini_rolling_A' in rec_data.columns:
            ax.plot(range(len(rec_data)), rec_data['gini_rolling_A'],
                    color='darkblue', alpha=0.8)
        ax.set_xlabel('Beat index')
        ax.set_ylabel('Rolling Gini (30-beat window)')
        ax.set_title(f'{label} — Rolling Gini')
        ax.set_ylim(0, 1)
    
    fig.suptitle('Figure 4: Curvature Burst Profiles — Normal vs Arrhythmic',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    
    path = FIGURES_DIR / f'fig4_burst_profiles.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def fig5_quadrant_heatmap(df):
    """Quadrant × class heatmap."""
    if 'quadrant_A' not in df.columns:
        print("  [SKIP] Fig 5: no quadrant data")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ct = pd.crosstab(df['aami_class'], df['quadrant_A'], normalize='index')
    ct = ct.reindex(index=['N', 'S', 'V', 'F'], columns=['Q1', 'Q2', 'Q3', 'Q4'])
    ct = ct.fillna(0)
    
    im = ax.imshow(ct.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=ct.values.max())
    
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Q1\nFast→Fast', 'Q2\nFast→Slow',
                         'Q3\nSlow→Slow', 'Q4\nSlow→Fast'])
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"{CLASS_NAMES[c]} ({c})" for c in ['N', 'S', 'V', 'F']])
    
    # Annotate cells
    for i in range(4):
        for j in range(4):
            val = ct.values[i, j]
            color = 'white' if val > 0.4 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, color=color, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Fraction of beats')
    ax.set_title('Figure 5: Quadrant Occupancy by Arrhythmia Class',
                 fontsize=13, fontweight='bold')
    
    path = FIGURES_DIR / f'fig5_quadrant_heatmap.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def fig6_trajectory_comparison(df):
    """Torus trajectories — consecutive beats connected by lines."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # Find a normal and arrhythmic record
    records = df.groupby('record').agg(
        v_frac=('aami_class', lambda x: (x == 'V').mean()),
        n_beats=('kappa_A', 'count'),
    )
    
    normal_rec = records[records['v_frac'] < 0.01].sort_values('n_beats', ascending=False)
    arrhythmic_rec = records[records['v_frac'] > 0.10].sort_values('v_frac', ascending=False)
    
    if len(normal_rec) == 0 or len(arrhythmic_rec) == 0:
        plt.close(fig)
        return
    
    for col, (rec_id, label) in enumerate([(normal_rec.index[0], 'Normal Sinus Rhythm'),
                                            (arrhythmic_rec.index[0], 'Ventricular Arrhythmia')]):
        ax = axes[col]
        rec_data = df[df['record'] == rec_id].iloc[50:250]  # 200 beats from middle
        
        t1 = rec_data['theta1_A'].values
        t2 = rec_data['theta2_A'].values
        classes = rec_data['aami_class'].values
        
        # Draw trajectory as connected line segments
        for i in range(len(t1) - 1):
            # Handle wrapping — don't draw line across torus boundary
            d1 = abs(t1[i+1] - t1[i])
            d2 = abs(t2[i+1] - t2[i])
            if d1 > np.pi or d2 > np.pi:
                continue  # Skip wrapped segments
            
            color = CLASS_COLORS.get(classes[i], '#607D8B')
            ax.plot([t1[i], t1[i+1]], [t2[i], t2[i+1]],
                    color=color, alpha=0.4, linewidth=0.5)
        
        # Plot points
        for cls in ['N', 'V', 'S', 'F']:
            mask = classes == cls
            if mask.any():
                ax.scatter(t1[mask], t2[mask], c=CLASS_COLORS[cls],
                           s=12, alpha=0.7, zorder=5, label=CLASS_NAMES[cls],
                           edgecolors='black', linewidths=0.3)
        
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(0, 2*np.pi)
        ax.set_xlabel('θ₁ (RR_pre)')
        ax.set_ylabel('θ₂ (RR_post)')
        ax.set_title(f'{label}\n(Record {rec_id}, beats 50–250)')
        ax.set_aspect('equal')
        ax.axhline(np.pi, color='gray', ls='--', alpha=0.2)
        ax.axvline(np.pi, color='gray', ls='--', alpha=0.2)
        ax.legend(fontsize=8, loc='upper right')
    
    fig.suptitle('Figure 6: Torus Trajectories — 200 Consecutive Beats',
                 fontsize=13, fontweight='bold')
    
    path = FIGURES_DIR / f'fig6_trajectory_comparison.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("Step 05: Generate Figures")
    print("=" * 60)
    
    # Load data
    beat_path = RESULTS_DIR / 'torus_curvature_analyzed.csv'
    record_path = RESULTS_DIR / 'record_curvature_stats.csv'
    
    if not beat_path.exists():
        # Fall back to non-analyzed version
        beat_path = RESULTS_DIR / 'torus_curvature.csv'
    
    if not beat_path.exists():
        print(f"ERROR: {beat_path} not found. Run steps 03-04 first.")
        sys.exit(1)
    
    df = pd.read_csv(beat_path)
    df['record'] = df['record'].astype(int).astype(str)
    print(f"Loaded {len(df):,} beats")
    
    df_records = None
    if record_path.exists():
        df_records = pd.read_csv(record_path)
        df_records['record'] = df_records['record'].astype(int).astype(str)
        print(f"Loaded {len(df_records)} record summaries")
    
    # Generate figures
    print("\nGenerating figures...")
    
    fig1_cardiac_ramachandran(df)
    fig2_curvature_distributions(df)
    
    if df_records is not None:
        fig3_gini_by_record(df_records)
    
    fig4_burst_profiles(df)
    fig5_quadrant_heatmap(df)
    fig6_trajectory_comparison(df)
    
    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == '__main__':
    main()

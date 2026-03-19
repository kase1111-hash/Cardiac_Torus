"""
07_multi_disease_figures.py — Publication figures for multi-disease comparison
True North Research | Cardiac Torus Pipeline

Generates:
  Fig 7: Torus curvature by cardiac condition (box/violin)
  Fig 8: Gini vs median curvature scatter, colored by condition
  Fig 9: Quadrant fingerprint — radar/bar chart per condition
  Fig 10: Diagnostic separation heatmap (pairwise effect sizes)
  Fig 11: Torus spread vs curvature — disease landscape
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, FIGURES_DIR, DPI, FIG_FORMAT

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
})

CONDITION_COLORS = {
    'Normal': '#2196F3',
    'Normal (MITDB)': '#64B5F6',
    'Atrial Fibrillation': '#F44336',
    'Congestive Heart Failure': '#FF9800',
    'Supraventricular Arrhythmia': '#9C27B0',
    'Supraventricular (MITDB)': '#CE93D8',
    'Ventricular Arrhythmia': '#E91E63',
}

CONDITION_ORDER = [
    'Normal', 'Normal (MITDB)',
    'Atrial Fibrillation',
    'Congestive Heart Failure',
    'Supraventricular Arrhythmia', 'Supraventricular (MITDB)',
    'Ventricular Arrhythmia',
]


def fig7_curvature_by_condition(df):
    """Box plots of curvature features by cardiac condition."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    present = [c for c in CONDITION_ORDER if c in df['condition'].unique()]

    for ax_idx, (metric, label) in enumerate([
        ('kappa_median', 'Median geodesic curvature κ'),
        ('gini', 'Curvature Gini coefficient'),
        ('kappa_cv', 'Curvature CV (std/mean)'),
    ]):
        ax = axes[ax_idx]
        data = []
        colors = []
        labels = []

        for cond in present:
            vals = df[df['condition'] == cond][metric].dropna()
            if len(vals) >= 3:
                data.append(vals.values)
                colors.append(CONDITION_COLORS.get(cond, '#607D8B'))
                short = cond.replace('Supraventricular', 'SVA').replace('Congestive Heart Failure', 'CHF')
                short = short.replace('Atrial Fibrillation', 'AF').replace('Ventricular Arrhythmia', 'VA')
                short = short.replace(' (MITDB)', '*')
                labels.append(f"{short}\n(n={len(vals)})")

        if not data:
            continue

        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=True,
                        flierprops=dict(marker='.', markersize=3, alpha=0.3),
                        medianprops=dict(color='black', linewidth=2))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=8, rotation=0)
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Figure 7: Torus Curvature Features by Cardiac Condition',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = FIGURES_DIR / f'fig7_multi_disease_curvature.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def fig8_gini_vs_curvature(df):
    """Scatter: Gini vs median κ, colored by condition."""
    fig, ax = plt.subplots(figsize=(10, 7))

    present = [c for c in CONDITION_ORDER if c in df['condition'].unique()]

    for cond in present:
        g = df[df['condition'] == cond]
        color = CONDITION_COLORS.get(cond, '#607D8B')
        short = cond.replace('Supraventricular', 'SVA').replace('Congestive Heart Failure', 'CHF')
        short = short.replace('Atrial Fibrillation', 'AF').replace('Ventricular Arrhythmia', 'VA')
        ax.scatter(g['kappa_median'], g['gini'],
                   c=color, s=50, alpha=0.7, edgecolors='black', linewidths=0.3,
                   label=f"{short} (n={len(g)})", zorder=3)

    ax.set_xlabel('Median geodesic curvature κ')
    ax.set_ylabel('Curvature Gini coefficient')
    ax.set_title('Figure 8: Disease Landscape on the Curvature-Gini Plane',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

    path = FIGURES_DIR / f'fig8_disease_landscape.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def fig9_quadrant_fingerprints(df):
    """Quadrant occupancy bars for each condition."""
    present = [c for c in CONDITION_ORDER if c in df['condition'].unique()]
    n_conds = len(present)
    if n_conds == 0:
        return

    fig, axes = plt.subplots(1, min(n_conds, 7), figsize=(min(n_conds * 2.5, 18), 4),
                              squeeze=False)
    axes = axes[0]

    quads = ['Q1', 'Q2', 'Q3', 'Q4']
    quad_labels = ['Q1\nFast→\nFast', 'Q2\nFast→\nSlow', 'Q3\nSlow→\nSlow', 'Q4\nSlow→\nFast']

    for idx, cond in enumerate(present[:7]):
        if idx >= len(axes):
            break
        ax = axes[idx]
        g = df[df['condition'] == cond]
        means = [g[f'quad_{q}_frac'].mean() for q in quads]
        color = CONDITION_COLORS.get(cond, '#607D8B')

        bars = ax.bar(range(4), means, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(quad_labels, fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Fraction' if idx == 0 else '')

        short = cond.replace('Supraventricular', 'SVA').replace('Congestive Heart Failure', 'CHF')
        short = short.replace('Atrial Fibrillation', 'AF').replace('Ventricular Arrhythmia', 'VA')
        short = short.replace(' (MITDB)', '*')
        ax.set_title(f"{short}\n(n={len(g)})", fontsize=9)

        # Annotate Q2 value (the diagnostic one)
        if means[1] > 0.01:
            ax.text(1, means[1] + 0.02, f'{means[1]:.2f}', ha='center', fontsize=8,
                    fontweight='bold')

    fig.suptitle('Figure 9: Quadrant Fingerprints by Cardiac Condition',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    path = FIGURES_DIR / f'fig9_quadrant_fingerprints.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def fig10_separation_heatmap(df):
    """Pairwise effect size heatmap."""
    from scipy import stats

    present = sorted([c for c in df['condition'].unique()
                      if len(df[df['condition'] == c]) >= 5])

    n = len(present)
    if n < 2:
        return

    effect_matrix = np.zeros((n, n))
    p_matrix = np.ones((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            g1 = df[df['condition'] == present[i]]['kappa_median'].dropna().values
            g2 = df[df['condition'] == present[j]]['kappa_median'].dropna().values
            if len(g1) < 3 or len(g2) < 3:
                continue
            U, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            r = 1 - 2*U / (len(g1) * len(g2))
            effect_matrix[i, j] = r
            p_matrix[i, j] = p

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.abs(effect_matrix), cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')

    short_names = []
    for c in present:
        s = c.replace('Supraventricular', 'SVA').replace('Congestive Heart Failure', 'CHF')
        s = s.replace('Atrial Fibrillation', 'AF').replace('Ventricular Arrhythmia', 'VA')
        s = s.replace(' (MITDB)', '*')
        short_names.append(s)

    ax.set_xticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_names, fontsize=9)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, '—', ha='center', va='center', fontsize=10)
                continue
            val = effect_matrix[i, j]
            p = p_matrix[i, j]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:+.2f}\n{sig}', ha='center', va='center',
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label='|Effect size r| (rank-biserial)')
    ax.set_title('Figure 10: Pairwise Diagnostic Separation (Median κ)',
                 fontsize=13, fontweight='bold')

    path = FIGURES_DIR / f'fig10_separation_heatmap.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def fig11_torus_landscape(df):
    """Multi-feature scatter: torus spread vs speed CV, sized by Gini."""
    if 'torus_spread' not in df.columns or df['torus_spread'].isna().all():
        print("  [SKIP] Fig 11: no torus_spread data")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    present = [c for c in CONDITION_ORDER if c in df['condition'].unique()]

    for cond in present:
        g = df[df['condition'] == cond].dropna(subset=['torus_spread', 'torus_speed_cv'])
        if len(g) < 2:
            continue
        color = CONDITION_COLORS.get(cond, '#607D8B')
        sizes = 30 + 200 * g['gini']
        short = cond.replace('Supraventricular', 'SVA').replace('Congestive Heart Failure', 'CHF')
        short = short.replace('Atrial Fibrillation', 'AF').replace('Ventricular Arrhythmia', 'VA')
        ax.scatter(g['torus_spread'], g['torus_speed_cv'],
                   c=color, s=sizes, alpha=0.6, edgecolors='black', linewidths=0.3,
                   label=f"{short} (n={len(g)})", zorder=3)

    ax.set_xlabel('Torus spread (σ of angular position)')
    ax.set_ylabel('Torus speed CV (variability of beat-to-beat velocity)')
    ax.set_title('Figure 11: Disease Landscape — Torus Dynamics\n'
                 '(point size ∝ curvature Gini)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

    path = FIGURES_DIR / f'fig11_torus_dynamics.{FIG_FORMAT}'
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("Step 07: Multi-Disease Figures")
    print("=" * 60)

    csv_path = RESULTS_DIR / 'multi_disease_records.csv'
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run 06_multi_disease.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records across {df['condition'].nunique()} conditions")

    print("\nGenerating figures...")
    fig7_curvature_by_condition(df)
    fig8_gini_vs_curvature(df)
    fig9_quadrant_fingerprints(df)
    fig10_separation_heatmap(df)
    fig11_torus_landscape(df)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == '__main__':
    main()

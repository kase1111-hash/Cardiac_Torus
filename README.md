# Cardiac Curvature on the Beat-Pair Torus
## Kase Branham

A geometric framework for arrhythmia classification using torus-mapped 
cardiac trajectories and geodesic curvature analysis.

### The Idea

Every existing cardiac monitor analyzes the ECG as a **waveform** — 
amplitude vs. time, PQRST morphology, frequency domain, or flat-plane 
phase-space attractors. This project maps consecutive heartbeats onto a 
**torus** (T²) and computes **geodesic curvature** of the resulting 
trajectory — the same mathematical framework that produced the Backbone 
Curvature Atlas for protein structure.

A healthy heart traces a tight, low-curvature orbit on the torus. 
Arrhythmias produce curvature spikes, regime shifts, and altered Gini 
distributions that are invisible in flat-plane analysis.

### Setup

```bash
# Create environment
python -m venv cardiac_env
# Windows:
cardiac_env\Scripts\activate
# Linux/Mac:
source cardiac_env/bin/activate

# Install dependencies
pip install wfdb numpy scipy matplotlib pandas seaborn
```

### Pipeline

Run scripts in order:

| Step | Script | Description |
|------|--------|-------------|
| 01 | `01_download_mitbih.py` | Download MIT-BIH Arrhythmia Database from PhysioNet |
| 02 | `02_extract_beats.py` | Extract beat-by-beat features (RR, amplitude, morphology) |
| 03 | `03_torus_mapping.py` | Map beat pairs onto T² and compute geodesic curvature |
| 04 | `04_curvature_analysis.py` | Gini coefficients, burst detection, regime classification |
| 05 | `05_figures.py` | Publication-quality figures |

### Data

- **Source:** MIT-BIH Arrhythmia Database (PhysioNet, free, open access)
- **48 records**, 47 subjects, ~110,000 annotated beats
- **360 Hz**, 2-channel ambulatory ECG
- Beat annotations by 2+ cardiologists

### Key Concepts

- **Beat-Pair Torus (T²):** Consecutive beats mapped to angular coords 
  (θ₁, θ₂) where θ₁ = normalized RR interval, θ₂ = normalized R-peak 
  amplitude ratio. Both are periodic/bounded → natural torus topology.
  
- **Geodesic Curvature (κ):** How sharply the cardiac trajectory bends 
  on T². High κ = sudden regime change. Low κ = stable rhythm.

- **Curvature Gini (G_κ):** Concentration of curvature. High Gini = 
  curvature concentrated at few beats (intermittent disruption). 
  Low Gini = distributed curvature (uniform dynamics).

- **Curvature Bursts:** Contiguous segments of high-κ beats, analogous 
  to the theta burst dynamics in the EEG dementia work.

"""
config.py — Shared configuration for Cardiac Torus pipeline
True North Research
"""

from pathlib import Path

# === DIRECTORIES ===
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data" / "mitbih"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === MIT-BIH RECORDS ===
# All 48 records in the database
# 100-series: roughly half random, half selected for arrhythmias
# 200-series: selected for clinically significant arrhythmias
ALL_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107',
    '108', '109', '111', '112', '113', '114', '115', '116',
    '117', '118', '119', '121', '122', '123', '124',
    '200', '201', '202', '203', '205', '207', '208', '209',
    '210', '212', '213', '214', '215', '217', '219', '220',
    '221', '222', '223', '228', '230', '231', '232', '233',
    '234',
]

# === SAMPLING ===
FS = 360  # MIT-BIH sampling frequency (Hz)

# === BEAT ANNOTATION SYMBOLS ===
# MIT-BIH uses single-character codes for beat types
# See: https://www.physionet.org/physiobank/annotations.shtml
NORMAL_BEATS = {'N', 'L', 'R', 'e', 'j'}  # Normal, LBBB, RBBB, atrial escape, junctional escape
PVC_BEATS = {'V'}                            # Premature ventricular contraction
APC_BEATS = {'A', 'a', 'S', 'J'}           # Atrial premature, aberrated APC, supra-V, junctional premature
FUSION_BEATS = {'F'}                         # Fusion of ventricular and normal
PACED_BEATS = {'/'}                          # Paced beat
UNKNOWN_BEATS = {'Q'}                        # Unclassifiable

# AAMI standard groupings (for clinical comparison)
AAMI_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',  # Normal
    'A': 'S', 'a': 'S', 'S': 'S', 'J': 'S',              # Supraventricular
    'V': 'V',                                               # Ventricular
    'F': 'F',                                               # Fusion
    '/': 'Q', 'Q': 'Q',                                    # Unknown/Paced
}

# === TORUS MAPPING PARAMETERS ===
# RR interval bounds (ms) for angular mapping
RR_MIN_MS = 200    # ~300 bpm (physiological minimum)
RR_MAX_MS = 2000   # ~30 bpm (physiological maximum)

# Amplitude ratio bounds for angular mapping
AMP_RATIO_MIN = 0.2   # Minimum R-peak amplitude ratio (beat/median)
AMP_RATIO_MAX = 3.0   # Maximum R-peak amplitude ratio

# === CURVATURE PARAMETERS ===
BURST_PERCENTILE = 90     # Threshold for high-curvature bursts
BURST_MIN_LENGTH = 2      # Minimum beats in a burst
BURST_MERGE_GAP = 2       # Merge bursts separated by ≤ this many beats

# === FIGURE PARAMETERS ===
DPI = 300
FIG_FORMAT = 'png'

"""
run_pipeline.py — Run the complete Cardiac Torus pipeline
True North Research

Usage:
  python run_pipeline.py          # Run all steps
  python run_pipeline.py 3 4 5    # Run specific steps
"""

import sys
import subprocess
import time
from pathlib import Path

STEPS = {
    1: ('01_download_mitbih.py', 'Download MIT-BIH from PhysioNet'),
    2: ('02_extract_beats.py', 'Extract beat-by-beat features'),
    3: ('03_torus_mapping.py', 'Map beats to torus + compute curvature'),
    4: ('04_curvature_analysis.py', 'Gini, bursts, statistics'),
    5: ('05_figures.py', 'Generate publication figures'),
    6: ('06_multi_disease.py', 'Download + analyze multi-disease databases'),
    7: ('07_multi_disease_figures.py', 'Multi-disease comparison figures'),
    8: ('08_valve_simulator.py', 'Synthetic valve motion simulator'),
    9: ('09_noise_robustness.py', 'PPG timing jitter robustness test'),
    10: ('10_chf_replication.py', 'CHF replication — independent cohort'),
    11: ('11_echonet_torus.py', 'EchoNet-Dynamic valve/wall motion torus'),
}

def main():
    script_dir = Path(__file__).parent
    
    if len(sys.argv) > 1:
        steps_to_run = [int(s) for s in sys.argv[1:]]
    else:
        steps_to_run = sorted(STEPS.keys())
    
    print("=" * 60)
    print("  CARDIAC CURVATURE ON THE BEAT-PAIR TORUS")
    print("  True North Research")
    print("=" * 60)
    print()
    
    for step_num in steps_to_run:
        if step_num not in STEPS:
            print(f"Unknown step: {step_num}")
            continue
        
        script, description = STEPS[step_num]
        script_path = script_dir / script
        
        print(f"\n{'='*60}")
        print(f"  Step {step_num}: {description}")
        print(f"{'='*60}")
        
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_dir),
        )
        elapsed = time.time() - t0
        
        if result.returncode != 0:
            print(f"\n  *** Step {step_num} FAILED (exit code {result.returncode}) ***")
            print(f"  Fix the issue and re-run: python run_pipeline.py {step_num}")
            sys.exit(1)
        
        print(f"\n  Step {step_num} completed in {elapsed:.1f}s")
    
    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("  Results: ./results/")
    print("  Figures: ./figures/")
    print("=" * 60)


if __name__ == '__main__':
    main()

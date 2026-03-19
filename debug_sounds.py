import os
from pathlib import Path

data_dir = Path(r"G:\Cardiac_Torus\data\heart_sounds")

for subset in sorted(data_dir.iterdir()):
    if not subset.is_dir():
        continue
    
    files = list(subset.iterdir())
    wav_count = sum(1 for f in files if f.suffix.lower() == '.wav')
    non_wav = [f.name for f in files if f.suffix.lower() != '.wav']
    
    print(f"\n{subset.name}: {wav_count} wav files")
    print(f"  Other files: {non_wav[:20]}")
    
    # Check for any csv, txt, or reference-like files
    for f in files:
        if f.suffix.lower() in ['.csv', '.txt', '.hea', '.dat']:
            print(f"  Found: {f.name} ({f.stat().st_size} bytes)")
            # Show first few lines
            try:
                with open(f, 'r') as fh:
                    lines = fh.readlines()[:5]
                    for line in lines:
                        print(f"    {line.rstrip()}")
            except:
                print(f"    (binary or unreadable)")
    
    # Check if wav files have companion .hea files
    sample_wavs = [f for f in files if f.suffix.lower() == '.wav'][:3]
    for w in sample_wavs:
        hea = w.with_suffix('.hea')
        dat = w.with_suffix('.dat')
        print(f"  Sample: {w.name} | .hea exists: {hea.exists()} | .dat exists: {dat.exists()}")

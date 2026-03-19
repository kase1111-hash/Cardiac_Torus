"""
Quick diagnostic: what does a CTU-UHB CTG file actually look like?
Run this and paste the output back.
"""
from pathlib import Path
import struct

data_dir = Path("data/ctg")

# Find first .dat and .hea
dat_files = sorted(data_dir.glob("*.dat"))
if not dat_files:
    dat_files = sorted(data_dir.rglob("*.dat"))
    
if not dat_files:
    print("No .dat files found!")
    print(f"Contents of {data_dir}:")
    for p in sorted(data_dir.iterdir())[:20]:
        print(f"  {p.name} ({'dir' if p.is_dir() else f'{p.stat().st_size} bytes'})")
else:
    print(f"Found {len(dat_files)} .dat files")
    print(f"First 5: {[f.name for f in dat_files[:5]]}")
    
    # Read first .hea file completely
    hea = dat_files[0].with_suffix('.hea')
    print(f"\n=== {hea.name} ===")
    if hea.exists():
        with open(hea, 'r') as f:
            for i, line in enumerate(f):
                print(f"  {i}: {line.rstrip()}")
    else:
        print("  .hea not found!")
    
    # Read first .dat file - first 100 bytes
    dat = dat_files[0]
    print(f"\n=== {dat.name} ({dat.stat().st_size} bytes) ===")
    with open(dat, 'rb') as f:
        raw = f.read(200)
    
    print(f"  First 40 bytes (hex): {raw[:40].hex()}")
    print(f"  First 40 bytes (raw): {raw[:40]}")
    
    # Try reading as int16
    vals_i16 = struct.unpack(f'<{min(20, len(raw)//2)}h', raw[:min(40, len(raw))])
    print(f"  As int16 LE: {vals_i16}")
    
    # Try reading as uint16
    vals_u16 = struct.unpack(f'<{min(20, len(raw)//2)}H', raw[:min(40, len(raw))])
    print(f"  As uint16 LE: {vals_u16}")
    
    # Try reading as int16 big-endian
    vals_i16be = struct.unpack(f'>{min(20, len(raw)//2)}h', raw[:min(40, len(raw))])
    print(f"  As int16 BE: {vals_i16be}")
    
    # Check if wfdb is available
    try:
        import wfdb
        print(f"\n=== wfdb available, trying to read ===")
        record_path = str(dat.with_suffix(''))
        rec = wfdb.rdrecord(record_path)
        print(f"  Channels: {rec.sig_name}")
        print(f"  Shape: {rec.p_signal.shape}")
        print(f"  Fs: {rec.fs}")
        print(f"  First 10 samples ch0: {rec.p_signal[:10, 0]}")
        print(f"  First 10 samples ch1: {rec.p_signal[:10, 1]}")
        print(f"  Comments: {rec.comments}")
    except ImportError:
        print("\n  wfdb not installed. Try: pip install wfdb")
    except Exception as e:
        print(f"\n  wfdb error: {e}")

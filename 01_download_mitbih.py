"""
01_download_mitbih.py — Download MIT-BIH Arrhythmia Database
True North Research | Cardiac Torus Pipeline

Downloads all 48 records from PhysioNet. Each record includes:
  - .dat (signal data, 2 channels, 360 Hz, 30 min)
  - .hea (header with signal metadata)
  - .atr (beat annotations by cardiologists)

Total download: ~100 MB
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, ALL_RECORDS

try:
    import wfdb
except ImportError:
    print("Install wfdb: pip install wfdb")
    sys.exit(1)


def main():
    print("=" * 60)
    print("Step 01: Download MIT-BIH Arrhythmia Database")
    print("=" * 60)
    print(f"Target directory: {DATA_DIR}")
    print(f"Records to download: {len(ALL_RECORDS)}")
    print()

    downloaded = 0
    failed = []

    for rec in ALL_RECORDS:
        try:
            wfdb.dl_database(
                'mitdb',
                dl_dir=str(DATA_DIR),
                records=[rec],
                overwrite=False
            )
            downloaded += 1
            print(f"  [{downloaded:2d}/{len(ALL_RECORDS)}] Record {rec} ✓")
        except Exception as e:
            failed.append(rec)
            print(f"  [FAIL] Record {rec}: {e}")

    print()
    print(f"Downloaded: {downloaded}")
    if failed:
        print(f"Failed: {len(failed)} — {failed}")
    else:
        print("All records downloaded successfully.")

    # Verify by reading one record
    try:
        test_rec = wfdb.rdrecord(str(DATA_DIR / '100'))
        test_ann = wfdb.rdann(str(DATA_DIR / '100'), 'atr')
        print(f"\nVerification (Record 100):")
        print(f"  Signals: {test_rec.sig_name}")
        print(f"  Samples: {test_rec.sig_len:,}")
        print(f"  Duration: {test_rec.sig_len / test_rec.fs:.0f} sec")
        print(f"  Beats annotated: {len(test_ann.sample):,}")
    except Exception as e:
        print(f"\nVerification failed: {e}")


if __name__ == '__main__':
    main()

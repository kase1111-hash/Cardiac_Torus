"""
download_echonet_lvh_v2.py — Resumable downloader with progress bar
Handles EchoNet-LVH as a single 69GB zip file.

Usage:
  python download_echonet_lvh_v2.py
  python download_echonet_lvh_v2.py --output G:\EchoNet-LVH
  
After download completes, it auto-extracts the zip.
"""

import os
import sys
import time
import zipfile
import argparse
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)

CONTAINER_SAS_URL = (
    "https://aimistanforddatasets01.blob.core.windows.net/echonetlvh"
    "?sv=2019-02-02&sr=c"
    "&sig=PHm%2B%2BLB5YkzSWMN3egHZ69omrdQ1gIvyl2SFuSQsQXs%3D"
    "&st=2026-03-18T22:34:13Z"
    "&se=2026-04-17T22:39:13Z"
    "&sp=rl"
)

BLOB_NAME = "EchoNet-LVH.zip"
CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB
MAX_RETRIES = 50
RETRY_BASE_DELAY = 5
CONNECTION_TIMEOUT = 30
READ_TIMEOUT = 120


def format_size(bytes_val):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m {seconds%60:.0f}s"
    else:
        return f"{seconds/3600:.0f}h {(seconds%3600)/60:.0f}m"


def download_with_resume(url, output_path, total_size=None):
    """Download a file with resume support and live progress."""
    
    existing_size = 0
    if output_path.exists():
        existing_size = output_path.stat().st_size
        if total_size and existing_size >= total_size:
            print(f"  Already complete: {format_size(existing_size)}")
            return True
        elif existing_size > 0:
            print(f"  Resuming from {format_size(existing_size)} "
                  f"({100*existing_size/total_size:.1f}%)" if total_size else "")

    attempt = 0
    start_time = time.time()
    session_bytes = 0

    while attempt < MAX_RETRIES:
        try:
            # Refresh file size (may have grown from previous attempt)
            if output_path.exists():
                existing_size = output_path.stat().st_size
                if total_size and existing_size >= total_size:
                    print(f"\n  Download complete: {format_size(existing_size)}")
                    return True

            headers = {}
            if existing_size > 0:
                headers['Range'] = f'bytes={existing_size}-'

            resp = requests.get(
                url, headers=headers, stream=True,
                timeout=(CONNECTION_TIMEOUT, READ_TIMEOUT)
            )

            if resp.status_code == 416:
                print(f"\n  Complete (server confirms all bytes received)")
                return True

            resp.raise_for_status()

            # Determine total size from response if not known
            if total_size is None:
                content_range = resp.headers.get('Content-Range', '')
                if '/' in content_range:
                    total_size = int(content_range.split('/')[-1])
                else:
                    total_size = int(resp.headers.get('Content-Length', 0)) + existing_size

            mode = 'ab' if existing_size > 0 and resp.status_code == 206 else 'wb'
            bytes_written = existing_size if mode == 'ab' else 0
            last_print = time.time()

            with open(output_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)
                        session_bytes += len(chunk)

                        # Print progress every 2 seconds
                        now = time.time()
                        if now - last_print >= 2:
                            elapsed = now - start_time
                            speed = session_bytes / elapsed if elapsed > 0 else 0
                            remaining = total_size - bytes_written if total_size else 0
                            eta = remaining / speed if speed > 0 else 0
                            pct = 100 * bytes_written / total_size if total_size else 0

                            bar_width = 30
                            filled = int(bar_width * pct / 100)
                            bar = '\u2588' * filled + '\u2591' * (bar_width - filled)

                            print(f"\r  [{bar}] {pct:5.1f}%  "
                                  f"{format_size(bytes_written)}/{format_size(total_size)}  "
                                  f"{speed/(1024*1024):.1f} MB/s  "
                                  f"ETA {format_time(eta)}    ",
                                  end='', flush=True)
                            last_print = now

            # Verify
            final_size = output_path.stat().st_size
            if total_size and final_size >= total_size:
                elapsed = time.time() - start_time
                avg_speed = session_bytes / elapsed if elapsed > 0 else 0
                print(f"\n\n  Download complete!")
                print(f"  Size: {format_size(final_size)}")
                print(f"  Time: {format_time(elapsed)}")
                print(f"  Avg speed: {avg_speed/(1024*1024):.1f} MB/s")
                return True
            else:
                # Incomplete — connection dropped during transfer
                attempt += 1
                delay = min(RETRY_BASE_DELAY * (2 ** min(attempt, 6)), 120)
                print(f"\n  Connection lost at {format_size(final_size)}. "
                      f"Retry {attempt}/{MAX_RETRIES} in {delay}s...")
                time.sleep(delay)

        except KeyboardInterrupt:
            current = output_path.stat().st_size if output_path.exists() else 0
            pct = 100 * current / total_size if total_size else 0
            print(f"\n\n  Paused at {format_size(current)} ({pct:.1f}%)")
            print(f"  Re-run this script to resume.")
            return False

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
                ConnectionResetError,
                IOError) as e:

            attempt += 1
            delay = min(RETRY_BASE_DELAY * (2 ** min(attempt, 6)), 120)
            current = output_path.stat().st_size if output_path.exists() else 0
            pct = 100 * current / total_size if total_size else 0

            print(f"\n  Error: {type(e).__name__} at {format_size(current)} ({pct:.1f}%)")
            print(f"  Retry {attempt}/{MAX_RETRIES} in {delay}s...")
            time.sleep(delay)

    print(f"\n  FAILED after {MAX_RETRIES} retries.")
    return False


def extract_zip(zip_path, output_dir):
    """Extract zip with progress."""
    print(f"\nExtracting {zip_path.name}...")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        members = zf.namelist()
        print(f"  {len(members)} files in archive")

        for i, member in enumerate(members):
            zf.extract(member, output_dir)
            if (i + 1) % 500 == 0 or i == len(members) - 1:
                pct = 100 * (i + 1) / len(members)
                print(f"\r  Extracting: {i+1}/{len(members)} ({pct:.0f}%)    ",
                      end='', flush=True)

    print(f"\n  Extraction complete!")
    print(f"  Files extracted to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=str(Path(r"G:\EchoNet-LVH")))
    parser.add_argument('--skip-extract', action='store_true',
                        help='Download only, do not extract')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EchoNet-LVH Resumable Downloader v2")
    print("=" * 60)

    zip_path = output_dir / BLOB_NAME
    blob_url = f"https://aimistanforddatasets01.blob.core.windows.net/echonetlvh/{BLOB_NAME}?{CONTAINER_SAS_URL.split('?')[1]}"

    # Get file size first
    print(f"\nChecking file size...")
    try:
        head = requests.head(blob_url, timeout=CONNECTION_TIMEOUT)
        total_size = int(head.headers.get('Content-Length', 0))
        print(f"  {BLOB_NAME}: {format_size(total_size)}")
    except Exception as e:
        print(f"  Could not get size: {e}")
        total_size = 73_789_000_000  # approximate

    # Download
    print(f"\nDownloading to: {zip_path}")
    print(f"  Ctrl+C to pause, re-run to resume\n")

    success = download_with_resume(blob_url, zip_path, total_size)

    if not success:
        print("\nDownload incomplete. Re-run to resume.")
        return

    # Extract
    if not args.skip_extract:
        extract_zip(zip_path, output_dir)

        # Check what we got
        video_dir = None
        for root, dirs, files in os.walk(output_dir):
            avis = [f for f in files if f.endswith('.avi')]
            csvs = [f for f in files if f.endswith('.csv')]
            if avis:
                video_dir = root
                print(f"\n  Found {len(avis)} video files in {root}")
            if csvs:
                for csv in csvs:
                    print(f"  Found: {os.path.join(root, csv)}")
            if video_dir:
                break

        print(f"\n  Ready for Step 13:")
        print(f"  python 13_echonet_lvh.py --data_dir \"{output_dir}\" --max_videos 500")
    else:
        print(f"\n  Zip saved. Extract manually or re-run without --skip-extract")


if __name__ == '__main__':
    main()

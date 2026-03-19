"""
download_echonet_lvh.py — Resumable downloader for EchoNet-LVH
True North Research

Azure Blob Storage downloads with:
  - Automatic resume on connection failure
  - Per-file integrity tracking
  - Retry with exponential backoff
  - Progress reporting
  - Parallel downloads (optional)

Usage:
  python download_echonet_lvh.py
  python download_echonet_lvh.py --output G:\EchoNet-LVH --workers 4
"""

import os
import sys
import json
import time
import argparse
import hashlib
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    print("Install requests: pip install requests")
    sys.exit(1)


# =====================================================================
# CONFIGURATION
# =====================================================================

# The SAS URL for the container
CONTAINER_SAS_URL = (
    "https://aimistanforddatasets01.blob.core.windows.net/echonetlvh"
    "?sv=2019-02-02&sr=c"
    "&sig=PHm%2B%2BLB5YkzSWMN3egHZ69omrdQ1gIvyl2SFuSQsQXs%3D"
    "&st=2026-03-18T22:34:13Z"
    "&se=2026-04-17T22:39:13Z"
    "&sp=rl"
)

DEFAULT_OUTPUT_DIR = Path(r"G:\EchoNet-LVH")
CHUNK_SIZE = 8 * 1024 * 1024   # 8 MB chunks
MAX_RETRIES = 10
RETRY_BASE_DELAY = 5            # seconds, doubles each retry
CONNECTION_TIMEOUT = 30
READ_TIMEOUT = 120
PROGRESS_FILE = "download_progress.json"


# =====================================================================
# AZURE BLOB LISTING
# =====================================================================

def parse_sas_url(sas_url):
    """Split container URL and SAS token."""
    parsed = urlparse(sas_url)
    base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    sas_token = parsed.query
    return base_url, sas_token


def list_blobs(container_url, sas_token, prefix=""):
    """
    List all blobs in an Azure container using the REST API.
    Handles pagination via the marker parameter.
    """
    blobs = []
    marker = None

    while True:
        params = {
            'restype': 'container',
            'comp': 'list',
            'maxresults': '5000',
        }
        if prefix:
            params['prefix'] = prefix
        if marker:
            params['marker'] = marker

        # Append SAS token
        url = f"{container_url}?{sas_token}&" + urlencode(params)

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(url, timeout=(CONNECTION_TIMEOUT, READ_TIMEOUT))
                resp.raise_for_status()
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"  List retry {attempt+1}/{MAX_RETRIES} in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    raise

        # Parse XML response (simple parsing without lxml)
        content = resp.text

        # Extract blob names
        import re
        names = re.findall(r'<Name>(.*?)</Name>', content)
        sizes = re.findall(r'<Content-Length>(.*?)</Content-Length>', content)

        for i, name in enumerate(names):
            size = int(sizes[i]) if i < len(sizes) else 0
            blobs.append({'name': name, 'size': size})

        # Check for continuation
        next_marker = re.findall(r'<NextMarker>(.*?)</NextMarker>', content)
        if next_marker and next_marker[0]:
            marker = next_marker[0]
        else:
            break

    return blobs


# =====================================================================
# RESUMABLE DOWNLOAD
# =====================================================================

def download_blob(container_url, sas_token, blob_name, blob_size,
                  output_dir, progress_tracker):
    """
    Download a single blob with resume support.
    
    Uses HTTP Range headers to resume from where we left off.
    """
    output_path = output_dir / blob_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if already complete
    if output_path.exists():
        existing_size = output_path.stat().st_size
        if existing_size == blob_size and blob_size > 0:
            return {'name': blob_name, 'status': 'already_complete', 'size': blob_size}
        elif existing_size > blob_size and blob_size > 0:
            # File is larger than expected — re-download
            output_path.unlink()
            existing_size = 0
        # else: partial download, resume from existing_size
    else:
        existing_size = 0

    blob_url = f"{container_url}/{blob_name}?{sas_token}"

    for attempt in range(MAX_RETRIES):
        try:
            # Get current file size (may have grown from previous attempt)
            if output_path.exists():
                existing_size = output_path.stat().st_size
                if existing_size == blob_size and blob_size > 0:
                    return {'name': blob_name, 'status': 'complete', 'size': blob_size}

            headers = {}
            if existing_size > 0:
                headers['Range'] = f'bytes={existing_size}-'

            resp = requests.get(
                blob_url,
                headers=headers,
                stream=True,
                timeout=(CONNECTION_TIMEOUT, READ_TIMEOUT)
            )

            # 206 = Partial Content (resume), 200 = Full content
            if resp.status_code == 416:
                # Range not satisfiable — file is already complete
                return {'name': blob_name, 'status': 'complete', 'size': existing_size}

            resp.raise_for_status()

            mode = 'ab' if existing_size > 0 and resp.status_code == 206 else 'wb'

            bytes_downloaded = existing_size if mode == 'ab' else 0

            with open(output_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)

            # Verify
            final_size = output_path.stat().st_size
            if blob_size > 0 and final_size != blob_size:
                # Incomplete — will retry
                continue

            return {'name': blob_name, 'status': 'complete', 'size': final_size}

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
                ConnectionResetError,
                IOError) as e:

            delay = min(RETRY_BASE_DELAY * (2 ** attempt), 120)
            if attempt < MAX_RETRIES - 1:
                current = output_path.stat().st_size if output_path.exists() else 0
                pct = f" ({100*current/blob_size:.0f}%)" if blob_size > 0 else ""
                print(f"    [{blob_name}] Retry {attempt+1}/{MAX_RETRIES} "
                      f"in {delay}s{pct}: {type(e).__name__}")
                time.sleep(delay)
            else:
                return {'name': blob_name, 'status': 'failed', 'error': str(e)}

    return {'name': blob_name, 'status': 'failed', 'error': 'max retries exceeded'}


# =====================================================================
# PROGRESS TRACKING
# =====================================================================

class ProgressTracker:
    def __init__(self, progress_file):
        self.file = Path(progress_file)
        self.data = self._load()

    def _load(self):
        if self.file.exists():
            try:
                with open(self.file) as f:
                    return json.load(f)
            except:
                return {'completed': [], 'failed': []}
        return {'completed': [], 'failed': []}

    def save(self):
        with open(self.file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def is_complete(self, name):
        return name in self.data['completed']

    def mark_complete(self, name):
        if name not in self.data['completed']:
            self.data['completed'].append(name)
        self.save()

    def mark_failed(self, name, error):
        self.data['failed'].append({'name': name, 'error': error})
        self.save()


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='Download EchoNet-LVH dataset')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help='Output directory')
    parser.add_argument('--workers', type=int, default=2,
                        help='Parallel download workers (1-8)')
    parser.add_argument('--list-only', action='store_true',
                        help='Just list blobs without downloading')
    parser.add_argument('--prefix', type=str, default='',
                        help='Only download blobs matching this prefix')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("EchoNet-LVH Resumable Downloader")
    print("=" * 65)

    container_url, sas_token = parse_sas_url(CONTAINER_SAS_URL)
    print(f"Container: {container_url}")
    print(f"Output: {output_dir}")

    # List all blobs
    print("\nListing blobs in container...")
    blobs = list_blobs(container_url, sas_token, prefix=args.prefix)
    print(f"Found {len(blobs)} blobs")

    if not blobs:
        print("No blobs found! Check if the SAS token has expired.")
        sys.exit(1)

    # Summarize
    total_size = sum(b['size'] for b in blobs)
    video_count = sum(1 for b in blobs if b['name'].endswith('.avi'))
    csv_count = sum(1 for b in blobs if b['name'].endswith('.csv'))

    print(f"  Total size: {total_size / (1024**3):.2f} GB")
    print(f"  Videos (.avi): {video_count}")
    print(f"  CSV files: {csv_count}")

    # Show sample of file names
    print(f"\n  Sample files:")
    for b in blobs[:10]:
        size_mb = b['size'] / (1024**2)
        print(f"    {b['name']:60s} {size_mb:8.1f} MB")
    if len(blobs) > 10:
        print(f"    ... and {len(blobs) - 10} more")

    if args.list_only:
        return

    # Progress tracking
    progress_file = output_dir / PROGRESS_FILE
    tracker = ProgressTracker(progress_file)
    previously_done = len(tracker.data['completed'])
    print(f"\nPreviously completed: {previously_done}")

    # Filter out already completed
    to_download = [b for b in blobs if not tracker.is_complete(b['name'])]

    # Also skip files that exist with correct size
    still_needed = []
    for b in to_download:
        fpath = output_dir / b['name']
        if fpath.exists() and fpath.stat().st_size == b['size'] and b['size'] > 0:
            tracker.mark_complete(b['name'])
        else:
            still_needed.append(b)

    newly_skipped = len(to_download) - len(still_needed)
    if newly_skipped > 0:
        print(f"Found {newly_skipped} already-downloaded files on disk")

    to_download = still_needed
    download_size = sum(b['size'] for b in to_download)

    print(f"Remaining: {len(to_download)} files ({download_size/(1024**3):.2f} GB)")

    if len(to_download) == 0:
        print("\nAll files already downloaded!")
        return

    # Sort: get CSV/metadata files first, then videos by size (small first)
    to_download.sort(key=lambda b: (
        0 if b['name'].endswith('.csv') else 1,
        b['size']
    ))

    print(f"\nStarting download with {args.workers} worker(s)...")
    print(f"Press Ctrl+C to pause (progress is saved, just re-run to resume)\n")

    completed = 0
    failed = 0
    bytes_done = 0
    start_time = time.time()

    try:
        if args.workers <= 1:
            # Sequential download
            for i, blob in enumerate(to_download):
                size_mb = blob['size'] / (1024**2)
                elapsed = time.time() - start_time
                speed = bytes_done / elapsed if elapsed > 0 else 0
                eta = (download_size - bytes_done) / speed if speed > 0 else 0

                print(f"  [{completed+failed+1}/{len(to_download)}] "
                      f"{blob['name'][:50]:50s} ({size_mb:.1f} MB) "
                      f"[{speed/(1024**2):.1f} MB/s, ETA {eta/60:.0f}m]",
                      end='', flush=True)

                result = download_blob(
                    container_url, sas_token,
                    blob['name'], blob['size'],
                    output_dir, tracker
                )

                if result['status'] in ('complete', 'already_complete'):
                    tracker.mark_complete(blob['name'])
                    completed += 1
                    bytes_done += blob['size']
                    print(f" \u2713")
                else:
                    tracker.mark_failed(blob['name'], result.get('error', ''))
                    failed += 1
                    print(f" FAILED: {result.get('error', '')[:50]}")

        else:
            # Parallel download
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {}
                for blob in to_download:
                    future = executor.submit(
                        download_blob,
                        container_url, sas_token,
                        blob['name'], blob['size'],
                        output_dir, tracker
                    )
                    futures[future] = blob

                for future in as_completed(futures):
                    blob = futures[future]
                    try:
                        result = future.result()
                        if result['status'] in ('complete', 'already_complete'):
                            tracker.mark_complete(blob['name'])
                            completed += 1
                            bytes_done += blob['size']
                        else:
                            tracker.mark_failed(blob['name'], result.get('error', ''))
                            failed += 1
                    except Exception as e:
                        tracker.mark_failed(blob['name'], str(e))
                        failed += 1

                    total_done = completed + failed
                    if total_done % 100 == 0 or total_done == len(to_download):
                        elapsed = time.time() - start_time
                        speed = bytes_done / elapsed if elapsed > 0 else 0
                        pct = 100 * bytes_done / download_size if download_size > 0 else 0
                        print(f"  Progress: {total_done}/{len(to_download)} files, "
                              f"{bytes_done/(1024**3):.1f}/{download_size/(1024**3):.1f} GB "
                              f"({pct:.0f}%) [{speed/(1024**2):.1f} MB/s]")

    except KeyboardInterrupt:
        print(f"\n\nPaused by user. Progress saved to {progress_file}")
        print(f"Re-run this script to resume from where you left off.")
        tracker.save()
        return

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*65}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*65}")
    print(f"  Completed: {completed + previously_done + newly_skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(blobs)} files")
    print(f"  Time: {elapsed/60:.1f} minutes")
    if elapsed > 0:
        print(f"  Speed: {bytes_done/(1024**2)/elapsed:.1f} MB/s average")

    if failed > 0:
        print(f"\n  Failed files saved in {progress_file}")
        print(f"  Re-run to retry failed downloads.")


if __name__ == '__main__':
    main()

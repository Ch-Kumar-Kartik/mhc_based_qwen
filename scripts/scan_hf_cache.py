#!/usr/bin/env python3
"""Scan a Hugging Face cache directory for files that fail UTF-8 decoding.

This helps identify corrupted or partial downloads that cause
`UnicodeDecodeError` during `datasets.load_dataset(...)`.

Usage:
  python scripts/scan_hf_cache.py --cache-dir E:/mhc_based_qwen-2/hf_cache --dataset-name ai4bharat/IndicCorpV2

The script walks the dataset cache folder and reports files that fail
to decode as UTF-8.
"""
import argparse
import codecs
from pathlib import Path
import json
import sys


def check_file_utf8(path: Path) -> (bool, str):
    """Return (is_valid, error_message)."""
    try:
        decoder = codecs.getincrementaldecoder('utf-8')()
        with path.open('rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                decoder.decode(chunk)
        # final decode (may raise)
        decoder.decode(b'', final=True)
        return True, ''
    except UnicodeDecodeError as e:
        return False, f'UnicodeDecodeError at position {e.start}: {e.reason}'
    except Exception as e:
        return False, f'OtherError: {type(e).__name__}: {e}'


def scan_cache(cache_dir: Path, dataset_name: str = None):
    # If dataset_name provided, look for hub datasets path; else scan whole cache
    results = []
    if dataset_name:
        # hub datasets path format: <cache>/hub/datasets--owner--repo
        hub = cache_dir / 'hub'
        if not hub.exists():
            print(f'Hub folder not found at {hub}', file=sys.stderr)
        else:
            # find any folder starting with datasets--<dataset owner/repo>
            for p in hub.rglob('*'):
                # only check files under datasets--<name> directories
                if 'datasets--' in str(p) and p.is_file():
                    is_valid, msg = check_file_utf8(p)
                    if not is_valid:
                        results.append({'path': str(p), 'error': msg})
    else:
        for p in cache_dir.rglob('*'):
            if p.is_file():
                is_valid, msg = check_file_utf8(p)
                if not is_valid:
                    results.append({'path': str(p), 'error': msg})

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=str, required=True)
    parser.add_argument('--dataset-name', type=str, default=None, help='Optional dataset identifier (e.g. ai4bharat/IndicCorpV2)')
    parser.add_argument('--output', type=str, default=None, help='Optional JSON output file')
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        print(f'Cache dir not found: {cache_dir}', file=sys.stderr)
        raise SystemExit(2)

    print(f'Scanning cache directory: {cache_dir} ... this may take a while')
    bad = scan_cache(cache_dir, args.dataset_name)

    print(f'Found {len(bad)} problematic files')
    for b in bad:
        print(b['path'], '->', b['error'])

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(bad, f, indent=2, ensure_ascii=False)
        print('Wrote JSON report to', args.output)


if __name__ == '__main__':
    main()

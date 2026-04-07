#!/usr/bin/env python3
"""
Pre-resize MIMIC-CXR images to 256×256 JPEG.

Reduces dataset from ~571 GB → ~10–15 GB so the full training set fits in
the server's page cache after the first epoch, eliminating NFS I/O as a
bottleneck from epoch 2 onward.

Maintains the exact same directory structure:
  SRC:  <data_dir>/files/p10/p10000032/s50414267/<dicom_id>.jpg
  DST:  <out_dir>/files/p10/p10000032/s50414267/<dicom_id>.jpg

Usage (run ONCE before training):
  conda activate unimiss
  python scripts/preprocess_resize.py \
      --data_dir /scratch/pkrish52/MIMIC \
      --out_dir  /scratch/mmohan21/MIMIC_CXR_Classification/data/resized \
      --size 256 \
      --quality 90 \
      --workers 16
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def resize_one(args):
    src_path, dst_path, size, quality = args
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            return True, str(src_path)   # already done
        img = Image.open(src_path).convert('RGB')
        img = img.resize((size, size), Image.LANCZOS)
        img.save(dst_path, 'JPEG', quality=quality)
        return True, str(src_path)
    except Exception as e:
        return False, f"{src_path}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/scratch/pkrish52/MIMIC',
                        help='Root MIMIC-CXR directory (contains /files)')
    parser.add_argument('--out_dir',
                        default='/scratch/mmohan21/MIMIC_CXR_Classification/data/resized',
                        help='Output root (same structure as data_dir/files)')
    parser.add_argument('--size',    type=int, default=256,
                        help='Resize shorter edge to this (square output)')
    parser.add_argument('--quality', type=int, default=90,
                        help='JPEG quality (90 gives ~20-40 KB per image)')
    parser.add_argument('--workers', type=int, default=16,
                        help='Parallel workers')
    args = parser.parse_args()

    src_root = Path(args.data_dir) / 'files'
    dst_root = Path(args.out_dir)  / 'files'

    print(f"Source : {src_root}")
    print(f"Dest   : {dst_root}")
    print(f"Size   : {args.size}×{args.size}  Quality: {args.quality}")
    print(f"Workers: {args.workers}")

    # Collect all jpg files
    print("Scanning source directory …")
    all_jpgs = list(src_root.rglob('*.jpg'))
    print(f"Found {len(all_jpgs):,} images")

    if not all_jpgs:
        print("ERROR: No .jpg files found. Check --data_dir.")
        sys.exit(1)

    tasks = []
    for src in all_jpgs:
        rel  = src.relative_to(src_root)
        dst  = dst_root / rel
        tasks.append((src, dst, args.size, args.quality))

    errors = []
    done   = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        with tqdm(total=len(tasks), unit='img', dynamic_ncols=True) as pbar:
            for ok, msg in pool.map(resize_one, tasks, chunksize=64):
                if not ok:
                    errors.append(msg)
                done += 1
                pbar.update(1)
                if done % 10000 == 0:
                    pbar.set_postfix(errors=len(errors))

    print(f"\nDone. {done - len(errors):,} succeeded, {len(errors):,} failed.")
    if errors:
        print("First 10 errors:")
        for e in errors[:10]:
            print(" ", e)

    # Estimate output size
    sample = list(dst_root.rglob('*.jpg'))[:1000]
    if sample:
        avg_bytes = sum(s.stat().st_size for s in sample) / len(sample)
        total_gb  = avg_bytes * len(all_jpgs) / 1e9
        print(f"\nEstimated output size: {total_gb:.1f} GB  (avg {avg_bytes/1024:.1f} KB/img)")


if __name__ == '__main__':
    main()

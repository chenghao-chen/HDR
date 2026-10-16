#!/usr/bin/env python
"""
Build the uncompressed packed cache for SIDD-Medium raw (see
hdr_data.sidd.build_cache). One-time, ~41 GB, idempotent.

    python scripts/cache_sidd_raw.py --root $SIDD --out $SIDD/cache_packed_f32
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdr_data.sidd import build_cache  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
n = build_cache(a.root, a.out, a.workers)
print(f"done: {n} planes in {a.out}")

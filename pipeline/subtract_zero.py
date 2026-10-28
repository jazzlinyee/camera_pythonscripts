#!/usr/bin/env python3
"""
subtract_zero.py - show just the smears: subtract the all-black (reflection) capture
from each smear capture and save viewable images.

    # a pipeline scene folder (uses captures/zero.tiff and captures/scan_*.tiff)
    python subtract_zero.py sessions/101426/scenes/scene1

    # any captures + the black-mask capture
    python subtract_zero.py --zero ../../first_test/basler_005.tiff ../../first_test/basler_0[0-6]?.tiff

Output (a NEW folder; the originals are never changed):
    <scene>/smears/   or   <capture folder>_smears/   (or --out FOLDER)
      <name>_smears.png    8-bit picture of capture - zero, negatives shown as black.
                           All frames share ONE brightness scale (set from the first
                           frames), so they can be compared side by side.
      <name>_smears.tiff   only with --raw: the exact difference as 32-bit float
                           (can be negative; values in the same units as the captures).

Options:
    --scale 0.5        shrink the PNGs (0.5 = half size; full size is 5472 x 3648)
    --per-frame        stretch each PNG on its own instead of the shared scale
    --percentile 99.8  which brightness becomes white (higher = darker, less clipping)
    --gamma 0.6        brighten faint smears (1 = linear)
    --raw              also save the exact float32 difference

Works with raw 12-bit or x16 (MSB-aligned) TIFFs, as long as the zero frame and the
captures were saved the same way (it warns if they look different).
"""

import argparse
import glob
import sys
from pathlib import Path

import cv2
import numpy as np


def read(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        sys.exit(f"Could not read {path}")
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


def looks_msb(im):
    return im.max() > 4095


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="a pipeline scene folder, or capture files (globs OK)")
    ap.add_argument("--zero", help="the all-black (reflection) capture")
    ap.add_argument("--out", help="output folder (default: next to the inputs, ending in _smears)")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--per-frame", action="store_true")
    ap.add_argument("--percentile", type=float, default=99.8)
    ap.add_argument("--gamma", type=float, default=0.6)
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()

    # ---- figure out zero frame + captures
    first = Path(a.inputs[0])
    if len(a.inputs) == 1 and first.is_dir() and (first / "captures" / "zero.tiff").is_file():
        zero_path = first / "captures" / "zero.tiff"
        files = sorted((first / "captures").glob("scan_*.tiff"))
        out = Path(a.out) if a.out else first / "smears"
    else:
        if not a.zero:
            sys.exit("Give the black-mask capture with --zero (or pass a pipeline scene folder).")
        zero_path = Path(a.zero)
        files = []
        for pat in a.inputs:
            files += [Path(p) for p in sorted(glob.glob(pat))] if any(c in pat for c in "*?[") else [Path(pat)]
        files = [f for f in files if f.resolve() != zero_path.resolve()]
        out = Path(a.out) if a.out else files[0].parent.with_name(files[0].parent.name + "_smears")
    if not files:
        sys.exit("No captures found.")
    if out.resolve() in {f.parent.resolve() for f in files}:
        sys.exit("The output folder must be different from the captures' folder.")
    out.mkdir(parents=True, exist_ok=True)

    zero = read(zero_path).astype(np.float32)
    print(f"zero frame: {zero_path}  ({zero.shape[1]}x{zero.shape[0]}, mean {zero.mean():.0f})")
    print(f"{len(files)} capture(s) -> {out}")

    # ---- shared brightness scale from up to 5 frames
    white = None
    if not a.per_frame:
        vals = []
        for f in files[:5]:
            d = read(f).astype(np.float32) - zero
            vals.append(np.percentile(d[::4, ::4], a.percentile))
        white = max(float(np.median(vals)), 1.0)
        print(f"shared scale: difference {white:.0f} = white (the {a.percentile}th percentile)")

    zmsb = looks_msb(zero)
    for i, f in enumerate(files, 1):
        im = read(f)
        if looks_msb(im) != zmsb and im.max() > 0:
            print(f"  WARNING: {f.name} and the zero frame look like they were saved differently "
                  f"(x16 vs raw) - the subtraction will be off.")
        d = im.astype(np.float32) - zero
        w = white if white is not None else max(float(np.percentile(d[::4, ::4], a.percentile)), 1.0)
        v = np.clip(d / w, 0, 1) ** a.gamma
        png = (v * 255).astype(np.uint8)
        if a.scale != 1.0:
            png = cv2.resize(png, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"{f.stem}_smears.png"), png)
        if a.raw:
            cv2.imwrite(str(out / f"{f.stem}_smears.tiff"), d)
        neg = (d < 0).mean()
        print(f"  [{i}/{len(files)}] {f.name}: smear signal 99.9% = {np.percentile(d[::4, ::4], 99.9):.0f}, "
              f"negative (noise) {100 * neg:.0f}% of pixels")
    print("done")


if __name__ == "__main__":
    main()

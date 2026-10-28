#!/usr/bin/env python3
"""
slm_scan.py - full-coverage SLM scan: shift the coded-aperture mask between frames so
every SLM pixel is open in at least one frame, then merge the frames into one full cube.

Two steps:

1) Make the shifted masks (display these on the SLM, one per frame):

     python slm_scan.py masks mask_slm_w_3.png --out scan_masks

   Writes scan_masks/mask_00.png ... mask_NN.png and scan_masks/scan_plan.csv
   (frame number, row shift, column shift), and checks every SLM pixel gets covered.

2) After capturing one camera frame per mask, reconstruct and merge:

     python slm_scan.py merge \
         --plan     scan_masks/scan_plan.csv \
         --reverse  /home/ra648/calib/reverse_lookup_062926.npz \
         --frames   "measurements/100926/img_{i:02d}.tiff" \
         --dark     "measurements/100926/zero_{i:02d}.tiff" \
         --qe-csv   SonyIMX183.csv --qe multiply \
         --out      measurements/100926/cube_full.npz

   {i} in --frames / --dark is the frame number. --dark can also be one fixed file
   shared by all frames. Each frame is reconstructed exactly like meas2hyper.ipynb
   (dark subtract -> cv2.remap per band -> keep open pixels), and the results are merged.
   Pixels open in more than one frame (the scan overlaps by a few rows/columns) are averaged.

   Output .npz:  cube (SLM_H, SLM_W, Nt) float32, wavelengths, coverage (SLM_H, SLM_W)
   = number of frames that sampled each pixel (0 = never sampled -> NaN in cube).

How the shifts are chosen: the mask's open blocks repeat on a (period_y x period_x) grid
(found automatically). Shifting by the block size in each direction until one full
period is covered visits every pixel. For mask_slm_w_3 (3x3 blocks, 10 x 80 period with
a 5-row stagger every 40 columns) that's 4 row shifts x 14 column shifts = 56 frames.
np.roll wraps around the edges; since 1080x1920 is a whole number of periods, the
pattern stays exactly the same shape - no cut-off blocks.

Needs: numpy, opencv-python.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Mask analysis + shift plan
# --------------------------------------------------------------------------- #
def load_mask(path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        sys.exit(f"Could not read mask: {path}")
    return m > 0


def find_period(mask: np.ndarray, axis: int) -> int:
    """Smallest shift along `axis` that maps the mask onto itself (with wrap-around)."""
    n = mask.shape[axis]
    for p in range(1, n + 1):
        if n % p == 0 and np.array_equal(np.roll(mask, p, axis=axis), mask):
            return p
    return n


def block_size(mask: np.ndarray, axis: int) -> int:
    """Length of the first run of open pixels along `axis` (block height / width)."""
    line = mask.any(axis=1 - axis)          # rows (axis 0) or columns (axis 1) with any open pixel
    idx = np.flatnonzero(line)
    if idx.size == 0:
        sys.exit("Mask has no open pixels.")
    run = 1
    while idx[0] + run < line.size and line[idx[0] + run]:
        run += 1
    return run


def shift_steps(period: int, block: int) -> list:
    """Shifts 0, block, 2*block, ... until one period is covered."""
    return list(range(0, period, block))


def make_plan(mask: np.ndarray):
    py, px = find_period(mask, 0), find_period(mask, 1)
    by, bx = block_size(mask, 0), block_size(mask, 1)
    # The open blocks within one period may be staggered; covering a full (py x px)
    # period with steps of the block size always works. If the stagger means a smaller
    # horizontal period covers everything, use that to save frames.
    px_eff = px
    for cand in range(bx, px):
        if px % cand == 0:
            cover = np.zeros_like(mask, dtype=bool)
            for dy in shift_steps(py, by):
                for dx in shift_steps(cand, bx):
                    cover |= np.roll(mask, (dy, dx), axis=(0, 1))
            if cover.all():
                px_eff = cand
                break
    plan = [(dy, dx) for dy in shift_steps(py, by) for dx in shift_steps(px_eff, bx)]
    return plan, dict(period=(py, px), block=(by, bx), x_range=px_eff)


def cmd_masks(args):
    mask = load_mask(args.mask)
    plan, info = make_plan(mask)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cover = np.zeros(mask.shape, dtype=np.int32)
    with open(out / "scan_plan.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "dy", "dx", "mask_file"])
        for i, (dy, dx) in enumerate(plan):
            shifted = np.roll(mask, (dy, dx), axis=(0, 1))
            cover += shifted
            name = f"mask_{i:02d}.png"
            cv2.imwrite(str(out / name), shifted.astype(np.uint8) * 255)
            w.writerow([i, dy, dx, name])

    print(f"Mask {mask.shape[1]}x{mask.shape[0]}: {info['block'][0]}x{info['block'][1]} blocks, "
          f"pattern repeats every {info['period'][0]} rows x {info['period'][1]} cols, "
          f"{mask.mean():.2%} open per frame")
    print(f"Row shifts: {sorted({d for d, _ in plan})}")
    print(f"Col shifts: {sorted({d for _, d in plan})}")
    print(f"-> {len(plan)} frames. Coverage: every pixel open "
          f"{cover.min()}-{cover.max()} times "
          f"({(cover > 1).mean():.1%} of pixels sampled more than once).")
    if cover.min() == 0:
        print("WARNING: some pixels are never open - check the mask.")
    print(f"Saved masks + scan_plan.csv to {out}")


# --------------------------------------------------------------------------- #
# Reconstruction + merge
# --------------------------------------------------------------------------- #
def read_image(path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        sys.exit(f"Could not read image: {path}")
    return im.astype(np.float32)


def reconstruct(g, open_mask, rev_x, rev_y, cube_sum, count):
    """Same as meas2hyper.ipynb cell 7, but accumulates into sum/count arrays."""
    for i in range(rev_x.shape[0]):
        sampled = cv2.remap(g, rev_x[i], rev_y[i],
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        keep = open_mask & np.isfinite(sampled) & np.isfinite(rev_x[i]) & np.isfinite(rev_y[i])
        cube_sum[keep, i] += sampled[keep]
        count[keep, i] += 1


def cmd_merge(args):
    plan_path = Path(args.plan)
    with open(plan_path) as f:
        plan = list(csv.DictReader(f))
    print(f"{len(plan)} frames in {plan_path}")

    rev = np.load(args.reverse)
    rev_x, rev_y, wl = rev["reverse_cam_x"], rev["reverse_cam_y"], rev["wl_target"]
    nt, sh, sw = rev_x.shape
    print(f"Reverse lookup: {nt} bands, SLM {sw}x{sh}")

    cube_sum = np.zeros((sh, sw, nt), dtype=np.float32)
    count = np.zeros((sh, sw, nt), dtype=np.uint8)
    fixed_dark = read_image(args.dark) if args.dark and "{" not in args.dark else None

    for row in plan:
        i = int(row["frame"])
        mask = load_mask(plan_path.parent / row["mask_file"])
        if mask.shape != (sh, sw):
            sys.exit(f"Mask {row['mask_file']} is {mask.shape}, reverse lookup expects {(sh, sw)}")
        g = read_image(args.frames.format(i=i))
        if fixed_dark is not None:
            g -= fixed_dark
        elif args.dark:
            g -= read_image(args.dark.format(i=i))
        reconstruct(g, mask, rev_x, rev_y, cube_sum, count)
        print(f"  frame {i:02d} (dy={row['dy']}, dx={row['dx']}) done", flush=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        cube = np.where(count > 0, cube_sum / count, np.nan).astype(np.float32)

    if args.qe != "none":
        qe = np.genfromtxt(args.qe_csv, delimiter=",", skip_header=1, usecols=1)
        if qe.size != nt:
            sys.exit(f"QE file has {qe.size} values, cube has {nt} bands")
        cube = cube * qe[None, None, :] if args.qe == "multiply" else cube / qe[None, None, :]

    coverage = count.max(axis=2)
    np.savez_compressed(args.out, cube=cube, wavelengths=wl, coverage=coverage)
    print(f"Saved {args.out}: cube {cube.shape}, "
          f"{(coverage > 0).mean():.1%} of SLM pixels sampled "
          f"(the rest fall outside the camera / calibrated area)")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("masks", help="make the shifted masks + scan_plan.csv")
    m.add_argument("mask", help="base mask, e.g. mask_slm_w_3.png")
    m.add_argument("--out", default="scan_masks", help="output folder (default: scan_masks)")
    m.set_defaults(func=cmd_masks)

    g = sub.add_parser("merge", help="reconstruct each frame and merge into one cube")
    g.add_argument("--plan", required=True, help="scan_plan.csv from the masks step")
    g.add_argument("--reverse", required=True, help="reverse_lookup_<date>.npz")
    g.add_argument("--frames", required=True, help='frame path pattern, e.g. "img_{i:02d}.tiff"')
    g.add_argument("--dark", help='dark frame: one file, or a pattern with {i}')
    g.add_argument("--qe-csv", default="SonyIMX183.csv")
    g.add_argument("--qe", choices=["multiply", "divide", "none"], default="multiply",
                   help="QE correction. 'multiply' matches meas2hyper.ipynb as it is now; "
                        "'divide' is the usual way to undo sensor sensitivity.")
    g.add_argument("--out", default="cube_full.npz")
    g.set_defaults(func=cmd_merge)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

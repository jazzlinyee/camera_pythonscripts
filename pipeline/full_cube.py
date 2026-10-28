#!/usr/bin/env python3
"""
full_cube.py - the FULLY SAMPLED cube: a spectrum at every SLM pixel (1920 x 1080 x 31 bands).

The 56 scan masks together open every SLM pixel at least once, so the scan itself is
complete. The holes in cube_combined.npz come from the reverse-lookup table: near the
edges of the SLM it has no entry (NaN) because those pixels were dark or off the sensor
during calibration. This script closes them in three tiers and records which tier each
value came from:

  source = 2  measured, using the calibrated table             (same maths as meas2hyper)
  source = 1  measured, using the table EXTENDED past its edge  (smooth fit of the table)
  source = 0  not measurable (the light lands off the sensor): copied from the nearest
              measured pixel in the same band, so the cube has no gaps

Usage
  # a pipeline scene (uses its manifest, masks, captures/zero.tiff and the session table)
  python full_cube.py sessions/101426/scenes/scene1

  # any captures: zero frame + captures + masks, in the same order (sorted)
  python full_cube.py --zero ../../first_test/basler_005.tiff \
      --captures "../../first_test/basler_0[0-6][0-9].tiff" \
      --masks "masks/scan_masks/mask_*.png" \
      --table ../../calibration_test/reverse_lookup_062926.npz \
      --out ../../first_test_cubes/cube_full.npz

  # only merge + fill per-capture cubes you already have (no captures needed;
  # tier 1 is skipped, so the table-less edge is filled from neighbours)
  python full_cube.py --cubes ../../first_test_cubes --out ../../first_test_cubes/cube_full.npz

Options
  --no-extend          don't extend the table (tiers 2 and 0 only)
  --degree 4           polynomial degree of the table extension
  --qe multiply|divide|none   QE correction (default multiply = what meas2hyper does)
  --qe-csv FILE        QE curve (default: SonyIMX183.csv in the lab's calibration folder)

Output .npz: cube (H, W, Nt) float32 with no NaN, wavelengths, source (H, W, Nt) uint8,
coverage (H, W) = how many frames sampled each pixel. Also <out>_preview.png and
<out>_source.png (white = calibrated, grey = extended, black = filled).
Nothing is written anywhere except --out (and its two PNGs).
"""

import argparse
import csv
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------- table
def load_table(path):
    with np.load(path) as d:
        return (d["reverse_cam_x"].astype(np.float32), d["reverse_cam_y"].astype(np.float32),
                np.asarray(d["wl_target"]))


def _design(u, v, deg):
    return np.stack([u ** i * v ** j for i in range(deg + 1) for j in range(deg + 1 - i)], -1)


def extend_table(rx, ry, cam_shape, degree=4, log=print):
    """Fill the table's NaN entries with a smooth 2-D polynomial fit of each band
    (camera x and y as functions of SLM x, y). Entries that then land off the sensor
    stay NaN - that light really never reaches the camera.
    Returns (ext_x, ext_y, extended) where `extended` marks the newly filled entries."""
    nt, H, W = rx.shape
    ch, cw = cam_shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = (xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)
    ex, ey = rx.copy(), ry.copy()
    extended = np.zeros(rx.shape, bool)
    err = []
    for b in range(nt):
        ok = np.isfinite(rx[b]) & np.isfinite(ry[b])
        hole = ~ok
        if not hole.any() or ok.sum() < 1000:
            continue
        idx = np.flatnonzero(ok)[:: max(1, ok.sum() // 200000)]
        A = _design(u.ravel()[idx], v.ravel()[idx], degree)
        cx = np.linalg.lstsq(A, rx[b].ravel()[idx], rcond=None)[0]
        cy = np.linalg.lstsq(A, ry[b].ravel()[idx], rcond=None)[0]
        err.append(np.median(np.hypot(A @ cx - rx[b].ravel()[idx], A @ cy - ry[b].ravel()[idx])))
        Ah = _design(u[hole], v[hole], degree)
        px, py = Ah @ cx, Ah @ cy
        on = (px >= 0) & (px <= cw - 1) & (py >= 0) & (py <= ch - 1)
        hx, hy = np.nonzero(hole)
        ex[b, hx[on], hy[on]] = px[on]
        ey[b, hx[on], hy[on]] = py[on]
        extended[b, hx[on], hy[on]] = True
    log(f"Table: {np.isfinite(rx).mean():.1%} of entries calibrated; extension adds "
        f"{extended.mean():.1%} (fit error inside the calibrated area: median {np.median(err):.1f} camera px)")
    return ex, ey, extended


# ----------------------------------------------------------------- reconstruction
def read_image(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        sys.exit(f"Could not read {path}")
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im.astype(np.float32)


def accumulate(g, open_mask, tx, ty, total, count):
    """meas2hyper.ipynb's gather step, accumulated into sum/count (bilinear remap at the
    open SLM pixels; NaN where the table has no entry or the point is off the sensor)."""
    for b in range(tx.shape[0]):
        s = cv2.remap(g, tx[b], ty[b], interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        keep = open_mask & np.isfinite(s)
        total[keep, b] += s[keep]
        count[keep, b] += 1


def reconstruct(pairs, zero_path, table, extend=True, degree=4, log=print):
    rx, ry, wl = load_table(table)
    nt, H, W = rx.shape
    zero = read_image(zero_path)
    if extend:
        tx, ty, extended = extend_table(rx, ry, zero.shape, degree, log)
    else:
        tx, ty, extended = rx, ry, np.zeros(rx.shape, bool)
    del rx, ry
    total = np.zeros((H, W, nt), np.float32)
    count = np.zeros((H, W, nt), np.uint8)
    t0 = time.time()
    for n, (cap, mask) in enumerate(pairs, 1):
        m = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape != (H, W):
            sys.exit(f"Mask {mask} missing or not {W}x{H}")
        accumulate(read_image(cap) - zero, m > 0, tx, ty, total, count)
        if n % 8 == 0 or n == len(pairs):
            log(f"  {n}/{len(pairs)} frames ({time.time() - t0:.0f}s)")
    with np.errstate(invalid="ignore", divide="ignore"):
        cube = np.where(count > 0, total / count, np.nan).astype(np.float32)
    source = np.where(np.isfinite(cube), np.where(extended.transpose(1, 2, 0), 1, 2), 0).astype(np.uint8)
    return cube, wl, source, count.max(axis=2)


def merge_cubes(files, log=print):
    total = count = wl = None
    for f in files:
        with np.load(f) as d:
            c = d["cube"].astype(np.float32)
            if total is None:
                wl, total, count = d["wavelengths"], np.zeros(c.shape, np.float32), np.zeros(c.shape, np.uint8)
        ok = np.isfinite(c)
        total[ok] += c[ok]
        count[ok] += 1
    log(f"Merged {len(files)} cubes.")
    with np.errstate(invalid="ignore", divide="ignore"):
        cube = np.where(count > 0, total / count, np.nan).astype(np.float32)
    return cube, wl, np.where(np.isfinite(cube), 2, 0).astype(np.uint8), count.max(axis=2)


# ----------------------------------------------------------------- QE + fill
def apply_qe(cube, mode, qe_csv):
    if mode == "none":
        return cube
    qe = np.genfromtxt(qe_csv, delimiter=",", skip_header=1, usecols=1)
    if qe.size != cube.shape[2]:
        sys.exit(f"QE file has {qe.size} values, cube has {cube.shape[2]} bands")
    return cube * qe[None, None, :] if mode == "multiply" else cube / qe[None, None, :]


def fill_nearest(cube):
    """Give every NaN the value of the nearest measured pixel in the same band."""
    from scipy.ndimage import distance_transform_edt
    out = cube.copy()
    for b in range(cube.shape[2]):
        bad = ~np.isfinite(cube[..., b])
        if bad.all():
            out[..., b] = 0
        elif bad.any():
            iy, ix = distance_transform_edt(bad, return_distances=False, return_indices=True)
            out[..., b] = cube[..., b][iy, ix]
    return out


def to_rgb(cube, wl):
    w = np.stack([np.exp(-0.5 * ((wl - mu) / 35.0) ** 2) for mu in (610, 545, 455)], 1)
    rgb = np.nan_to_num(cube) @ (w / w.sum(0))
    hi = np.percentile(rgb[rgb > 0], 99.5) if (rgb > 0).any() else 1.0
    return (np.clip(rgb / hi, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def save(out, cube, wl, source, coverage, log=print):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, cube=cube, wavelengths=wl, source=source, coverage=coverage)
    cv2.imwrite(str(out.with_name(out.stem + "_preview.png")), to_rgb(cube, wl)[..., ::-1])
    s = np.round(source.mean(axis=2) * 127.5).astype(np.uint8)          # 255 / 127 / 0
    cv2.imwrite(str(out.with_name(out.stem + "_source.png")), s)
    n = source.size
    log(f"Saved {out}: {cube.shape}, NaN-free. Values: calibrated {np.sum(source == 2) / n:.1%}, "
        f"extended table {np.sum(source == 1) / n:.1%}, filled from neighbours {np.sum(source == 0) / n:.1%}")


def build(pairs=None, zero=None, table=None, cubes=None, out=None, extend=True, degree=4,
          qe="multiply", qe_csv=None, log=print):
    if cubes:
        cube, wl, source, cov = merge_cubes(cubes, log)        # per-frame cubes already have QE
    else:
        cube, wl, source, cov = reconstruct(pairs, zero, table, extend, degree, log)
        if not qe_csv:
            sys.path.insert(0, str(HERE))
            from settings import default_lab_path
            qe_csv = default_lab_path("calibration_code") / "SonyIMX183.csv"
        cube = apply_qe(cube, qe, qe_csv)
    cube = fill_nearest(cube)
    save(out, cube, wl, source, cov, log)
    return source


def scene_pairs(scene_dir):
    man = json.loads((scene_dir / "manifest.json").read_text())
    return [(scene_dir / f["capture"], scene_dir / f["mask"]) for f in man["frames"]]


# ----------------------------------------------------------------- CLI
def _glob(pat):
    return [Path(p) for p in sorted(glob.glob(pat))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", nargs="?", help="a pipeline scene folder")
    ap.add_argument("--zero"), ap.add_argument("--captures"), ap.add_argument("--masks")
    ap.add_argument("--table"), ap.add_argument("--cubes", help="folder of cube_*.npz to merge")
    ap.add_argument("--out")
    ap.add_argument("--no-extend", action="store_true")
    ap.add_argument("--degree", type=int, default=4)
    ap.add_argument("--qe", choices=["multiply", "divide", "none"], default="multiply")
    ap.add_argument("--qe-csv")
    a = ap.parse_args()

    if a.cubes:
        files = [f for f in _glob(str(Path(a.cubes) / "cube_*.npz")) if f.stem[5:].isdigit()]
        if not files:
            sys.exit(f"No cube_NN.npz files in {a.cubes}")
        build(cubes=files, out=a.out or Path(a.cubes) / "cube_full.npz")
        return
    if a.scene:
        sd = Path(a.scene)
        sys.path.insert(0, str(HERE))
        from settings import Session
        session = Session(sd.resolve().parent.parent.name, root=sd.resolve().parent.parent.parent)
        pairs, zero = scene_pairs(sd), sd / "captures" / "zero.tiff"
        table = Path(a.table) if a.table else session.table
        out = a.out or sd / "cubes" / "cube_full.npz"
    else:
        if not (a.zero and a.captures and a.masks and a.table and a.out):
            sys.exit("Give a scene folder, or --zero --captures --masks --table --out, or --cubes.")
        zero = Path(a.zero)
        caps = [c for c in _glob(a.captures) if c.resolve() != zero.resolve()]
        masks = _glob(a.masks)
        if len(caps) != len(masks):
            sys.exit(f"{len(caps)} captures but {len(masks)} masks - they must pair up in sorted order.")
        pairs, table, out = list(zip(caps, masks)), Path(a.table), a.out
        print("Pairs:", f"{caps[0].name}+{masks[0].name} ... {caps[-1].name}+{masks[-1].name}")
    build(pairs, zero, table, out=out, extend=not a.no_extend, degree=a.degree, qe=a.qe, qe_csv=a.qe_csv)


if __name__ == "__main__":
    main()

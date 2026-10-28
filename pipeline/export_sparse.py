#!/usr/bin/env python3
"""
export_sparse.py - sparsely sampled cube + RGB image on SLM coordinates, one file per
capture, in the format of the lab's dataprep.ipynb (June meas_QE.npz).

For each Basler capture taken with ONE mask on the SLM (e.g. test_masks/mask_00.png):
  1. background: subtracted using --zero if given, otherwise ESTIMATED (see below)
  2. sparse cube: meas2hyper's calculation (same code as meas2hyper.ipynb cells 6-7,
     verified identical), x QE, then averaged over each 3x3 block like dataprep.ipynb
  3. RGB: the Blackfly frame taken at the same moment, debayered + flipped vertically and
     warped into SLM coordinates with guide_to_slm.npz (from guide_calib.ipynb), like dataprep

    python export_sparse.py --session 101426 \
        --guide sessions/101426/calibration/guide/guide_to_slm.npz \
        --mask masks/test_masks/mask_00.png \
        --out export_102126/redbox  labgui_state/captures_102126

Output per capture: <out>/<basler name>.npz with
    hypercube        (1080,1920,31) float32  3x3-averaged sparse cube, normalised to max 1
                                             (exactly like dataprep.ipynb), 0 = not sampled
    hypercube_raw    same, not normalised (raw 12-bit units x QE)
    hypercube_nosub  same as hypercube_raw but with NO background subtraction
    mask             (1080,1920) float32 1 = open SLM pixel
    rgb              (1080,1920,3) float32 0-1, Blackfly image on SLM coordinates
                     (channel order as dataprep.ipynb produces it)
    rgb_valid        (1080,1920) bool, where the Blackfly sees the SLM
    wavelengths      (31,) nm
    background       'measured' or 'estimated', plus the source file names
and <basler name>_preview.png (left: cube rendered as colour, right: RGB on SLM).

Background estimate (no black frame): background = alpha x (Gaussian blur, sigma 8 px, of the
capture itself), with alpha = 0.962 measured on manual_scene (which has a real black frame).
Tested there: per-block spectral SHAPE correlation 0.94 with the true-black result, but
absolute values off by roughly half. Use --zero whenever a real black frame exists.

Nothing is written except into --out.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import full_cube as fc            # noqa: E402  (load_table, accumulate, apply_qe)
from settings import Session, default_lab_path      # noqa: E402

ALPHA, SIGMA = 0.962, 8.0


def read_raw12(p):
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 2:
        sys.exit(f"Could not read {p}")
    im = im.astype(np.float32)
    return im / 16 if im.max() > 4095 else im          # MSB-aligned (x16) -> raw 12-bit


def pair_blackfly(baslers, folder):
    """Match each Basler capture with its Blackfly frame: in sorted order when the folder has
    the same number of each (the camera GUI saves one of each per capture), otherwise the
    Blackfly frame saved closest in time (within 3 s)."""
    bfs = sorted(folder.glob("blackfly_*.png"))
    if len(bfs) == len(baslers):
        pairs = dict(zip(baslers, bfs))
    else:
        pairs = {}
        for b in baslers:
            best = min(bfs, key=lambda c: abs(c.stat().st_mtime - b.stat().st_mtime), default=None)
            ok = best is not None and abs(best.stat().st_mtime - b.stat().st_mtime) <= 3
            pairs[b] = best if ok else None
    for b, f in pairs.items():
        if f is not None and abs(f.stat().st_mtime - b.stat().st_mtime) > 3:
            print(f"  note: {b.name} and {f.name} were saved "
                  f"{abs(f.stat().st_mtime - b.stat().st_mtime):.0f} s apart - check they show the same moment")
    return pairs


def block_average(cube, labels, n):
    """dataprep.ipynb cell 3: every pixel of a 3x3 block gets the block's mean (per band)."""
    out = np.full_like(cube, np.nan)
    flat = labels.ravel()
    for b in range(cube.shape[2]):
        v = cube[..., b].ravel()
        k = (flat > 0) & np.isfinite(v)
        s = np.bincount(flat[k], weights=v[k], minlength=n + 1)
        c = np.bincount(flat[k], minlength=n + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s / c
        m[0] = np.nan
        out[..., b] = m[labels]
    return out


def sparse_cube(g, mask, rx, ry, qe_csv):
    tot = np.zeros(mask.shape + (rx.shape[0],), np.float32)
    cnt = np.zeros(tot.shape, np.uint8)
    fc.accumulate(g, mask, rx, ry, tot, cnt)
    with np.errstate(invalid="ignore", divide="ignore"):
        cube = np.where(cnt > 0, tot / cnt, np.nan).astype(np.float32)
    return fc.apply_qe(cube, "multiply", qe_csv)        # meas2hyper.ipynb multiplies by QE


def rgb_on_slm(bf_path, gx, gy):
    raw = cv2.imread(str(bf_path), cv2.IMREAD_GRAYSCALE)
    rgb = cv2.flip(cv2.cvtColor(raw, cv2.COLOR_BayerRG2BGR), 0)      # as dataprep.ipynb
    out = np.stack([cv2.remap(rgb[..., c].astype(np.float32), gx, gy, cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0) for c in range(3)], -1)
    return (out / 255.0).astype(np.float32)


def preview(cube, wl, rgb, path):
    c = np.nan_to_num(fc.to_rgb(np.nan_to_num(cube), wl))[..., ::-1]      # to_rgb gives RGB
    r = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)                        # dataprep order shown as-is
    cv2.imwrite(str(path), np.hstack([c, r[..., ::-1]])[::2, ::2])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="folders with basler_*.tiff + blackfly_*.png")
    ap.add_argument("--session", required=True)
    ap.add_argument("--guide", required=True, help="guide_to_slm.npz from guide_calib.ipynb")
    ap.add_argument("--mask", required=True, help="the mask that was on the SLM for every capture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--zero", help="a real black-mask frame (then no estimate is used)")
    ap.add_argument("--qe-csv", default=str(default_lab_path("calibration_code") / "SonyIMX183.csv"))
    a = ap.parse_args()

    session = Session(a.session)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in a.folders:
        if out.resolve() == Path(f).resolve():
            sys.exit("--out must be a different folder from the captures.")

    print(f"Table: {session.table}")
    rx, ry, wl = fc.load_table(session.table)
    with np.load(a.guide) as d:
        gx, gy = d["guide_x"].astype(np.float32), d["guide_y"].astype(np.float32)
    rgb_valid = np.isfinite(gx) & np.isfinite(gy) & (gx >= 0) & (gy >= 0) & (gx <= 1439) & (gy <= 1079)
    gx, gy = np.where(np.isfinite(gx), gx, -1).astype(np.float32), np.where(np.isfinite(gy), gy, -1).astype(np.float32)
    m8 = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE)
    if m8 is None or m8.shape != rx.shape[1:]:
        sys.exit(f"Mask {a.mask} missing or not {rx.shape[2]}x{rx.shape[1]}")
    mask = m8 > 0
    from scipy.ndimage import label
    labels, n = label(mask)
    zero = read_raw12(a.zero) if a.zero else None
    print(f"RGB covers {rgb_valid.mean():.1%} of the SLM; mask has {n} open blocks; "
          f"background: {'measured from ' + a.zero if zero is not None else 'ESTIMATED (no black frame)'}")

    for folder in map(Path, a.folders):
        files = sorted(folder.glob("basler_*.tiff"))
        print(f"\n{folder}: {len(files)} Basler captures")
        bf_of = pair_blackfly(files, folder)
        for i, f in enumerate(files, 1):
            g = read_raw12(f)
            bg = zero if zero is not None else ALPHA * cv2.GaussianBlur(g, (0, 0), SIGMA)
            raw = block_average(sparse_cube(g - bg, mask, rx, ry, a.qe_csv), labels, n)
            nosub = block_average(sparse_cube(g, mask, rx, ry, a.qe_csv), labels, n)
            hc = np.nan_to_num(raw)
            hc = np.clip(hc / hc.max(), 0, 1) if hc.max() > 0 else hc          # dataprep cell 5
            bf = bf_of[f]
            rgb = rgb_on_slm(bf, gx, gy) if bf else np.zeros(mask.shape + (3,), np.float32)
            info = {"basler": f.name, "blackfly": bf.name if bf else None, "folder": str(folder.resolve()),
                    "mask": str(Path(a.mask).resolve()), "table": str(session.table),
                    "guide": str(Path(a.guide).resolve()), "made": datetime.now().isoformat(timespec="seconds"),
                    "background": "measured" if zero is not None else "estimated",
                    "background_method": (f"black frame {a.zero}" if zero is not None else
                                          f"alpha {ALPHA} x GaussianBlur(capture, sigma {SIGMA})"),
                    "qe": "multiplied (as meas2hyper.ipynb)", "units_raw": "raw 12-bit x QE"}
            np.savez_compressed(out / f"{folder.name}_{f.stem}.npz", hypercube=hc.astype(np.float32),
                                hypercube_raw=np.nan_to_num(raw).astype(np.float32),
                                hypercube_nosub=np.nan_to_num(nosub).astype(np.float32),
                                mask=mask.astype(np.float32), rgb=rgb, rgb_valid=rgb_valid,
                                wavelengths=wl, info=json.dumps(info))
            preview(raw, wl, rgb, out / f"{folder.name}_{f.stem}_preview.png")
            print(f"  [{i}/{len(files)}] {f.name} + {bf.name if bf else 'NO Blackfly frame'} -> "
                  f"{folder.name}_{f.stem}.npz")
    print("\ndone")


if __name__ == "__main__":
    main()

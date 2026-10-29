#!/usr/bin/env python3
"""
export_sparse.py - sparsely sampled cube + RGB image on SLM coordinates, one file per
capture, in the format of the lab's dataprep.ipynb (June meas_QE.npz).

For each Basler capture taken with ONE mask on the SLM (e.g. test_masks/mask_00.png):
  1. background: NOT subtracted by default (--background none, 2026-10-29: the PhD student
     comments out the zero subtraction in meas2hyper). --zero FILE subtracts a real
     black-mask frame; --background estimate uses the estimate described below.
     --positions: the sources are scenes made with the GUI's 'Capture position' (each
     position has mask.tiff + its own zero.tiff + rgb.png): each position's black frame is
     subtracted. Output <scene>_pos_NN.npz.
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
    hypercube_nosub  (only when a background was subtracted) the same without subtraction
    mask             (1080,1920) float32 1 = open SLM pixel
    rgb              (1080,1920,3) float32 0-1, Blackfly image on SLM coordinates
                     (channel order as dataprep.ipynb produces it)
    rgb_valid        (1080,1920) bool, where the Blackfly sees the SLM
    wavelengths      (31,) nm
    info             JSON: background ('none' / 'measured' / 'estimated') and the source files
and <basler name>_preview.png: top = whole SLM, bottom = zoom on the area the Blackfly sees;
left = sparse cube rendered as colour (gaps filled from the nearest block, for viewing only),
right = Blackfly RGB on SLM coordinates.

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


def read_raw12(p, scale=None):
    """Basler TIFF -> raw 12-bit float. scale: the recorded pixel scale (pipeline captures
    record it); None = guess (x16 if anything is above 4095, as camera-GUI captures are)."""
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 2:
        sys.exit(f"Could not read {p}")
    im = im.astype(np.float32)
    if scale:
        return im / float(scale)
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


_FILL = {}


def preview(cube, wl, rgb, rgb_valid, mask, path):
    """Viewing aid only: the sparse cube rendered as colour with each gap filled from the
    nearest sampled block, next to the Blackfly RGB; whole SLM on top, Blackfly area below."""
    key = mask.tobytes()[:64] + bytes(str(mask.sum()), "ascii")
    if key not in _FILL:
        from scipy.ndimage import distance_transform_edt
        _FILL.clear()
        _FILL[key] = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    iy, ix = _FILL[key]
    c = fc.to_rgb(np.nan_to_num(cube), wl)[iy, ix]                       # RGB
    r = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)                       # dataprep order, shown as RGB
    top = np.hstack([c, r])[::2, ::2]
    ys, xs = np.nonzero(rgb_valid)
    if len(ys):
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        zc, zr = c[y0:y1, x0:x1], r[y0:y1, x0:x1]
        bot = np.hstack([zc, zr])
        scale = top.shape[1] / bot.shape[1]
        bot = cv2.resize(bot, (top.shape[1], max(1, int(bot.shape[0] * scale))), interpolation=cv2.INTER_NEAREST)
        top = np.vstack([top, np.full((6, top.shape[1], 3), 255, np.uint8), bot])
    cv2.imwrite(str(path), top[..., ::-1])


def _jobs_from_folders(folders):
    """Hand-captured folders: every basler_*.tiff with its paired blackfly_*.png."""
    jobs = []
    for folder in map(Path, folders):
        files = sorted(folder.glob("basler_*.tiff"))
        print(f"{folder}: {len(files)} Basler captures")
        bf_of = pair_blackfly(files, folder)
        jobs += [dict(name=f"{folder.name}_{f.stem}", basler=f, blackfly=bf_of[f], zero=None, mask=None,
                      folder=folder) for f in files]
    return jobs


def _jobs_from_positions(scenes):
    """Pipeline 'sparse capture' scenes: positions/pos_NN/{mask.tiff, zero.tiff, rgb.png},
    each with its own black frame."""
    jobs = []
    for sd in map(Path, scenes):
        pos = sorted(p for p in (sd / "positions").glob("pos_*") if (p / "mask.tiff").is_file())
        print(f"{sd}: {len(pos)} positions")
        for p in pos:
            rec = json.loads((p / "position.json").read_text()) if (p / "position.json").is_file() else {}
            jobs.append(dict(name=f"{sd.name}_{p.name}", basler=p / "mask.tiff",
                             blackfly=(p / "rgb.png") if (p / "rgb.png").is_file() else None,
                             zero=(p / "zero.tiff") if (p / "zero.tiff").is_file() else None,
                             mask=rec.get("mask_file"), folder=p, scale=rec.get("pixel_scale", 1)))
    return jobs


def run(sources, session_name, guide, out, mask_path=None, zero_path=None, background="none",
        positions=False, qe_csv=None, log=print):
    session = Session(session_name)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    qe_csv = qe_csv or str(default_lab_path("calibration_code") / "SonyIMX183.csv")
    jobs = _jobs_from_positions(sources) if positions else _jobs_from_folders(sources)
    if not jobs:
        sys.exit("Nothing to export.")
    log(f"Table: {session.table}")
    rx, ry, wl = fc.load_table(session.table)
    with np.load(guide) as d:
        gx, gy = d["guide_x"].astype(np.float32), d["guide_y"].astype(np.float32)
    rgb_valid = np.isfinite(gx) & np.isfinite(gy) & (gx >= 0) & (gy >= 0) & (gx <= 1439) & (gy <= 1079)
    gx, gy = np.where(np.isfinite(gx), gx, -1).astype(np.float32), np.where(np.isfinite(gy), gy, -1).astype(np.float32)
    from scipy.ndimage import label
    masks = {}

    def get_mask(path):
        if path not in masks:
            m8 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if m8 is None or m8.shape != rx.shape[1:]:
                sys.exit(f"Mask {path} missing or not {rx.shape[2]}x{rx.shape[1]}")
            m = m8 > 0
            masks[path] = (m,) + label(m)
        return masks[path]

    global_zero = read_raw12(zero_path) if zero_path else None
    log(f"RGB covers {rgb_valid.mean():.1%} of the SLM.")
    for i, j in enumerate(jobs, 1):
        mpath = mask_path or j["mask"]
        if not mpath:
            sys.exit(f"No mask given for {j['basler']} (use --mask).")
        mask, labels, n = get_mask(str(mpath))
        g = read_raw12(j["basler"], j.get("scale"))
        if j["zero"] is not None:
            zero, mode, how = read_raw12(j["zero"], j.get("scale")), "measured", f"black frame {j['zero']}"
        elif global_zero is not None:
            zero, mode, how = global_zero, "measured", f"black frame {zero_path}"
        elif background == "estimate":
            zero, mode, how = ALPHA * cv2.GaussianBlur(g, (0, 0), SIGMA), "estimated", \
                f"alpha {ALPHA} x GaussianBlur(capture, sigma {SIGMA})"
        else:
            zero, mode, how = None, "none", "not subtracted"
        nosub = block_average(sparse_cube(g, mask, rx, ry, qe_csv), labels, n)
        raw = nosub if zero is None else block_average(sparse_cube(g - zero, mask, rx, ry, qe_csv), labels, n)
        hc = np.nan_to_num(raw)
        hc = np.clip(hc / hc.max(), 0, 1) if hc.max() > 0 else hc          # dataprep cell 5
        bf = j["blackfly"]
        rgb = rgb_on_slm(bf, gx, gy) if bf else np.zeros(mask.shape + (3,), np.float32)
        info = {"basler": str(j["basler"]), "blackfly": str(bf) if bf else None, "folder": str(Path(j["folder"]).resolve()),
                "mask": str(Path(mpath).resolve()), "table": str(session.table),
                "guide": str(Path(guide).resolve()), "made": datetime.now().isoformat(timespec="seconds"),
                "background": mode, "background_method": how,
                "qe": "multiplied (as meas2hyper.ipynb)", "units_raw": "raw 12-bit x QE"}
        extra = {} if zero is None else {"hypercube_nosub": np.nan_to_num(nosub).astype(np.float32)}
        np.savez_compressed(out / f"{j['name']}.npz", hypercube=hc.astype(np.float32),
                            hypercube_raw=np.nan_to_num(raw).astype(np.float32),
                            mask=mask.astype(np.float32), rgb=rgb, rgb_valid=rgb_valid,
                            wavelengths=wl, info=json.dumps(info), **extra)
        preview(raw, wl, rgb, rgb_valid, mask, out / f"{j['name']}_preview.png")
        log(f"  [{i}/{len(jobs)}] {Path(j['basler']).name} + {Path(bf).name if bf else 'NO Blackfly frame'}"
            f" ({'black frame subtracted' if mode == 'measured' else 'background ' + mode}) -> {j['name']}.npz")
    log(f"done -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", help="folders with basler_*.tiff + blackfly_*.png "
                                               "(or, with --positions, pipeline sparse-capture scene folders)")
    ap.add_argument("--session", required=True)
    ap.add_argument("--guide", required=True, help="guide_to_slm.npz from guide_calib.ipynb")
    ap.add_argument("--mask", help="the mask that was on the SLM (positions record their own)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", action="store_true",
                    help="sources are scenes from 'Capture position': each position's own black frame is subtracted")
    ap.add_argument("--zero", help="subtract this real black-mask frame")
    ap.add_argument("--background", choices=["none", "estimate"], default="none",
                    help="without a black frame: none (default) or the in-frame estimate")
    ap.add_argument("--qe-csv")
    a = ap.parse_args()
    for f in a.sources:
        if Path(a.out).resolve() == Path(f).resolve():
            sys.exit("--out must be a different folder from the captures.")
    run(a.sources, a.session, a.guide, a.out, mask_path=a.mask, zero_path=a.zero, background=a.background,
        positions=a.positions, qe_csv=a.qe_csv)


if __name__ == "__main__":
    main()

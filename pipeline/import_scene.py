#!/usr/bin/env python3
"""
import_scene.py - turn captures taken by hand (e.g. Auto Cycle in the camera GUI over
slm_scan/scan_masks) into a pipeline scene, so '3 Cube -> Make cube' can use it.

    python import_scene.py 101426 manual_scene_s sessions/101426/scenes/manual_scene

    session      the pipeline session (its table is used for the cube)
    scene        name of the NEW scene to create
    folder       where the hand captures are: basler_NNN.tiff (and blackfly_NNN.png)

The Basler files are taken in sorted order: the FIRST is the black mask (zero frame), the
next 56 are mask_00 ... mask_55 (the order Auto Cycle uses, black.png sorting first).
Use --first to start somewhere else, e.g. --first basler_004.tiff.
If it is interrupted, run the same command again: finished frames are kept.

Captures saved MSB-aligned (x16, values up to 65520) are converted back to raw 12-bit, which
is what the pipeline's own captures contain. The original folder is only read.

Options:
    --first FILE         the black-mask capture (default: the first basler_*.tiff)
    --exposure-us N      exposure used, recorded in the manifest (optional)
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from settings import Session  # noqa: E402
import stages  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session"), ap.add_argument("scene"), ap.add_argument("folder")
    ap.add_argument("--first")
    ap.add_argument("--exposure-us", type=float)
    a = ap.parse_args()

    session = Session(a.session)
    src = Path(a.folder)
    sd = session.scene_dir(a.scene)
    if sd.resolve() == src.resolve():
        sys.exit("Give the new scene a different name from the folder you are importing.")
    if (sd / "manifest.json").exists():
        sys.exit(f"Scene '{a.scene}' already exists - pick a new name.")
    plan = stages.scan_plan(session)

    basler = sorted(src.glob("basler_*.tiff")) or sorted(src.glob("basler_*.tif"))
    if a.first:
        names = [p.name for p in basler]
        if a.first not in names:
            sys.exit(f"{a.first} not found in {src}")
        basler = basler[names.index(a.first):]
    if len(basler) < len(plan) + 1:
        sys.exit(f"Need 1 black + {len(plan)} mask captures, found {len(basler)} Basler files.")
    zero_f, scans = basler[0], basler[1:len(plan) + 1]
    if len(basler) > len(plan) + 1:
        print(f"Note: {len(basler) - len(plan) - 1} extra Basler file(s) after {scans[-1].name} are ignored.")
    print(f"zero = {zero_f.name};  mask_00 = {scans[0].name} ... mask_{len(plan) - 1:02d} = {scans[-1].name}")

    def read12(p):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None or im.ndim != 2:
            sys.exit(f"Could not read {p} as a grayscale image")
        return im

    zero = read12(zero_f)
    msb = zero.max() > 4095 or read12(scans[0]).max() > 4095
    conv = (lambda im: (im >> 4).astype(np.uint16)) if msb else (lambda im: im.astype(np.uint16))
    print("Files are MSB-aligned (x16): converting to raw 12-bit." if msb else "Files are raw 12-bit.")

    cap = sd / "captures"
    (cap / "rgb").mkdir(parents=True, exist_ok=True)
    (sd / "masks").mkdir(parents=True, exist_ok=True)
    z = conv(zero)
    cv2.imwrite(str(cap / "zero_start.tiff"), z)
    cv2.imwrite(str(cap / "zero.tiff"), z)
    if z.mean() >= conv(read12(scans[0])).mean():
        print("WARNING: the zero frame is not darker than the first mask frame - check that "
              f"{zero_f.name} really is the black mask (use --first).")

    saturated, frames = [], []
    bf = 0
    for r, f in zip(plan, scans):
        k = r["frame"]
        out = cap / f"scan_{k:02d}.tiff"
        im = None
        if out.is_file() and out.stat().st_mtime >= f.stat().st_mtime:      # resume an interrupted import
            im = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)                   # None if it was cut off
        if im is None:
            im = conv(read12(f))
            cv2.imwrite(str(out), im)
        if (im >= 4095).mean() > 0.001:
            saturated.append(f"scan_{k:02d}.tiff")
        shutil.copy2(r["mask"], sd / "masks" / Path(r["mask"]).name)
        rgb = src / f.name.replace("basler_", "blackfly_").replace(".tiff", ".png").replace(".tif", ".png")
        if rgb.is_file():
            shutil.copy2(rgb, cap / "rgb" / f"rgb_{k:02d}.png")
            bf += 1
        frames.append({"frame": k, "dy": r["dy"], "dx": r["dx"], "mask": f"masks/{Path(r['mask']).name}",
                       "capture": f"captures/scan_{k:02d}.tiff", "source_file": f.name})
        print(f"\r  {k + 1}/{len(plan)}", end="", flush=True)
    print()

    manifest = {
        "scene": a.scene, "session": session.name,
        "captured": datetime.fromtimestamp(zero_f.stat().st_mtime).isoformat(timespec="seconds"),
        "imported_from": str(src.resolve()), "exposure_us": a.exposure_us, "settle_ms": None,
        "pixel_values": "raw 12-bit (0-4095) in uint16 TIFF" + (" (converted from x16)" if msb else ""),
        "table": str(session.table), "table_md5": stages._md5(session.table) if session.table.is_file() else None,
        "zero_frames": ["zero_start.tiff"], "zero_source_file": zero_f.name, "zero_drift": None,
        "blackfly": bf == len(plan), "frames": frames, "saturated_frames": saturated,
    }
    (sd / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Scene '{a.scene}' created in {sd}")
    if saturated:
        print(f"WARNING: {len(saturated)} frames have saturated pixels (>0.1%); the cube is wrong at those spots.")
    print("Next: 3 Cube tab -> select it -> Make cube   (or: python pipeline.py cube "
          f"{session.name} {a.scene})")


if __name__ == "__main__":
    main()

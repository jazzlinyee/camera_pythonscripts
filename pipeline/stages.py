"""
The three pipeline stages.

  calibrate  capture  -> Gray-code patterns under each laser (+ check frames)
             process  -> lab notebooks: graycode -> interpolation -> Reverse_lookup.py -> table
             check    -> does the table match the hardware's orientation?
  capture             -> all-zero + 56 scan masks of a scene (+ RGB guide frames)
  cube                -> lab meas2hyper.ipynb on every capture, combined into one cube

Hardware stages accept `ui`, an object with:
    ui.main(fn)        run fn on the GUI/main thread and return its result
    ui.prompt(msg)     show msg and wait for OK (False = cancel)
    ui.progress(i, n, text)
    ui.stopped()       True if the user pressed Stop
The CLI and the GUI each provide one.
"""

import csv
import hashlib
import re
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from checks import orientation_check
from hardware import raw12_to_png8, saturation_fraction
from labcode import CodeRule, LabCodeError, Rule, run_notebook, run_script

CHECK_MASKS = (0, 7, 14)
N_PATTERNS = 44


class StageError(RuntimeError):
    pass


class ConsoleUI:
    def __init__(self, log=print):
        self.log = log

    def main(self, fn):
        return fn()

    def prompt(self, msg):
        try:
            return input(f"\n>>> {msg}\n    Press Enter to continue (or type q to cancel): ").strip().lower() != "q"
        except EOFError:
            return True

    def progress(self, i, n, text=""):
        self.log(f"[{i}/{n}] {text}")

    def stopped(self):
        return False


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _load_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise StageError(f"Could not read mask {path}")
    if m.shape != (1080, 1920):
        raise StageError(f"{path} is {m.shape[1]}x{m.shape[0]}, the SLM needs 1920x1080")
    return m


def scan_plan(session):
    d = session.lab_path("scan_masks")
    p = d / "scan_plan.csv"
    if not p.is_file():
        raise StageError(f"No scan_plan.csv in {d} - run slm_scan.py masks first.")
    with open(p) as f:
        rows = [dict(frame=int(r["frame"]), dy=int(r["dy"]), dx=int(r["dx"]), mask=d / r["mask_file"])
                for r in csv.DictReader(f)]
    for k in CHECK_MASKS:
        if k >= len(rows) or rows[k]["frame"] != k:
            raise StageError("scan_plan.csv does not have the expected frames 0, 7 and 14.")
    return rows


# Basler TIFFs written by the pipeline (2026-10-30): raw 12-bit x 16 ("MSB-aligned", 0-65520),
# like the camera GUI saves, so they look normal in image viewers. Every reader divides by the
# scale recorded with the data (manifest.json / position.json / check/scale.json), so results
# stay in raw 12-bit units. Data saved before has no record -> scale 1.
PIXEL_SCALE = 16


def read_basler(path, scale=1):
    """A pipeline Basler TIFF as float32 raw 12-bit values."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise StageError(f"Could not read {path}")
    return im.astype(np.float32) / float(scale or 1)


def scene_scale(sd):
    p = Path(sd) / "manifest.json"
    return json.loads(p.read_text()).get("pixel_scale", 1) if p.is_file() else 1


def check_scale(chk):
    p = Path(chk) / "scale.json"
    return json.loads(p.read_text()).get("pixel_scale", 1) if p.is_file() else 1


def bright_fraction(raw):
    """How full the sensor is in the bright parts (99.9th percentile / 4095)."""
    return float(np.percentile(np.asarray(raw)[::4, ::4], 99.9)) / 4095


def _black():
    return np.zeros((1080, 1920), np.uint8)


# =============================================================== shot runner
def run_shots(rig, shots, ui, log):
    """shots: list of dicts {mask_index, save: {basler: path, blackfly: path|None}, fmt, group, exposure, light}"""
    if hasattr(rig, "begin"):
        ui.main(rig.begin)
    try:
        return _run_shots(rig, shots, ui, log)
    finally:
        if hasattr(rig, "end"):
            ui.main(rig.end)


def _fmt_dur(s):
    s = int(round(s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def _run_shots(rig, shots, ui, log):
    n = len(shots)
    group = None
    g_t0, g_i, g_n = time.time(), 0, 0
    for i, shot in enumerate(shots, 1):
        if ui.stopped():
            raise StageError("Stopped by user.")
        if shot.get("group") != group:
            group = shot.get("group")
            light = shot.get("light")
            if light is not None:
                if hasattr(rig, "set_light"):
                    rig.set_light(light)          # simulator
                else:
                    log(f"  >> {shot['prompt']}  (waiting for OK)")
                    if not ui.prompt(shot["prompt"]):
                        raise StageError("Cancelled.")
                    log("  >> OK - capturing")
            g_t0, g_i, g_n = time.time(), 0, sum(1 for s in shots if s.get("group") == group)
            if shot.get("exposure"):
                rig.set_exposure(shot["exposure"])
                log(f"  exposure {shot['exposure']:.0f} us")
        ui.main(lambda: rig.show(shot["mask_index"]))
        raw = rig.grab_basler()
        out = Path(shot["save"]["basler"])
        out.parent.mkdir(parents=True, exist_ok=True)
        if shot["fmt"] == "png8":
            cv2.imwrite(str(out), raw12_to_png8(raw))
        else:
            cv2.imwrite(str(out), (np.clip(raw, 0, 4095).astype(np.uint16) << 4))   # x16, see PIXEL_SCALE
        sat = saturation_fraction(raw)
        shot["saturated"] = sat
        if shot["save"].get("blackfly"):
            rgb = rig.grab_blackfly()
            if rgb is not None:
                Path(shot["save"]["blackfly"]).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(shot["save"]["blackfly"]), rgb)
                # the same raw Bayer bytes as a headerless .raw (the lab's img_rgb.raw format)
                np.ascontiguousarray(rgb).tofile(str(Path(shot["save"]["blackfly"]).with_suffix(".raw")))
        ui.progress(i, n, f"{out.name}" + (f"  ({sat:.1%} saturated!)" if sat > 0.001 else ""))
        shot["raw_stats"] = [int(raw.min()), float(raw.mean()), int(raw.max())]
        shot["bright"] = bright_fraction(raw)
        g_i += 1
        if shot.get("label"):                     # one line per frame (calibration)
            done = time.time() - g_t0
            left = done / g_i * (g_n - g_i)
            bright = float(np.percentile(raw[::4, ::4], 99.9)) / 4095
            log(f"  {shot['label']:<29s} -> {out.name:14s} brightest {bright:4.0%} of full scale"
                + (f"   SATURATED {sat:.1%}" if sat > 0.001 else "")
                + f"   [{_fmt_dur(done)} elapsed, ~{_fmt_dur(left)} left]")
    return shots


# =============================================================== CALIBRATE
def calibrate_capture(session, rig, ui, log=print, laser=None):
    """Capture the Gray-code patterns under one laser (laser='Blue' etc., the folder name in
    session.json), or under every laser in the list when laser is None. The orientation-check
    frames are added when the captured laser is the check laser. Other lasers' captures are
    left as they are."""
    cal = session.dir / "calibration"
    cap = cal / "Captures"
    pat_dir = session.lab_path("patterns")
    pats = [pat_dir / f"pat_{i:02d}.png" for i in range(N_PATTERNS)] + [pat_dir / "pat_white.png", pat_dir / "pat_black.png"]
    missing = [p for p in pats if not p.is_file()]
    if missing:
        raise StageError(f"Gray-code patterns missing in {pat_dir} (e.g. {missing[0].name}). "
                         f"They are made by the first cell of graycode.ipynb.")
    plan = scan_plan(session)
    flip = bool(session.settings["calibration"].get("flip_patterns_x", False))
    pat_imgs = [_load_mask(p) for p in pats]
    if flip:
        pat_imgs = [np.ascontiguousarray(m[:, ::-1]) for m in pat_imgs]
        log("Gray-code patterns will be shown FLIPPED left-right (flip_patterns_x). "
            "The orientation-check frames use the unflipped scan masks.")
    masks = pat_imgs + [_black()] + [_load_mask(plan[k]["mask"]) for k in CHECK_MASKS]
    ui.main(lambda: rig.load_masks(masks))
    names = [f"cap_{i:02d}.png" for i in range(N_PATTERNS)] + ["cap_white.png", "cap_black.png"]

    shots = []
    all_lasers = session.settings["calibration"]["lasers"]
    lasers = [l for l in all_lasers if laser is None or l[0] == laser]
    if not lasers:
        raise StageError(f"No laser called {laser!r} in session.json (have: {', '.join(l[0] for l in all_lasers)}).")
    check_laser = session.settings["calibration"]["check_laser"]
    t0 = time.time()
    for folder, nm, exp in lasers:
        prompt = (f"Turn ON the {nm} nm laser ({folder}) and make sure the other lasers are OFF. "
                  f"Nothing else in the setup should move from now on.")
        n_shots = len(names) + (1 + len(CHECK_MASKS) if folder == check_laser else 0)
        pat_names = [f"pat_{i:02d}" for i in range(N_PATTERNS)] + ["white", "black"]
        for idx, name in enumerate(names):
            shots.append(dict(mask_index=idx, save={"basler": cap / folder / name}, fmt="png8",
                              group=folder, light=folder, exposure=exp, prompt=prompt,
                              label=f"[{folder} {idx + 1}/{n_shots}] {pat_names[idx]:7s}"))
        if folder == check_laser:
            base = len(pats)
            shots.append(dict(mask_index=base, save={"basler": cap / "check" / "zero.tiff"}, fmt="raw16",
                              group=folder, light=folder, exposure=exp, prompt=prompt,
                              label=f"[{folder} {len(names) + 1}/{n_shots}] check: black"))
            for j, k in enumerate(CHECK_MASKS, 1):
                shots.append(dict(mask_index=base + j, save={"basler": cap / "check" / f"mask_{k:02d}.tiff"},
                                  fmt="raw16", group=folder, light=folder, exposure=exp, prompt=prompt,
                                  label=f"[{folder} {len(names) + 1 + j}/{n_shots}] check: mask_{k:02d}"))
    desc = ", ".join(f"{f} {nm:g} nm @ {exp:.0f} us" for f, nm, exp in lasers)
    has_check = any(l[0] == check_laser for l in lasers)
    log(f"Calibration capture: {desc} - {len(shots)} frames"
        + (" (incl. 4 orientation-check frames)" if has_check else f" (no check frames: those come with {check_laser})")
        + (", patterns FLIPPED left-right" if flip else ""))
    for folder, nm, exp in lasers:
        # A capture of this laser that was stopped partway leaves a mix of new and old
        # files: delete them and start again from the first pattern.
        pj = cap / folder / "patterns.json"
        if pj.is_file() and json.loads(pj.read_text()).get("incomplete"):
            old = sorted((cap / folder).glob("cap_*.png"))
            if folder == check_laser and (cap / "check").is_dir():
                old += sorted((cap / "check").glob("*.tiff"))
            for p in old:
                p.unlink()
            pj.unlink()
            log(f"Removed {len(old)} files from the stopped {folder} capture - capturing it again from the start.")
    for folder, nm, exp in lasers:          # marked complete below only if the capture finishes
        (cap / folder).mkdir(parents=True, exist_ok=True)
        (cap / folder / "patterns.json").write_text(json.dumps(
            {"incomplete": True, "started": _now(), "nm": nm, "exposure_us": exp, "flip_patterns_x": flip}, indent=2))
    run_shots(rig, shots, ui, log)
    if any(s["save"]["basler"].parent.name == "check" for s in shots):
        (cap / "check" / "scale.json").write_text(json.dumps({"pixel_scale": PIXEL_SCALE}))
    record = json.loads((cal / "capture.json").read_text()) if (cal / "capture.json").is_file() else {}
    per_laser = record.get("per_laser", {})
    for folder, nm, exp in lasers:          # record each laser separately (others are kept)
        sat = [s["save"]["basler"].name for s in shots if s["group"] == folder and s["saturated"] > 0.001]
        entry = {"nm": nm, "exposure_us": exp, "flip_patterns_x": flip, "captured": _now(),
                 "patterns_folder": str(pat_dir), "settle_ms": session.settings["settle_ms"],
                 "saturated_frames": sat, "check_frames": folder == check_laser}
        (cap / folder / "patterns.json").write_text(json.dumps(entry, indent=2))
        per_laser[folder] = entry
        if sat:
            log(f"WARNING: {folder}: {len(sat)} frames have saturated pixels - lower its exposure and capture it again.")
    info = {"per_laser": per_laser, "check_laser": check_laser, "last_capture": _now(),
            "last_seconds": round(time.time() - t0)}
    (cal / "capture.json").write_text(json.dumps(info, indent=2))
    log(f"Calibration captures saved in {cap} ({_fmt_dur(time.time() - t0)})")
    return info


def laser_wavelengths(session):
    """{'Blue': nm, 'Green': nm, 'Red': nm} from session.json's lasers list. The lab notebooks
    call these maps _405/_532/_635 (just names); the real wavelengths go into the fit."""
    wl = {folder: float(nm) for folder, nm, _ in session.settings["calibration"]["lasers"]}
    missing = [k for k in ("Blue", "Green", "Red") if k not in wl]
    if missing:
        raise StageError(f"session.json lasers list is missing {', '.join(missing)} (needed for the wavelength fit).")
    if not wl["Blue"] < wl["Green"] < wl["Red"]:
        raise StageError(f"Laser wavelengths look wrong: {wl}")
    return wl


def patterns_flipped(session, cap):
    """Were the Gray-code patterns shown flipped left-right? Read from the patterns.json each
    capture writes; captures without one fall back to session.json's flip_patterns_x."""
    flips = {}
    for folder in ("Blue", "Green", "Red"):
        p = cap / folder / "patterns.json"
        if p.is_file():
            flips[folder] = bool(json.loads(p.read_text())["flip_patterns_x"])
    unfinished = [f for f in ("Blue", "Green", "Red") if (cap / f / "patterns.json").is_file()
                  and json.loads((cap / f / "patterns.json").read_text()).get("incomplete")]
    if unfinished:
        raise StageError(f"The capture of {', '.join(unfinished)} was stopped before it finished - capture "
                         f"{'it' if len(unfinished) == 1 else 'them'} again on the Calibrate tab.")
    missing = [f for f in ("Blue", "Green", "Red") if not (cap / f / "cap_white.png").is_file()]
    if missing:
        raise StageError(f"Not captured yet: {', '.join(missing)} - capture "
                         f"{'it' if len(missing) == 1 else 'them'} on the Calibrate tab first.")
    if not flips:
        return bool(session.settings["calibration"].get("flip_patterns_x", False))
    if len(set(flips.values())) > 1 or len(flips) < 3:
        desc = ", ".join(k + (": flipped" if v else ": normal") for k, v in flips.items())
        raise StageError(f"The lasers were captured with different pattern flips ({desc}; a missing laser means "
                         "it was captured before this was recorded). Re-capture so all three match.")
    return next(iter(flips.values()))


def unflip_table(cal, log=print):
    """The table from flipped patterns is indexed by PATTERN column (= 1919 - SLM column).
    Reverse its SLM-x axis so it is indexed by the real SLM column, like the scan masks.
    The original is kept as reverse_lookup_pattern_coords.npz. Safe to run twice."""
    table, marker = cal / "reverse_lookup.npz", cal / "reverse_lookup.json"
    if marker.is_file():
        m = json.loads(marker.read_text())
        if m.get("unflipped") and m.get("md5") == _md5(table):
            log("Table is already un-flipped (SLM coordinates) - nothing to do.")
            return
    log("Patterns were flipped: un-flipping the table so it matches the scan masks...")
    with np.load(table) as d:
        data = {k: d[k] for k in d.files}
    table.replace(cal / "reverse_lookup_pattern_coords.npz")
    for k in ("reverse_cam_x", "reverse_cam_y"):
        data[k] = np.ascontiguousarray(data[k][:, :, ::-1])
    np.savez_compressed(table, **data)
    marker.write_text(json.dumps({"unflipped": True, "made": _now(), "md5": _md5(table),
                                  "original": "reverse_lookup_pattern_coords.npz"}, indent=2))
    log("Table un-flipped (original kept as reverse_lookup_pattern_coords.npz).")


def calibrate_process(session, log=print, steps=("graycode", "interpolation", "reverse", "unflip")):
    cal = session.dir / "calibration"
    cap, proj, nbs = cal / "Captures", cal / "proj_map", cal / "notebooks"
    work = cal / "work"
    for d in (proj, nbs, work / "patterns"):
        d.mkdir(parents=True, exist_ok=True)
    for folder in ("Blue", "Green", "Red"):
        if "graycode" in steps and not (cap / folder / "cap_white.png").is_file():
            raise StageError(f"Missing calibration captures in {cap / folder} - run 'calibrate capture' first.")
    flipped = patterns_flipped(session, cap)
    if flipped:
        log("Note: calibration patterns were shown flipped left-right; the table will be un-flipped at the end.")
    code = session.lab_path("calibration_code")
    decoded = Rule("decoded maps", r"/home/ra648/calib/proj_map/decoded_map_(\d+)_\d{6}\.npz", f"{proj}/decoded_map_\\1.npz", 3)
    t0 = time.time()
    if "graycode" in steps:
        log("1/3 graycode.ipynb - decoding the stripe patterns (a few minutes)...")
        rules = [Rule("captures", r"captures/\d{6}/(.+)", f"{cap}/\\1", 9), decoded]
        if not run_notebook(code / "graycode.ipynb", rules, work, nbs / "graycode.ipynb", log):
            raise StageError("graycode.ipynb failed (see above).")
    if "interpolation" in steps:
        log("2/3 interpolation_wavelengths.ipynb - extending to 400-700 nm (needs ~15 GB RAM at full size)...")
        rules = [decoded, Rule("interp map", r"/home/ra648/calib/proj_map/interp_maps_\d{6}\.npz", f"{proj}/interp_maps.npz", 1)]
        wl = laser_wavelengths(session)
        log(f"    laser wavelengths used for the fit: Blue {wl['Blue']:g}, Green {wl['Green']:g}, Red {wl['Red']:g} nm "
            f"(from session.json)")
        wl_rule = CodeRule("laser wavelengths", "wl_meas = np.array([405.0, 532.0, 635.0])",
                           f"wl_meas = np.array([{wl['Blue']:.1f}, {wl['Green']:.1f}, {wl['Red']:.1f}])", 1)
        if not run_notebook(code / "interpolation_wavelengths.ipynb", rules, work, nbs / "interpolation_wavelengths.ipynb",
                            log, code_rules=[wl_rule]):
            raise StageError("interpolation_wavelengths.ipynb failed (see above).")
    if "reverse" in steps:
        log("3/3 Reverse_lookup.py - building the table (about 1.5-2 h at full size; one line per band)...")
        rules = [Rule("interp map", r"/home/ra648/calib/proj_map/interp_maps_\d{6}\.npz", f"{proj}/interp_maps.npz", 1),
                 Rule("table", r"reverse_lookup_\d{6}\.npz", f"{cal}/reverse_lookup.npz", 1)]
        try:
            with np.load(proj / "interp_maps.npz") as d:      # reads only this small entry
                wls = [float(w) for w in d["wl_target"]]
        except Exception:  # noqa
            wls = []
        t_rev = time.time()

        def band_log(line):
            """'band 7: 60.3% of SLM pixels filled' -> with band count, nm and time left."""
            m = re.match(r"\s*band (\d+): (.*)", line)
            if not m:
                return log(line)
            i, n = int(m.group(1)), len(wls) or 31
            done = time.time() - t_rev
            nm = f" ({wls[i]:.0f} nm)" if i < len(wls) else ""
            log(f"    band {i + 1}/{n}{nm}: {m.group(2)}   [{_fmt_dur(done)} elapsed, "
                f"~{_fmt_dur(done / (i + 1) * (n - i - 1))} left]")
        if not run_script(code / "Reverse_lookup.py", rules, work, session.processing_python, band_log):
            raise StageError("Reverse_lookup.py failed (see above).")
        marker = cal / "reverse_lookup.json"
        if marker.is_file():
            marker.unlink()                     # a freshly built table is in pattern coordinates again
    if ("unflip" in steps or "reverse" in steps) and flipped:
        unflip_table(cal, log)
    log(f"Calibration processing done in {(time.time() - t0) / 60:.1f} min -> {cal / 'reverse_lookup.npz'}")


def calibrate_check(session, log=print):
    cal = session.cal_dir          # may be another session's calibration (use_from_session)
    chk = cal / "Captures" / "check"
    table = cal / "reverse_lookup.npz"
    if not table.is_file():
        raise StageError("No table yet - run 'calibrate process' first.")
    files = [chk / "zero.tiff"] + [chk / f"mask_{k:02d}.tiff" for k in CHECK_MASKS]
    if not all(f.is_file() for f in files):
        raise StageError(f"Check frames missing in {chk}.")
    plan = scan_plan(session)
    sc = check_scale(chk)
    zero = read_basler(files[0], sc)
    frames = {k: read_basler(chk / f"mask_{k:02d}.tiff", sc) for k in CHECK_MASKS}
    masks = {k: _load_mask(plan[k]["mask"]) > 0 for k in CHECK_MASKS}
    laser = session.settings["calibration"]["check_laser"]
    nm = next(nm for f, nm, _ in session.settings["calibration"]["lasers"] if f == laser)
    log(f"Orientation check (calibration, {nm} nm laser):")
    res = orientation_check(table, zero, frames, masks, band_nm=nm, log=log, strict_alignment=True)
    res["checked"] = _now()
    (cal / "check.json").write_text(json.dumps(res, indent=2))
    return res


# =============================================================== CAPTURE
def capture_scene(session, rig, scene, ui, log=print, use_blackfly=None):
    sd = session.scene_dir(scene)
    if (sd / "manifest.json").exists():
        raise StageError(f"Scene '{scene}' already exists in this session - pick a new name.")
    if (sd / "positions").is_dir():
        raise StageError(f"'{scene}' holds sparse positions - use another name for the full scan.")
    if not (session.settings["capture"]["zero_at_start"] or session.settings["capture"]["zero_at_end"]):
        raise StageError("At least one all-zero capture (start or end) is needed.")
    cap = sd / "captures"
    plan = scan_plan(session)
    s = session.settings
    exp = s["capture"]["exposure_us"]
    bf = s["blackfly"]["enabled"] if use_blackfly is None else use_blackfly
    masks = [_black()] + [_load_mask(r["mask"]) for r in plan]
    ui.main(lambda: rig.load_masks(masks))
    # keep a copy of the masks with the scene, so it is self-contained
    (sd / "masks").mkdir(parents=True, exist_ok=True)
    for r in plan:
        shutil.copy2(r["mask"], sd / "masks" / Path(r["mask"]).name)

    shots = []
    if s["capture"]["zero_at_start"]:
        shots.append(dict(mask_index=0, save={"basler": cap / "zero_start.tiff"}, fmt="raw16", group="scene", exposure=exp))
    for j, r in enumerate(plan, 1):
        shots.append(dict(mask_index=j, fmt="raw16", group="scene", exposure=exp,
                          save={"basler": cap / f"scan_{r['frame']:02d}.tiff",
                                "blackfly": (cap / "rgb" / f"rgb_{r['frame']:02d}.png") if bf else None}))
    if s["capture"]["zero_at_end"]:
        shots.append(dict(mask_index=0, save={"basler": cap / "zero_end.tiff"}, fmt="raw16", group="scene", exposure=exp))
    bf_settings = None                      # what the Blackfly reports (gamma / white balance)
    if bf and getattr(rig, "blackfly", None) is not None and hasattr(rig.blackfly, "raw_settings"):
        try:
            bf_settings = json.loads(json.dumps(rig.blackfly.raw_settings(), default=str))
            if bf_settings.get("GammaEnable") is True or str(bf_settings.get("BalanceWhiteAuto", "Off")) not in ("Off", "n/a"):
                log("WARNING: Blackfly gamma or white balance is ON - its frames will not be raw.")
        except Exception as e:  # noqa
            bf_settings = {"error": str(e)}
    log(f"Scene capture '{scene}': {len(shots)} frames at {exp:.0f} us. Keep the scene completely still.")
    t0 = time.time()
    run_shots(rig, shots, ui, log)

    zeros = [read_basler(cap / n) for n in ("zero_start.tiff", "zero_end.tiff") if (cap / n).is_file()]
    drift = None
    if len(zeros) == 2:
        drift = float(abs(zeros[0].mean() - zeros[1].mean()) / max(zeros[0].mean(), 1e-6))
        if drift > 0.02:
            log(f"WARNING: the all-zero frame changed by {drift:.1%} between start and end - "
                f"light level or scene may have changed during the capture.")
    cv2.imwrite(str(cap / "zero.tiff"), np.round(np.mean(zeros, axis=0)).astype(np.uint16))

    manifest = {
        "scene": scene, "session": session.name, "captured": _now(), "seconds": round(time.time() - t0),
        "exposure_us": exp, "settle_ms": s["settle_ms"], "pixel_scale": PIXEL_SCALE,
        "pixel_values": "raw 12-bit x 16 (0-65520) in uint16 TIFF - divide by pixel_scale",
        "bright_fraction": round(float(np.median([x["bright"] for x in shots[1:-1] or shots])), 3),
        "table": str(session.table), "table_md5": _md5(session.table) if session.table.is_file() else None,
        "zero_frames": [n for n in ("zero_start.tiff", "zero_end.tiff") if (cap / n).is_file()],
        "zero_drift": drift, "blackfly": bool(bf), "blackfly_settings": bf_settings,
        "frames": [{"frame": r["frame"], "dy": r["dy"], "dx": r["dx"], "mask": f"masks/{Path(r['mask']).name}",
                    "capture": f"captures/scan_{r['frame']:02d}.tiff"} for r in plan],
        "saturated_frames": [Path(x["save"]["basler"]).name for x in shots if x["saturated"] > 0.001],
    }
    (sd / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if manifest["saturated_frames"]:
        log(f"WARNING: {len(manifest['saturated_frames'])} frames have saturated pixels - lower the exposure.")
    log(f"Exposure {exp:.0f} us: bright parts reach {manifest['bright_fraction']:.0%} of full scale.")
    if manifest["bright_fraction"] < 0.4:
        log(f"WARNING: the frames are dark (bright parts at {manifest['bright_fraction']:.0%} of full scale) - "
            f"a longer exposure would give less noise.")
    log(f"Scene saved in {sd}")
    return manifest


# =============================================================== SPARSE CAPTURE (moving object)
def default_sparse_mask(session):
    """mask_00 of the scan masks (the single mask used for sparse measurements)."""
    return session.lab_path("scan_masks") / "mask_00.png"


def capture_position(session, rig, scene, ui, log=print, mask_path=None, use_blackfly=None):
    """One position of a moving object: the sparse mask (with a Blackfly frame), then the
    black mask (Basler only), at the same exposure, back to back. Saved in
    scenes/<scene>/positions/pos_NN/: mask.tiff, zero.tiff (raw 12-bit), rgb.png + rgb.raw,
    position.json. Returns the position record."""
    sd = session.scene_dir(scene)
    if (sd / "manifest.json").exists():
        raise StageError(f"'{scene}' is a full-scan scene - use another name for the sparse positions.")
    mask_path = Path(mask_path) if mask_path else default_sparse_mask(session)
    mask = _load_mask(mask_path)
    s = session.settings
    exp = s["capture"]["exposure_us"]
    bf = s["blackfly"]["enabled"] if use_blackfly is None else use_blackfly
    done = sorted(p for p in (sd / "positions").glob("pos_*") if (p / "position.json").is_file()) \
        if (sd / "positions").is_dir() else []
    k = len(done) + 1
    pd = sd / "positions" / f"pos_{k:02d}"
    if pd.exists():
        shutil.rmtree(pd)                    # a stopped attempt at this position: start it again
    pd.mkdir(parents=True)
    ui.main(lambda: rig.load_masks([mask, _black()]))
    shots = [dict(mask_index=0, fmt="raw16", group="pos", exposure=exp,
                  save={"basler": pd / "mask.tiff", "blackfly": (pd / "rgb.png") if bf else None}),
             dict(mask_index=1, fmt="raw16", group="pos", exposure=exp, save={"basler": pd / "zero.tiff"})]
    bf_settings = None
    if bf and getattr(rig, "blackfly", None) is not None and hasattr(rig.blackfly, "raw_settings"):
        try:
            bf_settings = json.loads(json.dumps(rig.blackfly.raw_settings(), default=str))
        except Exception as e:  # noqa
            bf_settings = {"error": str(e)}
    log(f"Position {k} of '{scene}': {mask_path.name} then black, {exp:.0f} us. Keep the object still.")
    t0 = time.time()
    run_shots(rig, shots, ui, log)
    m = read_basler(pd / "mask.tiff", PIXEL_SCALE)
    z = read_basler(pd / "zero.tiff", PIXEL_SCALE)
    rec = {"position": k, "scene": scene, "session": session.name, "captured": _now(),
           "seconds": round(time.time() - t0, 1), "exposure_us": exp, "settle_ms": s["settle_ms"],
           "mask_file": str(mask_path), "pixel_scale": PIXEL_SCALE,
           "pixel_values": "raw 12-bit x 16 (0-65520) in uint16 TIFF - divide by pixel_scale",
           "mask_mean": float(m.mean()), "zero_mean": float(z.mean()), "bright_fraction": round(shots[0]["bright"], 3),
           "saturated": {"mask.tiff": shots[0]["saturated"], "zero.tiff": shots[1]["saturated"]},
           "blackfly": bool(bf), "blackfly_settings": bf_settings,
           "table": str(session.table)}
    (pd / "position.json").write_text(json.dumps(rec, indent=2))
    if rec["zero_mean"] >= rec["mask_mean"]:
        log("WARNING: the black frame is not darker than the mask frame - did the SLM change?")
    if shots[0]["saturated"] > 0.001:
        log(f"WARNING: {shots[0]['saturated']:.1%} of the mask frame is saturated - lower the exposure.")
    log(f"Exposure {exp:.0f} us: bright parts reach {shots[0]['bright']:.0%} of full scale.")
    if shots[0]["bright"] < 0.4:
        log(f"WARNING: the mask frame is dark (bright parts at {shots[0]['bright']:.0%}) - consider a longer exposure.")
    log(f"Position {k} saved in {pd} ({rec['seconds']} s). Move the object for the next position.")
    return rec


def export_positions(session, scene, log=print, background="none"):
    """Sparse cube + RGB on SLM coordinates for every captured position of a scene, each with
    its own black frame subtracted (export_sparse.py --positions). -> scenes/<scene>/sparse_export/"""
    import export_sparse
    sd = session.scene_dir(scene)
    guide = session.cal_dir / "guide" / "guide_to_slm.npz"
    if not guide.is_file():
        raise StageError(f"No Blackfly -> SLM map at {guide} (run guide_calib.ipynb first).")
    if not any((sd / "positions").glob("pos_*/mask.tiff")):
        raise StageError(f"No captured positions in scene '{scene}'.")
    return export_sparse.run([sd], session.name, guide, sd / "sparse_export", positions=True,
                             background=background, log=log)


# =============================================================== CUBE
def scene_check(session, scene, log=print):
    sd = session.scene_dir(scene)
    sc = scene_scale(sd)
    zero = read_basler(sd / "captures" / "zero.tiff", sc)
    frames = {k: read_basler(sd / "captures" / f"scan_{k:02d}.tiff", sc) for k in CHECK_MASKS}
    masks = {k: cv2.imread(str(sd / "masks" / f"mask_{k:02d}.png"), cv2.IMREAD_GRAYSCALE) > 0 for k in CHECK_MASKS}
    log("Orientation check (scene captures vs table):")
    res = orientation_check(session.table, zero, frames, masks, band_nm=550, log=log, strict_alignment=False)
    (sd / "check.json").write_text(json.dumps(res, indent=2))
    return res


def make_cube(session, scene, log=print, force=False, skip_existing=True):
    sd = session.scene_dir(scene)
    man_path = sd / "manifest.json"
    if not man_path.is_file():
        raise StageError(f"No captured scene '{scene}' in session {session.name}.")
    if not session.table.is_file():
        raise StageError(f"No table at {session.table} - calibrate first.")
    man = json.loads(man_path.read_text())
    res = scene_check(session, scene, log)
    if not res["passed"] and session.settings["cube"]["require_check"] and not force:
        raise StageError("Orientation check failed, so the cube would be wrong: " + " ".join(res["problems"])
                         + " (Use --force to make it anyway.)")

    sc = scene_scale(sd)
    code = session.lab_path("calibration_code")
    out = sd / "cubes"
    out.mkdir(parents=True, exist_ok=True)
    zero = sd / "captures" / "zero.tiff"
    qe = code / "SonyIMX183.csv"
    done, failed = [], []
    t0 = time.time()
    for n, fr in enumerate(man["frames"], 1):
        k = fr["frame"]
        cube_path = out / f"cube_{k:02d}.npz"
        if skip_existing and cube_path.is_file():
            done.append(cube_path)
            continue
        rules = [Rule("mask", r".*mask[^/]*\.png", str(sd / fr["mask"]), 1),
                 Rule("capture", r".*/img_[^/]*\.tiff", str(sd / fr["capture"]), 1),
                 Rule("zero", r".*/zero_[^/]*\.tiff", str(zero), 1),
                 Rule("table", r".*reverse_lookup[^/]*\.npz", str(session.table), 1),
                 Rule("QE", r".*SonyIMX183\.csv", str(qe), 1),
                 Rule("output", r".*cube[^/]*\.npz", str(cube_path), 1)]
        t = time.time()
        quiet = []
        code_rules = []
        if sc != 1:          # x16 TIFFs: hand meas2hyper raw 12-bit values, as for every earlier cube
            code_rules = [CodeRule("pixel scale", "g_measured = im_mask.astype(np.float32) - im_zero.astype(np.float32)",
                                   f"g_measured = (im_mask.astype(np.float32) - im_zero.astype(np.float32)) / {float(sc)}", 1)]
        ok = run_notebook(code / "meas2hyper.ipynb", rules, out, None, log=quiet.append, code_rules=code_rules)
        if ok and cube_path.is_file():
            done.append(cube_path)
            log(f"[{n}/{len(man['frames'])}] frame {k:02d}: ok ({time.time() - t:.0f}s)")
        else:
            failed.append(k)
            for line in quiet:
                if "ERROR" in line:
                    log(line)
            log(f"[{n}/{len(man['frames'])}] frame {k:02d}: FAILED")
    if not done:
        raise StageError("No cubes were made.")
    shape, filled = combine_cubes(done, out / "cube_combined.npz")
    render_preview(out / "cube_combined.npz", out / "preview.png")
    info = {"made": _now(), "minutes": round((time.time() - t0) / 60, 1), "frames_ok": len(done),
            "frames_failed": failed, "filled_fraction": filled, "table": str(session.table), "check": res["passed"]}
    if session.settings["cube"].get("full", True) and not failed:
        # Fully sampled cube (no NaN): table extended past its edge, the rest filled from
        # neighbours; a 'source' array records which values were measured.
        import full_cube
        log("Making the fully sampled cube (cube_full.npz)…")
        src = full_cube.build(full_cube.scene_pairs(sd), zero, session.table, out=out / "cube_full.npz",
                              qe_csv=qe, log=log, scale=sc)
        info["full"] = {"calibrated": float((src == 2).mean()), "extended": float((src == 1).mean()),
                        "filled": float((src == 0).mean())}
    (out / "cube.json").write_text(json.dumps(info, indent=2))
    log(f"Combined cube {shape}: {filled:.1%} of SLM pixels have a spectrum -> {out / 'cube_combined.npz'}")
    if failed:
        log(f"{len(failed)} frames failed: {failed}. Re-run to retry them (finished frames are kept).")
    return info


def combine_cubes(files, out_path):
    total = count = wl = None
    for f in files:
        with np.load(f) as d:
            c = d["cube"]
            if total is None:
                wl = d["wavelengths"]
                total = np.zeros(c.shape, np.float32)
                count = np.zeros(c.shape, np.uint8)
            ok = np.isfinite(c)
            total[ok] += c[ok]
            count[ok] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        cube = np.where(count > 0, total / count, np.nan).astype(np.float32)
    coverage = count.max(axis=2)
    np.savez_compressed(out_path, cube=cube, wavelengths=wl, coverage=coverage)
    return cube.shape, float((coverage > 0).mean())


def cube_to_rgb(cube, wl):
    """Quick display rendering (not colorimetric): Gaussian R/G/B sensitivities."""
    c = np.nan_to_num(cube.astype(np.float32))
    w = np.stack([np.exp(-0.5 * ((wl - mu) / 35.0) ** 2) for mu in (610, 545, 455)], 1)  # (Nt, 3) R,G,B
    rgb = c @ (w / w.sum(0))
    hi = np.percentile(rgb[rgb > 0], 99.5) if (rgb > 0).any() else 1.0
    return (np.clip(rgb / hi, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def render_preview(cube_npz, png_path):
    with np.load(cube_npz) as d:
        rgb = cube_to_rgb(d["cube"], d["wavelengths"])
    cv2.imwrite(str(png_path), rgb[..., ::-1])

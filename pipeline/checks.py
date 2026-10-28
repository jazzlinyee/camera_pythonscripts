"""
Does the table (reverse lookup) match how the hardware behaves right now?

Uses three frames taken with scan masks 00, 07 and 14 plus an all-zero frame:
  * mask_07 is mask_00 moved 21 SLM columns right -> smears must move right on the
    camera by the amount the table predicts (catches a left-right flip),
  * mask_14 is mask_00 moved 3 SLM rows down -> smears must move down (catches an
    upside-down flip),
  * the table must put mask_00's open pixels on top of the light it actually
    produced (catches the table being shifted).
"""

import cv2
import numpy as np

SHIFT_PAIRS = {"x": (0, 7, 0, 21), "y": (0, 14, 3, 0)}   # (mask a, mask b, rows down, cols right)


def _hp(img, zero, sigma):
    a = img.astype(np.float32) - zero.astype(np.float32)
    return a - cv2.GaussianBlur(a, (0, 0), sigma)


def _best_shift(A, B, xs, ys):
    """Integer shift (dx, dy) of B relative to A with the highest normalized correlation."""
    H, W = A.shape
    m = max(max(abs(x) for x in xs), max(abs(y) for y in ys)) + 1
    a = A[m:H - m, m:W - m]
    a = (a - a.mean()) / (a.std() + 1e-9)
    best = (-2.0, 0, 0)
    for dy in ys:
        for dx in xs:
            b = B[m + dy:H - m + dy, m + dx:W - m + dx]
            c = float((a * ((b - b.mean()) / (b.std() + 1e-9))).mean())
            if c > best[0]:
                best = (c, dx, dy)
    return best


def orientation_check(table_path, zero, frames, masks, band_nm=550.0, log=print, strict_alignment=True):
    """frames/masks: dicts {0: ..., 7: ..., 14: ...} of camera images / SLM masks (bool).

    strict_alignment: the alignment test is reliable under a single laser line; for a
    broadband scene it is reported as a warning only.
    """
    with np.load(table_path) as d:
        wl = d["wl_target"]
        b = int(np.argmin(np.abs(wl - band_nm)))
        rx = d["reverse_cam_x"][b].astype(np.float32)
        ry = d["reverse_cam_y"][b].astype(np.float32)
    col_px = float(np.nanmedian(np.diff(rx, axis=1)))      # camera px per SLM column (signed)
    row_px = float(np.nanmedian(np.diff(ry, axis=0)))      # camera px per SLM row (signed)
    res = {"band_nm": float(wl[b]), "camera_px_per_slm_col": col_px, "camera_px_per_slm_row": row_px,
           "passed": True, "problems": []}

    # Region of the camera the table covers (central part), to keep the search quick.
    ok = np.isfinite(rx) & np.isfinite(ry)
    ys_, xs_ = np.nonzero(ok)
    u0, u1 = np.percentile(rx[ok], [25, 75]).astype(int)
    v0, v1 = np.percentile(ry[ok], [25, 75]).astype(int)
    pad = int(abs(col_px) * 40)
    H, W = zero.shape
    crop = (slice(max(0, v0 - pad), min(H, v1 + pad)), slice(max(0, u0 - pad), min(W, u1 + pad)))
    sigma = max(1.0, 1.3 * abs(row_px))
    hp = {k: _hp(frames[k], zero, sigma) for k in frames}

    for axis, (ka, kb, drow, dcol) in SHIFT_PAIRS.items():
        pred_x, pred_y = dcol * col_px, drow * row_px
        if axis == "x":
            R = int(abs(pred_x) * 1.6) + 3
            xs, ys = range(-R, R + 1), range(-3, 4)
        else:
            R = int(abs(pred_y) * 2) + 3
            xs, ys = range(-3, 4), range(-R, R + 1)
        corr, mx, my = _best_shift(hp[ka][crop], hp[kb][crop], xs, ys)
        pred = pred_x if axis == "x" else pred_y
        meas = mx if axis == "x" else my
        good = np.sign(meas) == np.sign(pred) and abs(meas - pred) <= max(3, 0.35 * abs(pred))
        res[f"shift_{axis}"] = {"predicted_px": round(pred, 1), "measured_px": int(meas), "corr": round(corr, 3), "ok": bool(good)}
        word = "left-right" if axis == "x" else "up-down"
        log(f"  {word} shift mask_{ka:02d}->mask_{kb:02d}: table predicts {pred:+.1f} px, "
            f"camera shows {meas:+d} px  {'OK' if good else 'MISMATCH'}")
        if not good:
            res["passed"] = False
            if np.sign(meas) == -np.sign(pred) and abs(abs(meas) - abs(pred)) <= max(3, 0.35 * abs(pred)):
                res["problems"].append(f"The image is FLIPPED {word} compared with the calibration.")
            else:
                res["problems"].append(f"Smears move {meas:+d} px {word} but the table predicts {pred:+.1f} px.")

    # Alignment: does the table land mask_00's open pixels on the light?
    m = masks[0]
    ring = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~m

    def contrast(ox, oy):
        s = cv2.remap(hp[0], rx + ox, ry + oy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        return float(np.nanmean(s[m]) - np.nanmean(s[ring]))

    c0 = contrast(0, 0)
    best = max((contrast(ox, oy), ox, oy) for oy in range(-6, 7) for ox in range(-6, 7))
    res["alignment"] = {"contrast_as_is": round(c0, 1), "best_contrast": round(best[0], 1),
                        "best_offset_px": [best[1], best[2]]}
    aligned = c0 > 0 and c0 >= 0.7 * best[0] and max(abs(best[1]), abs(best[2])) <= 2
    res["alignment"]["ok"] = bool(aligned)
    log(f"  alignment at {wl[b]:.0f} nm: contrast {c0:.0f} as-is, best {best[0]:.0f} at offset "
        f"({best[1]:+d}, {best[2]:+d}) px  {'OK' if aligned else ('MISALIGNED' if strict_alignment else 'unclear (info only - only reliable under a single laser)')}")
    if not aligned:
        if strict_alignment:
            res["passed"] = False
        (res["problems"] if strict_alignment else res.setdefault("warnings", [])).append(f"The table looks shifted by about ({best[1]:+d}, {best[2]:+d}) camera px "
                               f"(or the setup moved since calibration).")
    log("  CHECK PASSED" if res["passed"] else "  CHECK FAILED: " + " ".join(res["problems"]))
    return res

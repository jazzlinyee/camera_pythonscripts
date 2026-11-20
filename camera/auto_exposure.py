"""
auto_exposure.py - find the longest exposure that does not oversaturate (added 2026-11-20).

Used by the camera GUI's "Auto Exposure" button and by the pipeline (Auto buttons and the
"Find the exposure automatically before each capture" option). It only needs a function
that takes an exposure, sets it on the camera and returns a fresh frame (or a list of
frames, e.g. one per mask); it never talks to a camera itself.

"Oversaturated" (changed 2026-11-20) = a saturated AREA: somewhere a block of at least
4 x 4 pixels is entirely at full scale (4095 for the 12-bit Basler, 255 for the 8-bit
Blackfly). That is what shows up as a red patch in the Live window (it displays every 4th
pixel). Single saturated pixels - hot pixels, which multiply at long exposures, and tiny
glints - are ignored, however many there are. For the Blackfly's Bayer mosaic each colour
plane is checked on its own (a 2 x 2 block in one plane = a 4 x 4 area on the sensor).

Goal (changed 2026-11-20 at Rika's request): the exposure just before parts turn red, not
a safety margin below it. How it searches (usually 5-9 frames):
  1. Take a frame at the starting exposure.
  2. Saturated -> halve the exposure until it is not.
  3. Not saturated -> jump to where the brightest parts should just reach full scale
     (brightness grows in proportion to exposure), at least 5 % further.
  4. Once one exposure is known to be fine and a longer one to saturate, bisect between them
     until they are less than 2 % apart, and use the longer one that did NOT saturate.
The result is always an exposure whose frame was actually seen to be not oversaturated (or
the shortest allowed one, with a note, if even that saturates).
"""

import math

import numpy as np

DEFAULT_MAX_SATURATED = 1e-4      # brightness = the level only this fraction of pixels exceed
AREA = 4                          # a saturated area = a fully saturated AREA x AREA block
EDGE_GAP = 0.02                   # stop when "fine" and "saturated" are less than 2 % apart
AIM = 0.99                        # where the jump in step 3 aims (fraction of full scale)


def saturated_blocks(mask, b):
    """How many b x b windows of the boolean image `mask` are entirely True."""
    h, w = mask.shape
    if b <= 1:
        return int(np.count_nonzero(mask))
    if h < b or w < b or not mask.any():
        return 0
    r = mask[:h - b + 1].copy()
    for i in range(1, b):
        r &= mask[i:h - b + 1 + i]
    c = r[:, :w - b + 1].copy()
    for j in range(1, b):
        c &= r[:, j:w - b + 1 + j]
    return int(np.count_nonzero(c))


def measure(frame, full_scale, max_saturated=DEFAULT_MAX_SATURATED, bayer=False):
    """(saturated fraction, brightness, saturated areas) of one frame.
    saturated areas = number of fully saturated 4 x 4 blocks (per colour plane for a Bayer
    frame: 2 x 2 blocks); any > 0 means a part of the image clips. saturated fraction = all
    pixels at full scale, hot pixels included (for the log only). Brightness = the level that
    only `max_saturated` of the pixels exceed, as a fraction of full scale."""
    a = np.asarray(frame)
    full = a >= full_scale
    sat = float(np.count_nonzero(full)) / a.size
    if bayer and a.ndim == 2:
        areas = sum(saturated_blocks(full[i::2, j::2], max(1, AREA // 2)) for i in (0, 1) for j in (0, 1))
    else:
        areas = saturated_blocks(full, AREA)
    sub = a[::2, ::2].ravel() if a.ndim == 2 and a.size > 4_000_000 else a.ravel()
    if areas == 0:                 # only single saturated (hot) pixels: leave them out of the
        below = sub[sub < full_scale]          # brightness, or it would read "full" already
        if below.size:
            sub = below
    k = max(1, int(math.ceil(max_saturated * sub.size)))
    level = float(np.partition(sub, sub.size - k)[sub.size - k]) / full_scale
    return sat, level, areas


def _worst(frames, full_scale, max_saturated, bayer):
    if not isinstance(frames, (list, tuple)):
        frames = [frames]
    vals = [measure(f, full_scale, max_saturated, bayer) for f in frames if f is not None]
    if not vals:
        raise RuntimeError("no frame from the camera")
    return max(v[0] for v in vals), max(v[1] for v in vals), max(v[2] for v in vals)


def search(grab, start_us, full_scale, min_us=20.0, max_us=2_000_000.0,
           max_saturated=DEFAULT_MAX_SATURATED, max_steps=16, log=print,
           stopped=lambda: False, name="camera", bayer=False, **_ignored):
    """Find the longest exposure (us) whose frame is not oversaturated.

    grab(exposure_us) -> frame or list of frames, taken entirely at that exposure.
    Returns a dict: exposure_us, brightness, saturated, steps [{exposure_us, saturated,
    brightness}], note ('' when all went well).
    """
    clamp = lambda e: float(min(max(e, min_us), max_us))
    exp = clamp(start_us or min_us * 100)
    good = None           # (exposure, brightness, saturated) of the longest unsaturated frame
    bad = None            # shortest exposure seen saturated
    steps, note = [], ""
    for _ in range(max_steps):
        if stopped():
            raise RuntimeError("stopped")
        sat, level, areas = _worst(grab(exp), full_scale, max_saturated, bayer)
        steps.append({"exposure_us": round(exp, 1), "saturated": round(sat, 6), "brightness": round(level, 3),
                      "saturated_areas": areas})
        over = areas > 0
        log(f"  {name} auto exposure: {exp:,.0f} us -> brightest {level:.0%}"
            + (f", saturated area ({areas} blocks clip; too long)" if over
               else (f", ok (only single saturated pixels: {sat:.3%})" if sat > 0 else ", ok")))
        if over:
            bad = exp if bad is None else min(bad, exp)
            if exp <= min_us:
                note = f"saturated even at the shortest allowed exposure ({min_us:g} us)"
                break
        elif good is None or exp > good[0]:
            good = (exp, level, sat)
        if good and good[0] >= max_us:
            note = f"still only {good[1]:.0%} bright at the longest allowed exposure ({max_us:,.0f} us)"
            break
        if good and bad and bad / good[0] <= 1 + EDGE_GAP:
            break                                     # found the edge
        if good is None:
            new = exp / 2                             # everything so far saturated
        elif bad is None:
            g_exp, g_level = good[0], good[1]
            jump = g_exp * AIM / g_level if g_level > 0.005 else g_exp * 8
            new = max(jump, g_exp * 1.05)             # always probe further until it saturates
        else:
            new = (good[0] + bad) / 2                 # bisect
        exp = clamp(new)
    else:
        note = note or f"stopped after {max_steps} test frames"
    if good is None:
        result = {"exposure_us": float(min_us), "brightness": None, "saturated": steps[-1]["saturated"],
                  "saturated_areas": steps[-1]["saturated_areas"]}
        note = note or "no unsaturated exposure found"
    else:
        result = {"exposure_us": round(good[0], 1), "brightness": round(good[1], 3), "saturated": round(good[2], 6),
                  "saturated_areas": 0}
    result.update(steps=steps, note=note, full_scale=full_scale, max_saturated=max_saturated,
                  min_us=min_us, max_us=max_us, edge_gap=EDGE_GAP, area_px=AREA, bayer=bayer)
    log(f"  {name} auto exposure: using {result['exposure_us']:,.0f} us"
        + (f" (brightest {result['brightness']:.0%}, just below saturation)" if result["brightness"] is not None
           and not note else (f" (brightest {result['brightness']:.0%})" if result["brightness"] is not None else ""))
        + (f" - NOTE: {note}" if note else ""))
    return result

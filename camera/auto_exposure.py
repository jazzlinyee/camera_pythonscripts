"""
auto_exposure.py - find the longest exposure that does not oversaturate (added 2026-11-20).

Used by the camera GUI's "Auto Exposure" button and by the pipeline (Auto buttons and the
"Find the exposure automatically before each capture" option). It only needs a function
that takes an exposure, sets it on the camera and returns a fresh frame (or a list of
frames, e.g. one per mask); it never talks to a camera itself.

"Oversaturated" = more than `max_saturated` of all pixels at full scale (4095 for the 12-bit
Basler, 255 for the 8-bit Blackfly). Default 0.001 % (Rika, 2026-11-20): about 200 of the
Basler's 20 million pixels, enough to ignore a few hot pixels, while the Live window's
"saturated ... % of pixels" still reads 0.00 %.

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

DEFAULT_MAX_SATURATED = 1e-5      # allowed fraction of pixels at full scale (0.001 %)
EDGE_GAP = 0.02                   # stop when "fine" and "saturated" are less than 2 % apart
AIM = 0.99                        # where the jump in step 3 aims (fraction of full scale)


def measure(frame, full_scale, max_saturated=DEFAULT_MAX_SATURATED, bayer=False):
    """(saturated fraction, brightness) of one frame. saturated fraction = pixels at full
    scale / all pixels. Brightness = the level that only `max_saturated` of the pixels
    exceed (with 0.001 %: the 99.999th percentile), as a fraction of full scale. `bayer` is accepted for compatibility and not needed."""
    a = np.asarray(frame)
    sat = float(np.count_nonzero(a >= full_scale)) / a.size
    sub = a[::2, ::2].ravel() if a.ndim == 2 and a.size > 4_000_000 else a.ravel()
    k = max(1, int(math.ceil(max_saturated * sub.size)))
    level = float(np.partition(sub, sub.size - k)[sub.size - k]) / full_scale
    return sat, level


def _worst(frames, full_scale, max_saturated, bayer):
    if not isinstance(frames, (list, tuple)):
        frames = [frames]
    vals = [measure(f, full_scale, max_saturated, bayer) for f in frames if f is not None]
    if not vals:
        raise RuntimeError("no frame from the camera")
    return max(v[0] for v in vals), max(v[1] for v in vals)


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
        sat, level = _worst(grab(exp), full_scale, max_saturated, bayer)
        steps.append({"exposure_us": round(exp, 1), "saturated": round(sat, 6), "brightness": round(level, 3)})
        over = sat > max_saturated
        log(f"  {name} auto exposure: {exp:,.0f} us -> brightest {level:.0%}"
            + f", {sat:.3%} of pixels saturated" + (" (too long)" if over else " (ok)"))
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
        result = {"exposure_us": float(min_us), "brightness": None, "saturated": steps[-1]["saturated"]}
        note = note or "no unsaturated exposure found"
    else:
        result = {"exposure_us": round(good[0], 1), "brightness": round(good[1], 3), "saturated": round(good[2], 6)}
    result.update(steps=steps, note=note, full_scale=full_scale, max_saturated=max_saturated,
                  min_us=min_us, max_us=max_us, edge_gap=EDGE_GAP)
    log(f"  {name} auto exposure: using {result['exposure_us']:,.0f} us"
        + (f" (brightest {result['brightness']:.0%}, just below saturation)" if result["brightness"] is not None
           and not note else (f" (brightest {result['brightness']:.0%})" if result["brightness"] is not None else ""))
        + (f" - NOTE: {note}" if note else ""))
    return result

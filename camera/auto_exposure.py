"""
auto_exposure.py - find the longest exposure that does not oversaturate (added 2026-11-20).

Used by the camera GUI's "Auto exposure" button and by the pipeline (Auto buttons and the
"Find exposure before capturing" option). It only needs a function that takes an exposure,
sets it on the camera and returns a fresh frame (or a list of frames, e.g. one per mask);
it never talks to a camera itself.

"Not oversaturated" = at most `max_saturated` of the pixels (default 0.01 %, so a few hot
pixels don't count) are at full scale (4095 for the 12-bit Basler, 255 for the 8-bit
Blackfly). The aim is to put the brightest parts at `target` (default 95 %) of full scale.

How it searches (usually 3-8 frames):
  1. Take a frame at the starting exposure.
  2. Saturated -> halve the exposure until it is not.
  3. Not saturated -> brightness grows in proportion to exposure, so jump straight to
     exposure * target / brightness.
  4. When a too-long exposure is known, never go past halfway to it (bisection), and stop when
     the brightest parts are close to the target or the gap is under 3 %.
The result is always an exposure whose frame was actually seen to be NOT saturated (or the
shortest allowed one, with a note, if even that saturates).
"""

import math

import numpy as np

DEFAULT_MAX_SATURATED = 1e-4      # fraction of pixels allowed at full scale (0.01 %)
DEFAULT_TARGET = 0.95             # brightest parts at 95 % of full scale
CLOSE_ENOUGH = 0.06               # stop when within 6 % (of full scale) below the target


def measure(frame, full_scale, max_saturated=DEFAULT_MAX_SATURATED):
    """(saturated fraction, brightness) of one frame. Brightness = the level that only
    `max_saturated` of the pixels exceed, as a fraction of full scale, so it ignores the
    same few hot pixels the saturation rule ignores."""
    a = np.asarray(frame)
    sat = float(np.count_nonzero(a >= full_scale)) / a.size
    sub = a[::2, ::2].ravel() if a.ndim == 2 and a.size > 4_000_000 else a.ravel()
    k = max(1, int(math.ceil(max_saturated * sub.size)))
    level = float(np.partition(sub, sub.size - k)[sub.size - k]) / full_scale
    return sat, level


def _worst(frames, full_scale, max_saturated):
    if not isinstance(frames, (list, tuple)):
        frames = [frames]
    vals = [measure(f, full_scale, max_saturated) for f in frames if f is not None]
    if not vals:
        raise RuntimeError("no frame from the camera")
    return max(v[0] for v in vals), max(v[1] for v in vals)


def search(grab, start_us, full_scale, min_us=20.0, max_us=2_000_000.0,
           max_saturated=DEFAULT_MAX_SATURATED, target=DEFAULT_TARGET, max_steps=12,
           log=print, stopped=lambda: False, name="camera"):
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
        sat, level = _worst(grab(exp), full_scale, max_saturated)
        steps.append({"exposure_us": round(exp, 1), "saturated": round(sat, 6), "brightness": round(level, 3)})
        over = sat > max_saturated
        log(f"  {name} auto exposure: {exp:,.0f} us -> brightest {level:.0%}"
            + (f", {sat:.3%} saturated (too long)" if over else ", ok"))
        if over:
            bad = exp if bad is None else min(bad, exp)
            if exp <= min_us:
                note = f"saturated even at the shortest allowed exposure ({min_us:g} us)"
                break
            new = (good[0] + bad) / 2 if good else exp / 2
        else:
            if good is None or exp > good[0]:
                good = (exp, level, sat)
            if level >= target - CLOSE_ENOUGH:
                break
            if exp >= max_us:
                note = f"still only {level:.0%} bright at the longest allowed exposure ({max_us:,.0f} us)"
                break
            new = exp * target / level if level > 0.005 else exp * 8
            if bad is not None:
                new = min(new, (exp + bad) / 2)
        new = clamp(new)
        if good and bad and bad / good[0] < 1.03:
            break
        if good and new <= good[0] * 1.005:
            break
        exp = new
    else:
        note = note or f"stopped after {max_steps} test frames"
    if good is None:
        result = {"exposure_us": float(min_us), "brightness": None, "saturated": steps[-1]["saturated"]}
        note = note or "no unsaturated exposure found"
    else:
        result = {"exposure_us": round(good[0], 1), "brightness": round(good[1], 3), "saturated": round(good[2], 6)}
    result.update(steps=steps, note=note, full_scale=full_scale, max_saturated=max_saturated,
                  target=target, min_us=min_us, max_us=max_us)
    log(f"  {name} auto exposure: using {result['exposure_us']:,.0f} us"
        + (f" (brightest {result['brightness']:.0%})" if result["brightness"] is not None else "")
        + (f" - NOTE: {note}" if note else ""))
    return result

"""
Check that the Blackfly is delivering RAW frames: no gamma, no white balance, no demosaic.

Run it with the camera plugged in (and not open in any other program), from this folder:

    python check_blackfly_raw.py              # first camera found
    python check_blackfly_raw.py 12345678     # a specific serial number

It opens the camera the same way the GUI and pipeline do (BlackflyCamera), prints the
settings read back from the camera, grabs one frame and prints its shape, bit depth and the
mean of each of the four Bayer positions. Nothing is saved.

What "raw" looks like:
    PixelFormat       BayerRG8 (a Bayer format, not RGB8/BGR8/Mono8)
    GammaEnable       False
    BalanceWhiteAuto  Off
    BalanceRatio_Red  1.0     BalanceRatio_Blue  1.0
    frame             2-D (single channel), uint8, 1080 x 1440
"""

import sys
import time

import numpy as np

from blackfly_camera import BlackflyCamera


def main():
    serial = sys.argv[1] if len(sys.argv) > 1 else None
    cam = BlackflyCamera(serial=serial)
    cam.open()
    try:
        print("Settings read back from the camera:")
        settings = cam.raw_settings()
        for k, v in settings.items():
            print(f"  {k:18s} {v}")

        cam.start()
        frame = None
        t_end = time.time() + 10
        while frame is None and time.time() < t_end:
            frame = cam.grab_preview(timeout_ms=1000)
        cam.stop()
        if frame is None:
            print("\nNo frame within 10 s (exposure too long?) - settings above are still valid.")
            return

        print(f"\nFrame: shape {frame.shape}, dtype {frame.dtype}, min {frame.min()}, max {frame.max()}")
        if frame.ndim == 2:
            sites = {"(0,0)": frame[0::2, 0::2], "(0,1)": frame[0::2, 1::2],
                     "(1,0)": frame[1::2, 0::2], "(1,1)": frame[1::2, 1::2]}
            print("Mean of each 2x2 Bayer position (for RGGB: (0,0)=R, (0,1)/(1,0)=G, (1,1)=B):")
            for k, v in sites.items():
                print(f"  {k}: {v.mean():7.2f}")
        else:
            print("WARNING: the frame has colour channels - it is NOT a raw Bayer mosaic.")

        problems = []
        if "Bayer" not in str(settings["PixelFormat"]):
            problems.append(f"PixelFormat is {settings['PixelFormat']}, not a Bayer format")
        if settings["GammaEnable"] not in (False, "n/a"):
            problems.append("gamma is ON")
        if "Off" not in str(settings["BalanceWhiteAuto"]) and settings["BalanceWhiteAuto"] != "n/a":
            problems.append(f"BalanceWhiteAuto is {settings['BalanceWhiteAuto']}")
        for c in ("Red", "Blue"):
            r = settings[f"BalanceRatio_{c}"]
            if isinstance(r, (int, float)) and abs(r - 1.0) > 1e-3:
                problems.append(f"BalanceRatio {c} is {r}, not 1.0")
        if frame.ndim != 2:
            problems.append("frame is not single-channel")
        print("\nRESULT: " + ("RAW - no gamma, no white balance, no demosaic." if not problems
                              else "NOT fully raw: " + "; ".join(problems)))
    finally:
        cam.close()


if __name__ == "__main__":
    main()

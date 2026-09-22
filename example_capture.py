"""
Example: open both cameras, capture a few frames from each, then shut down.

Run with:
    python example_capture.py

This only works on a machine that has both cameras physically connected and
both SDKs installed (see the docstrings in blackfly_camera.py and
basler_camera.py for setup steps). To exercise the code without hardware,
see test_blackfly_camera.py (mocked PySpin) and test_basler_emulated.py
(pylon's built-in camera emulator) instead.

Output format (2026-09 meeting policy):
    - Blackfly: raw Bayer mosaic, no gamma/white-balance correction, .png
    - Basler:   true 12-bit mono (Mono12 -> Mono16), .tiff
"""

from pathlib import Path

from blackfly_camera import BlackflyCamera
from basler_camera import BaslerCamera

OUTPUT_DIR = Path("captures")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    """Capture 5 frames from each camera into OUTPUT_DIR, changing exposure partway through.

    Fill in serial numbers once you know them (printed on the camera
    housing, or visible in SpinView / pylon Viewer). Leaving serial=None
    grabs the first camera of that type the SDK finds -- fine with just
    one Blackfly and one Basler plugged in, ambiguous with more than one
    of the same model.
    """
    blackfly = BlackflyCamera(serial=None, exposure_us=10000, gain_db=0)
    basler = BaslerCamera(serial=None, exposure_us=10000, gain_db=0)

    with blackfly:
        for i in range(5):
            if i == 2:
                # Exposure-time controller: change exposure mid-stream,
                # no need to stop/restart acquisition.
                blackfly.set_exposure_time(20000)
                print(f"Blackfly exposure now {blackfly.get_exposure_time()} us")
            path = blackfly.capture(OUTPUT_DIR / f"blackfly_{i:03d}.png")
            print(f"Saved {path}")

    with basler:
        for i in range(5):
            if i == 2:
                basler.set_exposure_time(20000)
                print(f"Basler exposure now {basler.get_exposure_time()} us")
            path = basler.capture(OUTPUT_DIR / f"basler_{i:03d}.tiff")
            print(f"Saved {path}")


if __name__ == "__main__":
    main()

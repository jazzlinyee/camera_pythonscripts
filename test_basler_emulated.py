"""
Smoke test for BaslerCamera using pylon's built-in Camera Emulation
transport layer -- no physical Basler camera required.

Requirements (real packages, must be installed to run this):
    pip install pypylon
    pip install opencv-python-headless   # or: pip install pillow

Run:
    python3 test_basler_emulated.py

How it works:
    Setting the PYLON_CAMEMU environment variable to a number BEFORE the
    pylon runtime initializes tells pylon to create that many virtual
    cameras. They enumerate just like real hardware (serials such as
    "0815-0000") and stream synthetic test-pattern images, so
    basler_camera.py runs against them completely unmodified -- this file
    doesn't touch BaslerCamera's code at all, it only sets the environment
    variable first and then uses the class normally.

    NOTE: basler_camera.py now requests "Mono12" from the camera (2026-09
    meeting policy: true 12-bit capture). Real Basler ace hardware supports
    this natively. Whether the *emulated* device also supports Mono12
    depends on your pylon SDK version -- if this script raises a
    CameraError about PixelFormat, that's a limitation of the emulator,
    not a sign that basler_camera.py is broken; try again once real
    hardware is connected, or check your pylon version's Camera Emulation
    docs (https://docs.baslerweb.com/camera-emulation) for its supported
    formats.
"""

import os

# Must be set BEFORE pypylon (and therefore basler_camera) is imported, so
# the pylon runtime registers the emulated devices when it initializes.
os.environ.setdefault("PYLON_CAMEMU", "1")

from pathlib import Path  # noqa: E402  (import order matters here, see above)

from basler_camera import BaslerCamera  # noqa: E402

OUTPUT_DIR = Path("captures_emulated")


def main():
    """Open one emulated Basler camera, capture 3 frames, then close it."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    cam = BaslerCamera(serial=None, exposure_us=10000, gain_db=0)
    with cam:
        print(f"Opened emulated camera: {cam.name}")
        for i in range(3):
            path = cam.capture(OUTPUT_DIR / f"emulated_{i:03d}.tiff")
            print(f"Saved {path}")

    print(
        "Done -- BaslerCamera.open/start/capture/stop/close all worked "
        "against the pylon emulator."
    )


if __name__ == "__main__":
    main()

"""
Quick viewer/inspector for the Basler camera's raw 12-bit .tiff captures.

As of 2026-09-22, basler_camera.py saves captures MSB-aligned (each
12-bit value shifted up to occupy the top of the 16-bit range, e.g. a
real reading of 188 is stored as 3008) specifically so the .tiff already
displays at roughly correct brightness in an ordinary viewer on its own
-- this script is no longer needed for routine viewing of new captures.

It's still useful for: checking exact min/max/mean numbers, viewing
older captures saved before 2026-09-22 (LSB-aligned, true value stored
directly -- these DO still look solid black in a normal viewer), or
double-checking a file you're not sure about. It auto-detects which
convention a given file uses (by whether its max value exceeds 4095) and
stretches accordingly.

Usage:
    python3 view_tiff.py test_captures/basler_003.tiff
"""

import sys

import cv2
import numpy as np


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 view_tiff.py <path_to_tiff>")
        sys.exit(1)

    path = sys.argv[1]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Could not load image: {path}")
        sys.exit(1)

    print(f"dtype: {img.dtype}, min: {img.min()}, max: {img.max()}, mean: {img.mean():.2f}")

    if img.max() <= 4095:
        scale = 4095.0
        print("Detected LSB-aligned data (pre-2026-09-22 capture, raw value stored directly).")
    else:
        scale = 65535.0
        print("Detected MSB-aligned data (2026-09-22+ capture, raw value stored *16).")

    stretched = np.clip(img, 0, scale) / scale * 255
    stretched = stretched.astype(np.uint8)

    cv2.imshow(f"{path} (stretched for viewing)", stretched)
    print("Press any key on the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

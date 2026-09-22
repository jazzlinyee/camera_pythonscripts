"""
Quick viewer for the Basler camera's raw 12-bit .tiff captures.

basler_camera.py saves true 12-bit sensor data (0-4095) inside a 16-bit
TIFF file. Most normal image viewers (e.g. macOS Preview) display a
16-bit TIFF by treating the *full* 16-bit range (0-65535) as black-to-
white -- since the real data only ever fills the bottom ~6% of that
range, a perfectly well-exposed capture can look solid black when opened
normally, even though the actual pixel data is fine.

This script loads a .tiff, prints its actual min/max/mean, and stretches
the true 12-bit range to 0-255 so you can see what was really captured.

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

    stretched = np.clip(img, 0, 4095) / 4095.0 * 255
    stretched = stretched.astype(np.uint8)

    cv2.imshow(f"{path} (stretched to true 12-bit range)", stretched)
    print("Press any key on the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

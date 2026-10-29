"""
Turns a captured image or an SLM mask into a small Tkinter-showable
PhotoImage, for the live preview panels in gui_test_camera_slm.py.

Handles the three kinds of array this project produces:
    - Blackfly captures: 8-bit BGR color (as saved to .png)
    - Basler captures: 16-bit mono, MSB-aligned since 2026-09-22 (as
      saved to .tiff) -- auto-detects older LSB-aligned captures too,
      same convention view_tiff.py checks for
    - SLM masks: 8-bit grayscale numpy arrays, already 0-255

No Pillow dependency -- resizes with cv2.resize() and hands Tkinter raw
PGM/PPM (netpbm) bytes, which tkinter.PhotoImage loads natively. Same
trick slm_display.py uses for masks, extended here to cover color images
and thumbnail-sized resizing.
"""

import cv2
import numpy as np


def to_display_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert any of this project's image arrays to uint8, ready to
    preview. BGR/grayscale 8-bit pass through unchanged; 16-bit mono
    gets the same auto-detected LSB-vs-MSB stretch as view_tiff.py."""
    if arr.dtype == np.uint8:
        return arr

    if arr.dtype == np.uint16:
        if arr.max() <= 4095:
            scale = 4095.0  # LSB-aligned (pre-2026-09-22 Basler capture)
        else:
            scale = 65535.0  # MSB-aligned (2026-09-22+ Basler capture)
        stretched = np.clip(arr, 0, scale) / scale * 255
        return stretched.astype(np.uint8)

    # Fallback for anything else: normalize whatever range it actually has.
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255
    return arr.astype(np.uint8)


def _resize_to_fit(arr: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Shrink (never enlarge) arr to fit within max_w x max_h, preserving
    aspect ratio."""
    h, w = arr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    if (new_w, new_h) == (w, h):
        return arr
    return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _to_netpbm_bytes(arr: np.ndarray) -> bytes:
    """Encode a uint8 array (2D grayscale or 3-channel BGR) as binary
    PGM/PPM bytes -- tkinter.PhotoImage loads these natively."""
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.ndim == 2:
        h, w = arr.shape
        header = f"P5\n{w} {h}\n255\n".encode("ascii")
        return header + arr.tobytes()
    elif arr.ndim == 3 and arr.shape[2] == 3:
        h, w = arr.shape[:2]
        rgb = arr[:, :, ::-1]  # BGR (opencv convention) -> RGB (netpbm)
        header = f"P6\n{w} {h}\n255\n".encode("ascii")
        return header + rgb.tobytes()
    else:
        raise ValueError(f"Unsupported array shape for preview: {arr.shape}")


def array_to_photoimage(arr: np.ndarray, max_w: int, max_h: int):
    """Convert an image array into a tkinter.PhotoImage scaled to fit
    within (max_w, max_h), preserving aspect ratio and never upscaling.

    Caller must keep a reference to the returned PhotoImage alive for as
    long as it's displayed (Tkinter doesn't hold its own reference) --
    e.g. `label.image = photo` alongside `label.configure(image=photo)`.
    """
    import tkinter as tk  # lazy import: keeps this module importable headless

    display = to_display_uint8(arr)
    display = _resize_to_fit(display, max_w, max_h)
    return tk.PhotoImage(data=_to_netpbm_bytes(display))

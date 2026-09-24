"""
GUI control panel for interactive camera + SLM testing on real hardware.

This is the point-and-click sibling of test_on_hardware_camera_slm.py --
same underlying camera and SLM control, but as buttons/fields instead of
a typed command prompt, with the current mask and the last capture shown
live in the window instead of only saved to disk.

Run:
    python3 gui_test_camera_slm.py [--mirrored] [masks_folder]

If masks_folder is given, loads real masks from it (see
load_masks_from_folder()'s docstring in slm_display.py for naming/sizing
rules). If omitted, falls back to the same generated test patterns as
test_on_hardware_slm.py (checkerboard/white/black/stripes).

Pass --mirrored if the SLM is set to mirror your laptop screen rather
than extend the desktop (see slm_display.py's SLMDisplay docstring).

Layout:
    - Camera panel: pick Blackfly/Basler, optional serial, open/close,
      set/read exposure, capture.
    - SLM panel: open/close the display, buttons for each mask.
    - Current mask preview and last-capture preview, side by side.
    - Log panel at the bottom (replaces the old REPL's printed lines).

Only one tkinter root exists in this process (this window) -- the SLM's
own window is created as a Toplevel of it via SLMDisplay(..., master=...),
not a second independent Tk() (see slm_display.py's SLMDisplay
docstring for why that matters).
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from basler_camera import BaslerCamera
from blackfly_camera import BlackflyCamera
from camera_base import CameraError
from preview_utils import array_to_photoimage
from slm_display import SLMDisplay, SLMError, load_masks_from_folder
from test_on_hardware_slm import make_test_masks

OUTPUT_DIR = Path("test_captures")
THUMB_W, THUMB_H = 340, 230  # bumped up 2026-09-22 so the bigger window has room to use
DEFAULT_MASK_NAMES = ["Checkerboard", "White", "Black", "Stripes"]


class App:
    def __init__(self, root: tk.Tk, masks, mask_names, mirrored: bool):
        self.root = root
        self.masks = masks
        self.mask_names = mask_names
        self.mirrored = mirrored

        self.cam = None
        self.cam_label = None
        self.cam_ext = None
        self.shot = 0

        self.slm = None

        self._mask_thumb_photo = None
        self._capture_thumb_photo = None

        self._auto_running = False
        self._auto_after_id = None
        self._auto_index = 0
        self._auto_interval_s = None
        self._auto_settle_s = None

        root.title("Camera + SLM control panel")
        self._build_widgets()
        self.log(
            f"{len(masks)} mask(s) loaded"
            + (" from folder." if mask_names[0].startswith("Mask") else " (generated test patterns).")
        )
        if mirrored:
            self.log("--mirrored passed -- SLM open will target the primary monitor.")

    # ---------------------------------------------------------- layout --

    def _build_widgets(self):
        pad = {"padx": 6, "pady": 4}

        cam_frame = ttk.LabelFrame(self.root, text="Camera")
        cam_frame.grid(row=0, column=0, sticky="nsew", **pad)

        self.cam_choice = tk.StringVar(value="blackfly")
        ttk.Radiobutton(
            cam_frame, text="Blackfly S", variable=self.cam_choice, value="blackfly"
        ).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            cam_frame, text="Basler ace", variable=self.cam_choice, value="basler"
        ).grid(row=1, column=0, sticky="w")

        ttk.Label(cam_frame, text="Serial (blank = auto):").grid(row=0, column=1, sticky="e")
        self.serial_entry = ttk.Entry(cam_frame, width=16)
        self.serial_entry.grid(row=0, column=2, sticky="w")

        self.open_cam_btn = ttk.Button(cam_frame, text="Open Camera", command=self.open_camera)
        self.open_cam_btn.grid(row=1, column=1, **pad)
        self.close_cam_btn = ttk.Button(
            cam_frame, text="Close Camera", command=self.close_camera, state="disabled"
        )
        self.close_cam_btn.grid(row=1, column=2, **pad)

        ttk.Label(cam_frame, text="Exposure (us):").grid(row=2, column=0, sticky="e")
        self.exposure_entry = ttk.Entry(cam_frame, width=10)
        self.exposure_entry.grid(row=2, column=1, sticky="w")
        self.set_exposure_btn = ttk.Button(
            cam_frame, text="Set Exposure", command=self.set_exposure, state="disabled"
        )
        self.set_exposure_btn.grid(row=2, column=2, sticky="w")

        self.exposure_label = ttk.Label(cam_frame, text="Current exposure: --")
        self.exposure_label.grid(row=3, column=0, columnspan=3, sticky="w", **pad)

        self.capture_btn = ttk.Button(
            cam_frame, text="Capture", command=self.capture, state="disabled"
        )
        self.capture_btn.grid(row=4, column=0, columnspan=3, pady=10)

        slm_frame = ttk.LabelFrame(self.root, text="SLM")
        slm_frame.grid(row=0, column=1, sticky="nsew", **pad)

        self.open_slm_btn = ttk.Button(slm_frame, text="Open SLM", command=self.open_slm)
        self.open_slm_btn.grid(row=0, column=0, **pad)
        self.close_slm_btn = ttk.Button(
            slm_frame, text="Close SLM", command=self.close_slm, state="disabled"
        )
        self.close_slm_btn.grid(row=0, column=1, **pad)

        self.mask_buttons = []
        for i, name in enumerate(self.mask_names):
            b = ttk.Button(
                slm_frame, text=name, command=lambda i=i: self.show_mask(i), state="disabled"
            )
            b.grid(row=1 + i // 2, column=i % 2, sticky="ew", **pad)
            self.mask_buttons.append(b)

        auto_frame = ttk.LabelFrame(self.root, text="Auto Cycle")
        auto_frame.grid(row=0, column=2, sticky="nsew", **pad)

        ttk.Label(auto_frame, text="Switch every (s):").grid(row=0, column=0, sticky="e")
        self.auto_interval_entry = ttk.Entry(auto_frame, width=6)
        self.auto_interval_entry.insert(0, "5")
        self.auto_interval_entry.grid(row=0, column=1, sticky="w")

        ttk.Label(auto_frame, text="Settle before capture (s):").grid(row=1, column=0, sticky="e")
        self.auto_settle_entry = ttk.Entry(auto_frame, width=6)
        self.auto_settle_entry.insert(0, "2")
        self.auto_settle_entry.grid(row=1, column=1, sticky="w")

        self.start_auto_btn = ttk.Button(
            auto_frame, text="Start Auto Cycle", command=self.start_auto_cycle
        )
        self.start_auto_btn.grid(row=2, column=0, columnspan=2, sticky="ew", **pad)
        self.stop_auto_btn = ttk.Button(
            auto_frame, text="Stop Auto Cycle", command=self.stop_auto_cycle, state="disabled"
        )
        self.stop_auto_btn.grid(row=3, column=0, columnspan=2, sticky="ew", **pad)

        self.auto_status_label = ttk.Label(auto_frame, text="Not running")
        self.auto_status_label.grid(row=4, column=0, columnspan=2, sticky="w", **pad)

        ttk.Label(
            auto_frame,
            text="Needs camera + SLM\nboth open first.",
            justify="left",
            foreground="gray40",
        ).grid(row=5, column=0, columnspan=2, sticky="w", **pad)

        preview_frame = ttk.Frame(self.root)
        preview_frame.grid(row=1, column=0, columnspan=3, sticky="nsew", **pad)

        mask_prev_frame = ttk.LabelFrame(preview_frame, text="Current mask")
        mask_prev_frame.grid(row=0, column=0, **pad)
        # NOTE: a tk.Label's width/height are measured in *characters and
        # text lines*, not pixels, until an actual image is assigned to
        # it -- setting width=220, height=150 here meant "220 characters
        # by 150 text lines" (enormous) until the first mask/capture was
        # shown, which was blowing up the whole window's layout. Give it
        # a real blank placeholder image up front instead, so it's
        # always in pixel-sized "image mode" from the start.
        blank = np.zeros((THUMB_H, THUMB_W), dtype=np.uint8)
        self._blank_photo = array_to_photoimage(blank, THUMB_W, THUMB_H)

        self.mask_preview_label = tk.Label(
            mask_prev_frame, image=self._blank_photo, background="gray20"
        )
        self.mask_preview_label.pack()

        cap_prev_frame = ttk.LabelFrame(preview_frame, text="Last capture")
        cap_prev_frame.grid(row=0, column=1, **pad)
        self.capture_preview_label = tk.Label(
            cap_prev_frame, image=self._blank_photo, background="gray20"
        )
        self.capture_preview_label.pack()
        self.capture_path_label = ttk.Label(cap_prev_frame, text="No capture yet")
        self.capture_path_label.pack(anchor="w")

        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.grid(row=2, column=0, columnspan=3, sticky="nsew", **pad)
        self.log_text = ScrolledText(log_frame, width=80, height=6, state="disabled")
        self.log_text.pack(fill="both", expand=True)

        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    # --------------------------------------------------- camera actions --

    def open_camera(self):
        choice = self.cam_choice.get()
        if self.cam is not None:
            if self.cam_label == choice:
                self.log(f"{choice} camera is already open.")
                return
            self.log(f"Switching from {self.cam_label} to {choice} -- closing current camera first.")
            self._shutdown_camera()

        serial = self.serial_entry.get().strip() or None
        try:
            if choice == "blackfly":
                self.cam = BlackflyCamera(serial=serial)
                self.cam_label, self.cam_ext = "blackfly", ".png"
            else:
                self.cam = BaslerCamera(serial=serial)
                self.cam_label, self.cam_ext = "basler", ".tiff"
            self.cam.open()
            self.cam.start()
        except CameraError as e:
            self.log(f"Failed to open/start {choice} camera: {e}")
            self.cam = None
            return

        self.log(f"Opened {self.cam_label} camera.")
        self.open_cam_btn.configure(state="disabled")
        self.close_cam_btn.configure(state="normal")
        self.set_exposure_btn.configure(state="normal")
        self.capture_btn.configure(state="normal")
        self.refresh_exposure()

    def close_camera(self):
        if self.cam is None:
            return
        if self._auto_running:
            self.log("Stopping auto cycle -- camera is closing.")
            self.stop_auto_cycle()
        self._shutdown_camera()
        self.log("Camera closed.")

    def _shutdown_camera(self):
        """Stop/close whatever camera is currently open and reset the
        camera controls to their closed state, without logging "closed"
        (the caller logs whatever's contextually appropriate -- a plain
        close, or a switch to a different camera)."""
        try:
            self.cam.stop()
            self.cam.close()
        except CameraError as e:
            self.log(f"Error while closing camera: {e}")
        self.cam = None
        self.open_cam_btn.configure(state="normal")
        self.close_cam_btn.configure(state="disabled")
        self.set_exposure_btn.configure(state="disabled")
        self.capture_btn.configure(state="disabled")
        self.exposure_label.configure(text="Current exposure: --")

    def set_exposure(self):
        if self.cam is None:
            return
        raw = self.exposure_entry.get().strip()
        try:
            value = float(raw)
            self.cam.set_exposure_time(value)
            self.log(f"Exposure set to {value} us")
            self._flush_stale_frame()
        except ValueError:
            self.log(f"'{raw}' is not a valid exposure value (expected microseconds, e.g. 15000).")
        except CameraError as e:
            self.log(f"Could not set exposure: {e}")
        self.refresh_exposure()

    def _flush_stale_frame(self):
        """Discard exactly one grabbed frame right after an exposure change.

        set_exposure_time() changes the register instantly, but the sensor
        may already have a frame mid-exposure (or freshly landed in the
        transport buffer) under the OLD setting. StreamBufferHandlingMode
        = NewestOnly (see blackfly_camera.py's _configure()) only
        guarantees capture() returns the newest *arrived* frame -- not one
        that reflects a setting changed moments ago. Grabbing and
        discarding one throwaway frame here means the very next real
        Capture click is guaranteed to reflect the new exposure. This
        calls only the existing public capture() API -- no changes to
        blackfly_camera.py/basler_camera.py.
        """
        try:
            OUTPUT_DIR.mkdir(exist_ok=True)
            flush_path = OUTPUT_DIR / f"_exposure_flush{self.cam_ext}"
            self.cam.capture(flush_path)
            flush_path.unlink(missing_ok=True)
        except CameraError as e:
            self.log(f"(non-fatal) couldn't flush stale frame after exposure change: {e}")

    def refresh_exposure(self):
        if self.cam is None:
            return
        try:
            value = self.cam.get_exposure_time()
            self.exposure_label.configure(text=f"Current exposure: {value} us")
        except CameraError as e:
            self.log(f"Could not read exposure: {e}")

    def capture(self):
        if self.cam is None:
            return
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / f"{self.cam_label}_{self.shot:03d}{self.cam_ext}"
        try:
            saved = self.cam.capture(path)
        except CameraError as e:
            self.log(f"Capture failed: {e}")
            return

        self.shot += 1
        self.log(f"Saved {saved}")
        self.capture_path_label.configure(text=str(saved))

        img = cv2.imread(str(saved), cv2.IMREAD_UNCHANGED)
        if img is None:
            self.log(f"Saved {saved}, but couldn't reload it here for the preview.")
            return
        self._capture_thumb_photo = array_to_photoimage(img, THUMB_W, THUMB_H)
        self.capture_preview_label.configure(image=self._capture_thumb_photo)

    # ------------------------------------------------------- SLM actions --

    def open_slm(self):
        if self.slm is not None:
            self.log("SLM already open.")
            return
        try:
            self.slm = SLMDisplay(self.masks, mirrored=self.mirrored, master=self.root)
            self.slm.open()
        except SLMError as e:
            self.log(f"Failed to open SLM display: {e}")
            self.slm = None
            return

        self.log("SLM display opened -- check it landed on the SLM's screen.")
        self.open_slm_btn.configure(state="disabled")
        self.close_slm_btn.configure(state="normal")
        for b in self.mask_buttons:
            b.configure(state="normal")
        self.show_mask(0)

    def close_slm(self):
        if self.slm is None:
            return
        if self._auto_running:
            self.log("Stopping auto cycle -- SLM is closing.")
            self.stop_auto_cycle()
        self.slm.close()
        self.slm = None
        self.log("SLM display closed.")
        self.open_slm_btn.configure(state="normal")
        self.close_slm_btn.configure(state="disabled")
        for b in self.mask_buttons:
            b.configure(state="disabled")

    def show_mask(self, index: int):
        if self.slm is None:
            return
        try:
            self.slm.show_mask(index)
        except SLMError as e:
            self.log(f"Could not show mask {index}: {e}")
            return

        name = self.mask_names[index] if index < len(self.mask_names) else f"Mask {index}"
        self.log(f"Showing mask {index} ({name}).")
        self._mask_thumb_photo = array_to_photoimage(self.masks[index], THUMB_W, THUMB_H)
        self.mask_preview_label.configure(image=self._mask_thumb_photo)

        # Same stale-frame race as set_exposure() (see _flush_stale_frame's
        # docstring): if a camera is open, the sensor may already have a
        # frame in flight that started exposing under the OLD mask when we
        # switch to a new one. Discard one throwaway grab so the next real
        # Capture reliably reflects the mask now on screen, rather than a
        # blend of old and new.
        if self.cam is not None:
            self._flush_stale_frame()

    # --------------------------------------------------- auto cycle -----

    def start_auto_cycle(self):
        """Every `interval` seconds: switch to the next mask, wait
        `settle` seconds for the LC panel to physically settle (see
        hardware-notes.md's 2026-09-22 finding -- switching masks isn't
        instantaneous, especially transitioning towards black), then
        capture. Uses tk's own `after()` scheduler rather than a blocking
        loop, so the rest of the GUI stays responsive (Stop button, manual
        controls) while this runs."""
        if self._auto_running:
            return
        if self.cam is None or self.slm is None:
            self.log("Auto cycle needs both a camera and the SLM open first.")
            return
        try:
            interval_s = float(self.auto_interval_entry.get().strip())
            settle_s = float(self.auto_settle_entry.get().strip())
        except ValueError:
            self.log("Interval and settle time must be numbers, in seconds.")
            return
        if interval_s <= settle_s:
            self.log(
                f"Interval ({interval_s:g}s) must be longer than the settle time ({settle_s:g}s)."
            )
            return

        self._auto_running = True
        self._auto_interval_s = interval_s
        self._auto_settle_s = settle_s
        self.start_auto_btn.configure(state="disabled")
        self.stop_auto_btn.configure(state="normal")
        for b in self.mask_buttons:
            b.configure(state="disabled")
        self.capture_btn.configure(state="disabled")
        self.auto_status_label.configure(text=f"Running: every {interval_s:g}s")
        self.log(
            f"Auto cycle started: switch masks every {interval_s:g}s, "
            f"capture {settle_s:g}s after each switch."
        )
        self._auto_cycle_tick()

    def stop_auto_cycle(self):
        if not self._auto_running:
            return
        self._auto_running = False
        if self._auto_after_id is not None:
            self.root.after_cancel(self._auto_after_id)
            self._auto_after_id = None
        self.start_auto_btn.configure(state="normal")
        self.stop_auto_btn.configure(state="disabled")
        if self.slm is not None:
            for b in self.mask_buttons:
                b.configure(state="normal")
        if self.cam is not None:
            self.capture_btn.configure(state="normal")
        self.auto_status_label.configure(text="Not running")
        self.log("Auto cycle stopped.")

    def _auto_cycle_tick(self):
        if not self._auto_running:
            return
        self._auto_index = (self._auto_index + 1) % len(self.masks)
        self.show_mask(self._auto_index)  # also flushes one stale camera frame
        settle_ms = int(self._auto_settle_s * 1000)
        self._auto_after_id = self.root.after(settle_ms, self._auto_capture_then_reschedule)

    def _auto_capture_then_reschedule(self):
        if not self._auto_running:
            return
        self.capture()
        remaining_ms = max(0, int((self._auto_interval_s - self._auto_settle_s) * 1000))
        self._auto_after_id = self.root.after(remaining_ms, self._auto_cycle_tick)

    # ------------------------------------------------------------- exit --

    def on_close(self):
        if self._auto_running:
            self.stop_auto_cycle()
        if self.slm is not None:
            self.slm.close()
        if self.cam is not None:
            self._shutdown_camera()
        self.root.destroy()


def main():
    args = sys.argv[1:]
    mirrored = "--mirrored" in args
    args = [a for a in args if a != "--mirrored"]

    if args:
        masks = load_masks_from_folder(args[0])
        mask_names = [f"Mask {i}" for i in range(len(masks))]
    else:
        masks = make_test_masks()
        mask_names = DEFAULT_MASK_NAMES[: len(masks)]

    root = tk.Tk()
    app = App(root, masks, mask_names, mirrored)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.update_idletasks()
    # Size the starting window relative to the actual screen instead of a
    # fixed pixel cap -- a fixed cap (760x620, tried 2026-09-22) looks tiny
    # on a large display but huge on a small one. Target a solid, usable
    # fraction of the screen; never smaller than the content actually
    # needs, never bigger than the screen itself. The window stays freely
    # resizable by hand from here either way.
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    req_w, req_h = root.winfo_reqwidth(), root.winfo_reqheight()
    target_w = min(max(req_w, int(screen_w * 0.55)), screen_w - 100)
    target_h = min(max(req_h, int(screen_h * 0.70)), screen_h - 100)
    root.geometry(f"{target_w}x{target_h}")
    root.mainloop()


if __name__ == "__main__":
    main()

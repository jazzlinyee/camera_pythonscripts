"""
GUI control panel for interactive camera + SLM testing on real hardware.

This is the point-and-click sibling of test_on_hardware_camera_slm.py --
same underlying camera and SLM control, but as buttons/fields instead of
a typed command prompt, with the current mask and the last capture shown
live in the window instead of only saved to disk.

Run:
    python3 gui_test_camera_slm.py [--mirrored] [masks_folder]

If masks_folder is given, it's registered as the first mask folder (see
"Mask folders" below). If omitted, a small folder of the built-in test
patterns (checkerboard/white/black/stripes) is created at ./test_patterns
(if it doesn't already exist) and registered instead -- so even the
built-in patterns live on disk like any other mask folder, rather than
being a special in-memory case.

Pass --mirrored if the SLM is set to mirror your laptop screen rather
than extend the desktop (see slm_display.py's SLMDisplay docstring).

Mask folders (changed 2026-09-26 -- masks are managed on disk, not in
this app):
    Masks are organized entirely as real folders on your computer, one
    folder per "set". "Add Folder..." registers a folder (named after
    itself); "Remove Folder" un-registers one (never touches the actual
    files); "Refresh" re-reads every registered folder's current
    contents from disk. There's no "add one mask" or "delete one mask"
    inside the app anymore -- to reorganize, rename, add, or delete
    individual masks, just do it in Finder (drag a file from one
    registered folder to another, delete it, whatever), then hit
    Refresh. This is also why masks are named after their filename with
    no in-app naming step: the filename *is* the name, so renaming a
    file in Finder is how you rename a mask.

    The list of registered folders is remembered across runs in
    mask_folders.json (next to this script) -- added/removed folders are
    saved there automatically, and restored on the next launch. A folder
    that's been moved or deleted since is skipped (and dropped from the
    remembered list) rather than erroring.

Layout:
    - Camera panel: pick Blackfly/Basler, optional serial, open/close,
      set/read exposure, capture. Blackfly and Basler are independent
      slots -- both can be open at once, which is what Capture Both
      needs (a back-to-back, non-hardware-synced shot from each).
    - SLM panel: open/close the display; "Add Folder...", "Remove
      Folder", "Refresh" manage which folders' masks are loaded (see
      above). Masks show in a tree, grouped by folder (click the arrow
      to expand/collapse one); double-click a mask to show it on the
      SLM -- added/refreshed live on an already-open display via
      SLMDisplay.set_masks().
    - Auto Cycle panel: automatically switch masks and capture on a
      timer, with a separate settle delay before each capture (see
      slm_display.py -- LC panels don't switch instantly). Interval and
      settle time are in milliseconds. "Loop through" picks which mask
      folder to cycle -- one specific folder, or "All masks" across
      every registered folder. Re-scans folders fresh each time you hit
      Start, so a Finder reorganization just before starting is picked
      up automatically. Optionally "Vary exposure each switch" steps
      exposure through a start/step/end sweep in lockstep with the mask
      switches (wrapping independently, so mask count and exposure step
      count don't need to match) -- a single-mask loop with this on is a
      plain exposure bracket. "Load Exposure List..." replaces that
      sweep with an explicit list read from a text file (one
      microsecond value per line; blank lines and '#' comments are
      ignored) -- useful when you want specific, hand-picked exposure
      values per step rather than an evenly-spaced sweep. "Clear"
      drops the loaded list and goes back to start/step/end.
    - Current mask preview, plus separate Blackfly/Basler capture
      previews.
    - Log panel at the bottom (replaces the old REPL's printed lines).

Only one tkinter root exists in this process (this window) -- the SLM's
own window is created as a Toplevel of it via SLMDisplay(..., master=...),
not a second independent Tk() (see slm_display.py's SLMDisplay
docstring for why that matters).
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from basler_camera import BaslerCamera
from blackfly_camera import BlackflyCamera
from camera_base import CameraError
from preview_utils import array_to_photoimage
from slm_display import SLMDisplay, SLMError
from test_on_hardware_slm import HEIGHT, WIDTH, make_test_masks

OUTPUT_DIR = Path("test_captures")
FOLDER_CONFIG_PATH = Path("mask_folders.json")  # remembers registered mask folders across runs
THUMB_W, THUMB_H = 340, 230  # bumped up 2026-09-22 so the bigger window has room to use
DEFAULT_MASK_NAMES = ["checkerboard", "white", "black", "stripes"]
MASK_EXPECTED_SIZE = (WIDTH, HEIGHT)  # the SLM panel's native resolution
MASK_IMAGE_GLOBS = ("*.png", "*.tif", "*.tiff", "*.bmp", "*.jpg", "*.jpeg")

# (class, file extension) for each selectable camera -- added 2026-09-24
# alongside dual/simultaneous-capture support.
CAM_CLASSES = {"blackfly": (BlackflyCamera, ".png"), "basler": (BaslerCamera, ".tiff")}

# Auto Cycle's special "Loop through" selection meaning "ignore which
# folder a mask came from, loop over everything registered."
ALL_MASKS_LABEL = "All masks"


def ensure_test_pattern_folder() -> Path:
    """Create ./test_patterns with the built-in checkerboard/white/black/
    stripes patterns, if it doesn't already exist (never overwrites a
    file that's already there, in case someone edited one). This is what
    gets registered as the default mask folder when no folder is passed
    on the command line -- so even the built-in patterns live on disk
    like any other mask set, editable/rearrangeable in Finder the same
    way."""
    folder = Path("test_patterns")
    folder.mkdir(exist_ok=True)
    for name, mask in zip(DEFAULT_MASK_NAMES, make_test_masks()):
        path = folder / f"{name}.png"
        if not path.exists():
            cv2.imwrite(str(path), mask)
    return folder


class App:
    def __init__(self, root: tk.Tk, mirrored: bool):
        self.root = root
        self.mirrored = mirrored

        # Registered mask folders: list of (Path, label) tuples. label is
        # normally the folder's own name; disambiguated to the full path
        # if two registered folders happen to share a basename. This is
        # the only thing the app itself owns about masks -- the actual
        # mask images always come from a fresh scan of these folders (see
        # refresh_masks()), never from an in-app add/delete list, so
        # renaming/adding/deleting/moving files in Finder is the way to
        # edit them.
        self.mask_root_folders = []
        self.masks = []
        self.mask_names = []
        self.mask_folders = []  # parallel to masks/mask_names: which folder label each came from

        # Two independent camera slots (not just "the current one") so
        # both can be open at once -- needed for Capture Both. Which one
        # Set Exposure/Capture/Open/Close act on is whichever radio
        # button is currently selected; opening one no longer auto-closes
        # the other (that was fine when only one camera was ever used at
        # a time, but not once dual capture is a thing).
        self.cams = {"blackfly": None, "basler": None}
        self.shots = {"blackfly": 0, "basler": 0}

        self.slm = None

        self._mask_thumb_photo = None
        self._blackfly_thumb_photo = None
        self._basler_thumb_photo = None

        self._auto_running = False
        self._auto_after_id = None
        self._auto_index = 0
        self._auto_interval_ms = None
        self._auto_settle_ms = None
        self._auto_exp_index = 0
        self._auto_exp_values = None  # None = exposure variation off for this run
        self._auto_exp_file_values = None  # loaded from a text file -- takes
        self._auto_exp_file_name = None    # precedence over start/step/end if set

        root.title("Camera + SLM control panel")
        self._build_widgets()
        if mirrored:
            self.log("--mirrored passed -- SLM open will target the primary monitor.")

    # ---------------------------------------------------------- layout --

    def _build_widgets(self):
        pad = {"padx": 6, "pady": 4}

        cam_frame = ttk.LabelFrame(self.root, text="Camera")
        cam_frame.grid(row=0, column=0, sticky="nsew", **pad)

        self.cam_choice = tk.StringVar(value="blackfly")
        # Set/Capture/Open/Close act on whichever radio is selected --
        # refresh their enabled state whenever the selection changes, even
        # though nothing is being opened/closed by the switch itself.
        self.cam_choice.trace_add("write", lambda *_args: self._refresh_camera_buttons())
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
        self.capture_btn.grid(row=4, column=0, columnspan=3, pady=(10, 0))

        self.capture_both_btn = ttk.Button(
            cam_frame,
            text="Capture Both",
            command=self.capture_both,
            state="disabled",
        )
        self.capture_both_btn.grid(row=5, column=0, columnspan=3, pady=(0, 4))
        ttk.Label(
            cam_frame,
            text="(back-to-back, not hardware-synced)",
            foreground="gray40",
        ).grid(row=6, column=0, columnspan=3, pady=(0, 10))

        slm_frame = ttk.LabelFrame(self.root, text="SLM")
        slm_frame.grid(row=0, column=1, sticky="nsew", **pad)
        self.slm_frame = slm_frame

        self.open_slm_btn = ttk.Button(slm_frame, text="Open SLM", command=self.open_slm)
        self.open_slm_btn.grid(row=0, column=0, **pad)
        self.close_slm_btn = ttk.Button(
            slm_frame, text="Close SLM", command=self.close_slm, state="disabled"
        )
        self.close_slm_btn.grid(row=0, column=1, **pad)
        self.add_folder_btn = ttk.Button(
            slm_frame, text="Add Folder...", command=self.add_folder_dialog
        )
        self.add_folder_btn.grid(row=0, column=2, **pad)
        self.remove_folder_btn = ttk.Button(
            slm_frame, text="Remove Folder", command=self.remove_selected_folder
        )
        self.remove_folder_btn.grid(row=0, column=3, **pad)
        self.refresh_btn = ttk.Button(slm_frame, text="Refresh", command=self.refresh_masks)
        self.refresh_btn.grid(row=0, column=4, **pad)

        # Two-level tree (folder > individual mask), read-only display of
        # whatever the last refresh_masks() found on disk. Folders
        # collapse/expand (click the twisty arrow); double-click a mask
        # to show it; select one or more folder rows (ctrl/shift-click)
        # and hit Remove Folder to un-register them (files on disk are
        # never touched from here -- see the module docstring).
        tree_frame = ttk.Frame(slm_frame)
        tree_frame.grid(row=1, column=0, columnspan=5, sticky="nsew", padx=6, pady=4)
        self.mask_tree = ttk.Treeview(
            tree_frame, show="tree", selectmode="extended", height=8
        )
        self.mask_tree.column("#0", width=240)
        self.mask_tree.pack(side="left", fill="both", expand=True)
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.mask_tree.yview)
        tree_scroll.pack(side="right", fill="y")
        self.mask_tree.configure(yscrollcommand=tree_scroll.set)
        self.mask_tree.bind("<Double-Button-1>", self._on_mask_tree_double_click)

        auto_frame = ttk.LabelFrame(self.root, text="Auto Cycle")
        auto_frame.grid(row=0, column=2, sticky="nsew", **pad)

        ttk.Label(auto_frame, text="Switch every (ms):").grid(row=0, column=0, sticky="e")
        self.auto_interval_entry = ttk.Entry(auto_frame, width=8)
        self.auto_interval_entry.insert(0, "5000")
        self.auto_interval_entry.grid(row=0, column=1, sticky="w")

        ttk.Label(auto_frame, text="Settle before capture (ms):").grid(row=1, column=0, sticky="e")
        self.auto_settle_entry = ttk.Entry(auto_frame, width=8)
        self.auto_settle_entry.insert(0, "2000")
        self.auto_settle_entry.grid(row=1, column=1, sticky="w")

        ttk.Label(auto_frame, text="Loop through:").grid(row=2, column=0, sticky="e")
        self.auto_set_var = tk.StringVar(value=ALL_MASKS_LABEL)
        self.auto_set_combo = ttk.Combobox(
            auto_frame, textvariable=self.auto_set_var, state="readonly", width=14
        )
        self.auto_set_combo.grid(row=2, column=1, sticky="w")

        self.auto_both_var = tk.BooleanVar(value=False)
        self.auto_both_check = ttk.Checkbutton(
            auto_frame, text="Capture both cameras", variable=self.auto_both_var
        )
        self.auto_both_check.grid(row=3, column=0, columnspan=2, sticky="w", **pad)

        # Exposure variation -- optional, off by default. When on, every
        # switch also steps the exposure through a start/step/end sweep
        # (wrapping back to start once past end, same as the mask loop
        # wraps), applied to whichever camera(s) Capture would use --
        # the selected one, or both if "Capture both cameras" is checked.
        # A single-mask loop with this on is a plain exposure bracket:
        # same mask, stepping exposure each capture.
        self.auto_exp_var = tk.BooleanVar(value=False)
        self.auto_exp_check = ttk.Checkbutton(
            auto_frame, text="Vary exposure each switch", variable=self.auto_exp_var
        )
        self.auto_exp_check.grid(row=4, column=0, columnspan=2, sticky="w", **pad)

        ttk.Label(auto_frame, text="Exposure start (us):").grid(row=5, column=0, sticky="e")
        self.auto_exp_start_entry = ttk.Entry(auto_frame, width=8)
        self.auto_exp_start_entry.grid(row=5, column=1, sticky="w")

        ttk.Label(auto_frame, text="Exposure step (us):").grid(row=6, column=0, sticky="e")
        self.auto_exp_step_entry = ttk.Entry(auto_frame, width=8)
        self.auto_exp_step_entry.grid(row=6, column=1, sticky="w")

        ttk.Label(auto_frame, text="Exposure end (us):").grid(row=7, column=0, sticky="e")
        self.auto_exp_end_entry = ttk.Entry(auto_frame, width=8)
        self.auto_exp_end_entry.grid(row=7, column=1, sticky="w")

        # Alternative to the start/step/end sweep above: an explicit,
        # hand-authored list of exposure values (one microsecond value per
        # line in a text file). When a list is loaded it takes precedence
        # over start/step/end -- see _auto_exposure_values().
        exp_file_btns = ttk.Frame(auto_frame)
        exp_file_btns.grid(row=8, column=0, columnspan=2, sticky="ew", **pad)
        self.load_exp_file_btn = ttk.Button(
            exp_file_btns, text="Load Exposure List...", command=self.load_exposure_list_dialog
        )
        self.load_exp_file_btn.pack(side="left", fill="x", expand=True)
        self.clear_exp_file_btn = ttk.Button(
            exp_file_btns, text="Clear", command=self.clear_exposure_list, state="disabled"
        )
        self.clear_exp_file_btn.pack(side="left", padx=(4, 0))
        self.auto_exp_file_label = ttk.Label(
            auto_frame, text="(no exposure list loaded)", foreground="gray40", wraplength=220, justify="left"
        )
        self.auto_exp_file_label.grid(row=9, column=0, columnspan=2, sticky="w", **pad)

        self.start_auto_btn = ttk.Button(
            auto_frame, text="Start Auto Cycle", command=self.start_auto_cycle
        )
        self.start_auto_btn.grid(row=10, column=0, columnspan=2, sticky="ew", **pad)
        self.stop_auto_btn = ttk.Button(
            auto_frame, text="Stop Auto Cycle", command=self.stop_auto_cycle, state="disabled"
        )
        self.stop_auto_btn.grid(row=11, column=0, columnspan=2, sticky="ew", **pad)

        self.auto_status_label = ttk.Label(auto_frame, text="Not running")
        self.auto_status_label.grid(row=12, column=0, columnspan=2, sticky="w", **pad)

        ttk.Label(
            auto_frame,
            text="Needs SLM + (one camera,\nor both if checked) open first.",
            justify="left",
            foreground="gray40",
        ).grid(row=13, column=0, columnspan=2, sticky="w", **pad)

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

        # Two capture previews (not one shared "last capture") -- added
        # 2026-09-24 alongside Capture Both, so a dual capture shows both
        # results at once instead of one overwriting the other.
        blackfly_prev_frame = ttk.LabelFrame(preview_frame, text="Blackfly capture")
        blackfly_prev_frame.grid(row=0, column=1, **pad)
        self.blackfly_preview_label = tk.Label(
            blackfly_prev_frame, image=self._blank_photo, background="gray20"
        )
        self.blackfly_preview_label.pack()
        self.blackfly_path_label = ttk.Label(blackfly_prev_frame, text="No capture yet")
        self.blackfly_path_label.pack(anchor="w")

        basler_prev_frame = ttk.LabelFrame(preview_frame, text="Basler capture")
        basler_prev_frame.grid(row=0, column=2, **pad)
        self.basler_preview_label = tk.Label(
            basler_prev_frame, image=self._blank_photo, background="gray20"
        )
        self.basler_preview_label.pack()
        self.basler_path_label = ttk.Label(basler_prev_frame, text="No capture yet")
        self.basler_path_label.pack(anchor="w")

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
        if self.cams[choice] is not None:
            self.log(f"{choice} camera is already open.")
            return

        serial = self.serial_entry.get().strip() or None
        cls, _ext = CAM_CLASSES[choice]
        try:
            cam = cls(serial=serial)
            cam.open()
            cam.start()
        except CameraError as e:
            self.log(f"Failed to open/start {choice} camera: {e}")
            return

        self.cams[choice] = cam
        self.log(f"Opened {choice} camera.")
        self._refresh_camera_buttons()

    def close_camera(self):
        choice = self.cam_choice.get()
        if self.cams[choice] is None:
            return
        if self._auto_running:
            self.log("Stopping auto cycle -- a camera is closing.")
            self.stop_auto_cycle()
        self._shutdown_camera(choice)
        self.log(f"{choice} camera closed.")

    def _shutdown_camera(self, choice: str):
        """Stop/close the given camera slot and reset its controls,
        without logging "closed" (the caller logs whatever's contextually
        appropriate)."""
        cam = self.cams.get(choice)
        if cam is None:
            return
        try:
            cam.stop()
            cam.close()
        except CameraError as e:
            self.log(f"Error while closing {choice} camera: {e}")
        self.cams[choice] = None
        self._refresh_camera_buttons()

    def _refresh_camera_buttons(self):
        """Sync the Open/Close/Set Exposure/Capture buttons and the
        exposure readout to whichever camera the radio button currently
        selects, and Capture Both to whether *both* slots are open. Called
        after any open/close, and whenever the radio selection itself
        changes (nothing opens/closes then, but the controls now need to
        reflect a different camera)."""
        choice = self.cam_choice.get()
        is_open = self.cams[choice] is not None
        self.open_cam_btn.configure(state="disabled" if is_open else "normal")
        self.close_cam_btn.configure(state="normal" if is_open else "disabled")
        self.set_exposure_btn.configure(state="normal" if is_open else "disabled")
        self.capture_btn.configure(state="normal" if is_open else "disabled")
        if is_open:
            self.refresh_exposure()
        else:
            self.exposure_label.configure(text="Current exposure: --")

        both_open = all(c is not None for c in self.cams.values())
        self.capture_both_btn.configure(state="normal" if both_open else "disabled")

    def set_exposure(self):
        choice = self.cam_choice.get()
        cam = self.cams[choice]
        if cam is None:
            return
        raw = self.exposure_entry.get().strip()
        try:
            value = float(raw)
            cam.set_exposure_time(value)
            self.log(f"Exposure set to {value} us ({choice}).")
            self._flush_stale_frame(choice)
        except ValueError:
            self.log(f"'{raw}' is not a valid exposure value (expected microseconds, e.g. 15000).")
        except CameraError as e:
            self.log(f"Could not set exposure: {e}")
        self.refresh_exposure()

    def _flush_stale_frame(self, choice: str):
        """Discard exactly one grabbed frame from the given camera, right
        after an exposure change or a mask switch on it.

        set_exposure_time() changes the register instantly, but the sensor
        may already have a frame mid-exposure (or freshly landed in the
        transport buffer) under the OLD setting. StreamBufferHandlingMode
        = NewestOnly (see blackfly_camera.py's _configure()) only
        guarantees capture() returns the newest *arrived* frame -- not one
        that reflects a setting changed moments ago. Grabbing and
        discarding one throwaway frame here means the very next real
        Capture click is guaranteed to reflect the change. This calls only
        the existing public capture() API -- no changes to
        blackfly_camera.py/basler_camera.py.
        """
        cam = self.cams.get(choice)
        if cam is None:
            return
        try:
            OUTPUT_DIR.mkdir(exist_ok=True)
            ext = CAM_CLASSES[choice][1]
            flush_path = OUTPUT_DIR / f"_exposure_flush_{choice}{ext}"
            cam.capture(flush_path)
            flush_path.unlink(missing_ok=True)
        except CameraError as e:
            self.log(f"(non-fatal) couldn't flush stale frame on {choice}: {e}")

    def refresh_exposure(self):
        choice = self.cam_choice.get()
        cam = self.cams[choice]
        if cam is None:
            return
        try:
            value = cam.get_exposure_time()
            self.exposure_label.configure(text=f"Current exposure: {value} us")
        except CameraError as e:
            self.log(f"Could not read exposure: {e}")

    def capture(self):
        self._capture_one(self.cam_choice.get())

    def capture_both(self):
        """Capture from both cameras, one right after the other. This is
        NOT a hardware-synced simultaneous trigger -- there's no shared
        trigger line wired up between these two USB3 Vision cameras in
        this project, so the two shots are back-to-back software calls,
        typically well under a second apart but not truly the same
        instant. Good enough for "roughly the same scene from both
        cameras"; not good enough for anything timing-critical."""
        if not all(c is not None for c in self.cams.values()):
            self.log("Capture Both needs both cameras open.")
            return
        self.log("Capturing both cameras back-to-back (not hardware-synced).")
        self._capture_one("blackfly")
        self._capture_one("basler")

    def _capture_one(self, choice: str):
        cam = self.cams[choice]
        if cam is None:
            return None
        ext = CAM_CLASSES[choice][1]
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / f"{choice}_{self.shots[choice]:03d}{ext}"
        try:
            saved = cam.capture(path)
        except CameraError as e:
            self.log(f"Capture failed ({choice}): {e}")
            return None

        self.shots[choice] += 1
        self.log(f"Saved {saved}")

        img = cv2.imread(str(saved), cv2.IMREAD_UNCHANGED)
        if img is None:
            self.log(f"Saved {saved}, but couldn't reload it here for the preview.")
            return saved

        photo = array_to_photoimage(img, THUMB_W, THUMB_H)
        if choice == "blackfly":
            self._blackfly_thumb_photo = photo
            self.blackfly_preview_label.configure(image=photo)
            self.blackfly_path_label.configure(text=str(saved))
        else:
            self._basler_thumb_photo = photo
            self.basler_preview_label.configure(image=photo)
            self.basler_path_label.configure(text=str(saved))
        return saved

    # ------------------------------------------------------- SLM actions --

    def open_slm(self):
        if self.slm is not None:
            self.log("SLM already open.")
            return
        if not self.masks:
            self.log("No masks loaded -- add a folder first.")
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

    def show_mask(self, index: int):
        if self.slm is None:
            self.log("Open the SLM first.")
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
        # switch to a new one. Discard one throwaway grab per open camera
        # so the next real Capture (or Capture Both) reliably reflects the
        # mask now on screen, rather than a blend of old and new.
        for cam_choice, cam in self.cams.items():
            if cam is not None:
                self._flush_stale_frame(cam_choice)

    # ----------------------------------------------------- mask folders --

    def add_folder_dialog(self):
        folder = filedialog.askdirectory(title="Choose a folder of mask images")
        if not folder:
            return
        self.add_folder_path(Path(folder))

    def _register_folder(self, folder_path: Path):
        """Add a folder to self.mask_root_folders if it isn't already
        there (named after itself, disambiguated against a same-named
        folder already registered). No side effects beyond that -- no
        rescan, no save to disk -- so callers that register several
        folders at once (load_saved_folders()) can batch those into one
        refresh/save instead of one each. Returns the label used, or None
        if it was already registered."""
        folder_path = Path(folder_path).resolve()
        if any(existing == folder_path for existing, _label in self.mask_root_folders):
            return None

        label = folder_path.name
        existing_labels = {lbl for _p, lbl in self.mask_root_folders}
        if label in existing_labels:
            # Two different folders happen to share a basename (e.g. two
            # "masks" folders in different places) -- use the full path
            # instead so they don't collide in the tree or the Auto Cycle
            # dropdown.
            label = str(folder_path)

        self.mask_root_folders.append((folder_path, label))
        return label

    def add_folder_path(self, folder_path: Path):
        """Register one folder of masks, rescan so it shows up right
        away, and remember it for next time this app is launched (see
        load_saved_folders())."""
        folder_path = Path(folder_path).resolve()
        label = self._register_folder(folder_path)
        if label is None:
            self.log(f"'{folder_path}' is already added.")
            return

        self.log(f"Added folder '{label}' ({folder_path}).")
        self.refresh_masks()
        self._save_folder_config()

    def load_saved_folders(self):
        """Restore whichever folders were registered last time this app
        was closed (see _save_folder_config()). Silently does nothing if
        there's no saved config yet (first run). A folder that's been
        moved/deleted since is skipped (logged) and dropped from the
        saved list, so it doesn't keep getting reported as missing on
        every future launch."""
        if not FOLDER_CONFIG_PATH.exists():
            return
        try:
            saved_paths = json.loads(FOLDER_CONFIG_PATH.read_text())
        except (OSError, json.JSONDecodeError) as e:
            self.log(f"Couldn't read remembered mask folders ({FOLDER_CONFIG_PATH}): {e}")
            return

        restored, missing = 0, 0
        for raw in saved_paths:
            folder_path = Path(raw)
            if not folder_path.is_dir():
                missing += 1
                continue
            if self._register_folder(folder_path) is not None:
                restored += 1

        if restored:
            self.refresh_masks()
            self.log(f"Restored {restored} mask folder(s) from last session.")
        if missing:
            self.log(f"{missing} previously-remembered folder(s) no longer exist -- not restored.")
            self._save_folder_config()  # drop the missing ones so they stop being reported

    def _save_folder_config(self):
        try:
            paths = [str(path) for path, _label in self.mask_root_folders]
            FOLDER_CONFIG_PATH.write_text(json.dumps(paths, indent=2))
        except OSError as e:
            self.log(f"(non-fatal) couldn't save remembered mask folders: {e}")

    def remove_selected_folder(self):
        """Un-register whichever folder row(s) are selected in the tree.
        Never touches files on disk -- this only stops the app from
        tracking that folder. Selecting individual mask rows (not a
        folder row) does nothing here; masks themselves are only ever
        added/removed by editing the folder's contents in Finder, then
        hitting Refresh."""
        selection = self.mask_tree.selection()
        folder_iids = [iid for iid in selection if iid.startswith("folder::")]
        if not folder_iids:
            self.log(
                "Select a folder in the list first (masks themselves are "
                "managed in Finder -- add/remove/rename the file there, "
                "then hit Refresh)."
            )
            return

        labels_to_remove = {iid[len("folder::"):] for iid in folder_iids}
        if len(labels_to_remove) >= len(self.mask_root_folders):
            self.log("Can't remove every folder -- at least one must remain.")
            return

        if not messagebox.askyesno(
            "Remove folder(s)",
            f"Stop tracking {len(labels_to_remove)} folder(s)? The files "
            f"themselves are not touched.",
            parent=self.root,
        ):
            return

        self.mask_root_folders = [
            (path, label) for path, label in self.mask_root_folders if label not in labels_to_remove
        ]
        self.log(f"Removed folder(s): {', '.join(sorted(labels_to_remove))}.")
        self.refresh_masks()
        self._save_folder_config()

    def refresh_masks(self):
        """Re-read every registered folder's current contents from disk
        and rebuild self.masks/mask_names/mask_folders from scratch, then
        the tree and the Auto Cycle folder list, then push the result into
        an already-open SLM display. Call this any time the on-disk
        contents may have changed outside the app -- the app doesn't
        watch the filesystem, it only knows what was there as of the last
        refresh."""
        self._rescan_all_folders()
        self._rebuild_mask_tree()
        self._refresh_folder_choices()

        if self.slm is not None:
            if not self.masks:
                self.log("No valid masks found across the registered folders -- SLM display left as-is.")
            else:
                try:
                    self.slm.set_masks(self.masks)
                except SLMError as e:
                    self.log(f"Could not update the open SLM display: {e}")

        self.log(
            f"Refreshed: {len(self.masks)} mask(s) across "
            f"{len(self.mask_root_folders)} folder(s)."
        )

    def _rescan_all_folders(self):
        """Re-read every registered folder's current image files from
        disk. Files that aren't the SLM's native resolution are skipped
        (logged, not fatal to the rest of the scan) rather than resized,
        so a hard-edged amplitude mask never gets blurred into
        intermediate gray values that aren't a real amplitude level."""
        masks, names, folders = [], [], []
        want_w, want_h = MASK_EXPECTED_SIZE

        for folder_path, label in self.mask_root_folders:
            paths = sorted({p for ext in MASK_IMAGE_GLOBS for p in folder_path.glob(ext)})
            for path in paths:
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    self.log(f"Could not load {path.name} in '{label}' -- skipping.")
                    continue
                h, w = mask.shape[:2]
                if (w, h) != (want_w, want_h):
                    self.log(
                        f"'{path.name}' in '{label}' is {w}x{h}, expected "
                        f"{want_w}x{want_h} -- skipping."
                    )
                    continue
                masks.append(mask)
                names.append(path.stem)
                folders.append(label)

        self.masks = masks
        self.mask_names = names
        self.mask_folders = folders

    def _refresh_folder_choices(self):
        """Recompute the distinct folder labels for Auto Cycle's "Loop
        through" dropdown. Falls back to ALL_MASKS_LABEL if the
        currently-selected one no longer exists."""
        names = [ALL_MASKS_LABEL] + sorted(set(self.mask_folders))
        self.auto_set_combo.configure(values=names)
        if self.auto_set_var.get() not in names:
            self.auto_set_var.set(ALL_MASKS_LABEL)

    def _auto_loop_indices(self):
        """Indices into self.masks that Auto Cycle should loop over, per
        the "Loop through" selection."""
        selected = self.auto_set_var.get()
        if selected == ALL_MASKS_LABEL:
            return list(range(len(self.masks)))
        return [i for i, f in enumerate(self.mask_folders) if f == selected]

    def _rebuild_mask_tree(self):
        """Rebuild the two-level (folder > mask) tree from the current
        self.masks/mask_names/mask_folders. Rebuild-from-scratch is
        simpler and less error-prone than patching iids in place after a
        refresh shifts indices around. Keeps whichever folders were
        already expanded, expanded."""
        open_folders = {
            iid for iid in self.mask_tree.get_children("") if self.mask_tree.item(iid, "open")
        }
        self.mask_tree.delete(*self.mask_tree.get_children(""))

        folder_order = []
        folder_children = {}
        for i, folder in enumerate(self.mask_folders):
            if folder not in folder_children:
                folder_children[folder] = []
                folder_order.append(folder)
            folder_children[folder].append(i)

        for folder in folder_order:
            folder_iid = f"folder::{folder}"
            self.mask_tree.insert(
                "", "end", iid=folder_iid, text=folder, open=(folder_iid in open_folders)
            )
            for i in folder_children[folder]:
                self.mask_tree.insert(folder_iid, "end", iid=str(i), text=self.mask_names[i])

    def _on_mask_tree_double_click(self, event):
        """Double-clicking a mask row shows it; double-clicking a folder
        row just expands/collapses it (Treeview's own default behavior --
        nothing to show for a folder itself)."""
        iid = self.mask_tree.identify_row(event.y)
        if not iid or iid.startswith("folder::"):
            return
        if self._auto_running:
            self.log("Stop Auto Cycle before changing masks manually.")
            return
        self.show_mask(int(iid))

    # --------------------------------------------------- auto cycle -----

    def start_auto_cycle(self):
        """Every `interval` milliseconds: switch to the next mask (within
        whichever folder "Loop through" selects), wait `settle`
        milliseconds for the LC panel to physically settle (see
        hardware-notes.md's 2026-09-22 finding -- switching masks isn't
        instantaneous, especially transitioning towards black), then
        capture. Uses tk's own `after()` scheduler rather than a blocking
        loop, so the rest of the GUI stays responsive (Stop button, manual
        controls) while this runs.

        Re-scans every registered folder right before starting, so a
        Finder reorganization done just before clicking Start is picked
        up automatically."""
        if self._auto_running:
            return
        both = self.auto_both_var.get()
        if self.slm is None or (
            (both and not all(c is not None for c in self.cams.values()))
            or (not both and self.cams[self.cam_choice.get()] is None)
        ):
            need = "both cameras" if both else "the currently-selected camera"
            self.log(f"Auto cycle needs {need} and the SLM open first.")
            return

        self.refresh_masks()

        indices = self._auto_loop_indices()
        if not indices:
            self.log(f"'{self.auto_set_var.get()}' has no masks -- nothing to loop over.")
            return

        try:
            interval_ms = int(float(self.auto_interval_entry.get().strip()))
            settle_ms = int(float(self.auto_settle_entry.get().strip()))
        except ValueError:
            self.log("Interval and settle time must be numbers, in milliseconds.")
            return
        if interval_ms <= settle_ms:
            self.log(
                f"Interval ({interval_ms}ms) must be longer than the settle time ({settle_ms}ms)."
            )
            return

        exp_values = None
        if self.auto_exp_var.get():
            exp_values = self._auto_exposure_values()
            if exp_values is None:
                self.log("Exposure start/step/end must be numbers, in microseconds.")
                return

        self._auto_running = True
        self._auto_interval_ms = interval_ms
        self._auto_settle_ms = settle_ms
        self._auto_index = -1  # first tick's +1 lands on 0, i.e. indices[0]
        self._auto_exp_values = exp_values
        self._auto_exp_index = -1
        self.start_auto_btn.configure(state="disabled")
        self.stop_auto_btn.configure(state="normal")
        self.mask_tree.configure(selectmode="none")
        self.capture_btn.configure(state="disabled")
        self.capture_both_btn.configure(state="disabled")
        self.auto_both_check.configure(state="disabled")
        self.auto_set_combo.configure(state="disabled")
        self.add_folder_btn.configure(state="disabled")
        self.remove_folder_btn.configure(state="disabled")
        self.refresh_btn.configure(state="disabled")
        self.auto_exp_check.configure(state="disabled")
        self.auto_exp_start_entry.configure(state="disabled")
        self.auto_exp_step_entry.configure(state="disabled")
        self.auto_exp_end_entry.configure(state="disabled")
        self.load_exp_file_btn.configure(state="disabled")
        self.clear_exp_file_btn.configure(state="disabled")
        self.auto_status_label.configure(text=f"Running: every {interval_ms}ms")
        capture_desc = "both cameras" if both else f"the {self.cam_choice.get()} camera"
        set_desc = self.auto_set_var.get()
        exp_desc = f", exposure over {len(exp_values)} step(s)" if exp_values else ""
        self.log(
            f"Auto cycle started: switch masks (set '{set_desc}', "
            f"{len(indices)} mask(s)) every {interval_ms}ms, capture from "
            f"{capture_desc} {settle_ms}ms after each switch{exp_desc}."
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
        self.auto_both_check.configure(state="normal")
        self.auto_set_combo.configure(state="readonly")
        self.add_folder_btn.configure(state="normal")
        self.remove_folder_btn.configure(state="normal")
        self.refresh_btn.configure(state="normal")
        self.auto_exp_check.configure(state="normal")
        if self._auto_exp_file_values is None:
            self.auto_exp_start_entry.configure(state="normal")
            self.auto_exp_step_entry.configure(state="normal")
            self.auto_exp_end_entry.configure(state="normal")
        self.load_exp_file_btn.configure(state="normal")
        self.clear_exp_file_btn.configure(
            state="normal" if self._auto_exp_file_values is not None else "disabled"
        )
        self.mask_tree.configure(selectmode="extended")
        self._refresh_camera_buttons()  # restores Capture/Capture Both correctly
        self.auto_status_label.configure(text="Not running")
        self.log("Auto cycle stopped.")

    def _auto_cycle_tick(self):
        if not self._auto_running:
            return
        indices = self._auto_loop_indices()
        if not indices:
            self.log(f"'{self.auto_set_var.get()}' has no masks left -- stopping auto cycle.")
            self.stop_auto_cycle()
            return
        self._auto_index = (self._auto_index + 1) % len(indices)
        self.show_mask(indices[self._auto_index])  # also flushes one stale camera frame

        if self._auto_exp_values:
            self._auto_exp_index = (self._auto_exp_index + 1) % len(self._auto_exp_values)
            self._auto_apply_exposure(self._auto_exp_values[self._auto_exp_index])

        self._auto_after_id = self.root.after(self._auto_settle_ms, self._auto_capture_then_reschedule)

    def load_exposure_list_dialog(self):
        """Load a text file of hand-authored exposure values (one
        microsecond value per line -- blank lines and lines starting with
        '#' are ignored). Once loaded, this list takes precedence over
        the start/step/end sweep above for as long as it stays loaded."""
        path = filedialog.askopenfilename(
            title="Choose a text file of exposure values (one per line, in microseconds)",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            lines = Path(path).read_text().splitlines()
        except OSError as e:
            self.log(f"Could not read '{path}': {e}")
            return

        values = []
        skipped = 0
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                values.append(float(line))
            except ValueError:
                skipped += 1

        if not values:
            self.log(f"'{path}' has no usable exposure values -- nothing loaded.")
            return

        self._auto_exp_file_values = values
        self._auto_exp_file_name = Path(path).name
        self.auto_exp_var.set(True)
        self.auto_exp_start_entry.configure(state="disabled")
        self.auto_exp_step_entry.configure(state="disabled")
        self.auto_exp_end_entry.configure(state="disabled")
        self.clear_exp_file_btn.configure(state="normal")
        skip_desc = f", skipped {skipped} invalid line(s)" if skipped else ""
        self.auto_exp_file_label.configure(
            text=f"Loaded '{self._auto_exp_file_name}': {len(values)} value(s){skip_desc}"
        )
        self.log(
            f"Loaded {len(values)} exposure value(s) from '{self._auto_exp_file_name}'"
            f"{skip_desc}. This list will be used instead of start/step/end."
        )

    def clear_exposure_list(self):
        """Drop the loaded exposure list and fall back to the
        start/step/end sweep fields again."""
        if self._auto_exp_file_values is None:
            return
        self._auto_exp_file_values = None
        self._auto_exp_file_name = None
        self.auto_exp_start_entry.configure(state="normal")
        self.auto_exp_step_entry.configure(state="normal")
        self.auto_exp_end_entry.configure(state="normal")
        self.clear_exp_file_btn.configure(state="disabled")
        self.auto_exp_file_label.configure(text="(no exposure list loaded)")
        self.log("Exposure list cleared -- back to start/step/end.")

    def _auto_exposure_values(self):
        """Parse the start/step/end exposure entries into the sweep of
        values Auto Cycle steps through (wrapping back to the start once
        past end, same as the mask loop wraps). Returns None if any of
        the three fields isn't a number. A step of 0 (or one that would
        never reach end) still returns a single-value list rather than
        looping forever building it.

        If a text-file exposure list is loaded (see
        load_exposure_list_dialog()), it takes precedence and is returned
        as-is instead of computing a sweep."""
        if self._auto_exp_file_values is not None:
            return list(self._auto_exp_file_values)
        try:
            start = float(self.auto_exp_start_entry.get().strip())
            step = float(self.auto_exp_step_entry.get().strip())
            end = float(self.auto_exp_end_entry.get().strip())
        except ValueError:
            return None

        if step == 0:
            return [start]

        values = []
        v = start
        if step > 0:
            while v <= end + 1e-6:
                values.append(v)
                v += step
        else:
            while v >= end - 1e-6:
                values.append(v)
                v += step
        return values or [start]

    def _auto_apply_exposure(self, value: float):
        """Set exposure to `value` on whichever camera(s) this cycle is
        capturing from -- the selected one, or both if "Capture both
        cameras" is checked -- and flush the resulting stale frame on
        each, same as a manual Set Exposure click does."""
        targets = list(self.cams.keys()) if self.auto_both_var.get() else [self.cam_choice.get()]
        applied_any = False
        for choice in targets:
            cam = self.cams.get(choice)
            if cam is None:
                continue
            try:
                cam.set_exposure_time(value)
                self._flush_stale_frame(choice)
                applied_any = True
            except CameraError as e:
                self.log(f"Could not set exposure on {choice}: {e}")
        if applied_any:
            self.log(f"Exposure set to {value:g} us for this cycle.")
            if not self.auto_both_var.get():
                self.refresh_exposure()

    def _auto_capture_then_reschedule(self):
        if not self._auto_running:
            return
        if self.auto_both_var.get():
            self.capture_both()
        else:
            self.capture()
        remaining_ms = max(0, self._auto_interval_ms - self._auto_settle_ms)
        self._auto_after_id = self.root.after(remaining_ms, self._auto_cycle_tick)

    # ------------------------------------------------------------- exit --

    def on_close(self):
        if self._auto_running:
            self.stop_auto_cycle()
        if self.slm is not None:
            self.slm.close()
        for choice in list(self.cams):
            if self.cams[choice] is not None:
                self._shutdown_camera(choice)
        self.root.destroy()


def main():
    args = sys.argv[1:]
    mirrored = "--mirrored" in args
    args = [a for a in args if a != "--mirrored"]

    root = tk.Tk()
    app = App(root, mirrored)
    app.load_saved_folders()

    if args:
        app.add_folder_path(Path(args[0]))
    elif not app.mask_root_folders:
        # Only fall back to the generated test patterns when there's
        # nothing remembered from a previous session and no folder was
        # passed on the command line -- otherwise this would re-add them
        # (harmlessly, but pointlessly) on every single launch.
        app.add_folder_path(ensure_test_pattern_folder())

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

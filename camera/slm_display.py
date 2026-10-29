"""
Display control for the HOLOEYE amplitude SLM (HES 7020-1 6001).

Unlike the two cameras in this project, this SLM has no vendor SDK / frame-
grab API to call into (see project notes). It's addressed purely as a
second monitor over HDMI: whatever image is shown in a borderless window
on that display *is* the mask pattern. This module treats it exactly that
way -- open a borderless window that exactly covers the SLM's monitor,
then control which mask image is showing and for how long.

Setup:
    pip install opencv-python-headless   # only used to LOAD mask images
    pip install screeninfo                # for finding the SLM's monitor geometry
    tkinter (standard library, but some conda/pyenv Python builds omit the
        Tk bindings -- check with `python3 -c "import tkinter"`; if that
        fails, try `conda install -c conda-forge tk`, or rebuild the env
        with a conda-forge Python, which normally bundles Tk support)

    IMPORTANT -- this module used to require the full (non-headless)
    opencv-python package, because it displayed images via cv2's own GUI
    window. It now displays via Tkinter instead (see "Why Tkinter"
    below), so cv2 is only used by load_masks_from_folder() to read mask
    files off disk -- opencv-python-headless is enough. If you still have
    the full opencv-python installed from before, there's no need to
    remove it; either package works for the loading path.

    Physically: plug the SLM's driver unit into an HDMI output as an
    EXTENDED desktop display, not a mirrored one, and power it on. Confirm
    your OS's display settings show it as a separate ~1920x1080 monitor
    before running anything here. (mirrored=True exists as a fallback for
    when extending genuinely isn't available -- see its docstring below --
    but extending is the normal, recommended setup.)

Why Tkinter, not cv2's own window (changed 2026-09-22):
    This module originally displayed masks in an OpenCV highgui window.
    Getting that window to actually cover the SLM's monitor edge-to-edge
    on macOS turned out to be persistently unreliable:
      - cv2.moveWindow()/resizeWindow() position the window's outer
        FRAME (title bar included), not its content area. However
        precisely the monitor's bounds were requested, a title-bar-sized
        strip of chrome always ate into the top of the display, and an
        equal-sized strip of mask content was pushed off the bottom.
      - Trying to compensate from outside the window (shifting it up by
        the title bar's height, growing height to match) didn't help --
        macOS appears to refuse moving a normal window's title bar fully
        off-screen, clamping it back, so the strip persisted regardless
        of how the offset was tuned.
      - cv2's own native fullscreen mode
        (cv2.WND_PROP_FULLSCREEN/WINDOW_FULLSCREEN) was tried twice as an
        alternative and failed both times -- stuck on a blank/gray frame
        with no content, and on one attempt it also fullscreened on the
        wrong monitor entirely, ignoring the window's actual position.
      - cv2.getWindowImageRect(), used to diagnose all of the above,
        turned out to report coordinates in Retina-scaled physical
        pixels while moveWindow/resizeWindow accept plain points
        (matching screeninfo) -- so comparing the two directly produced
        nonsense numbers, which is part of why the title-bar
        compensation attempt above couldn't be made to work reliably.
    A Tkinter window created with overrideredirect(True) has no title
    bar or border in the first place, so there's nothing to compensate
    for, and its geometry() position/size describes the content area
    directly rather than an outer frame. This sidesteps the whole
    problem rather than working around it.

Usage:
    from slm_display import SLMDisplay, load_masks_from_folder

    masks = load_masks_from_folder("masks/")   # sorted list of grayscale arrays
    with SLMDisplay(masks) as slm:
        slm.show_mask(0)                       # display one mask, under your own control
        ...
        slm.run_sequence(interval_s=2.0, cycles=3)   # or cycle through all of them automatically

Not implemented here (flagged for later, out of scope for this pass):
    The SLM's USB connection (calibration + trigger-sync output, per the
    HOLOEYE product page) isn't touched by this module -- it only drives
    the HDMI/display side. If mask changes ever need to be
    hardware-triggered/synced to camera exposures rather than timed in
    software, that's a separate piece of work on the USB side.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from screeninfo import get_monitors
except ImportError:
    get_monitors = None

try:
    import tkinter as tk
except ImportError:
    tk = None


class SLMError(RuntimeError):
    """Raised for any SLM display failure (monitor not found, bad mask, etc.)."""


def load_masks_from_folder(
    folder: Union[str, Path], pattern: str = "*.png"
) -> List[np.ndarray]:
    """Load all mask images in `folder` matching `pattern`, sorted by filename.

    Loads each as a single-channel (grayscale) array -- amplitude masks are
    grayscale intensity patterns, not color images. Sort order is filename
    order, so name files so that sorts into the sequence you want (e.g.
    mask_000.png, mask_001.png, ...).

    Raises:
        SLMError: if opencv isn't installed, no files match, or a file
            fails to load as an image.
    """
    if cv2 is None:
        raise SLMError(
            "opencv-python(-headless) is not installed. Run: "
            "pip install opencv-python-headless"
        )

    folder = Path(folder)
    paths = sorted(folder.glob(pattern))
    if not paths:
        raise SLMError(f"No files matching {pattern!r} found in {folder}.")

    masks = []
    for path in paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise SLMError(f"Could not load mask image: {path}")
        masks.append(mask)
    return masks


def _ndarray_to_pgm_bytes(arr: np.ndarray) -> bytes:
    """Encode a uint8 grayscale array as binary PGM (P5) bytes.

    tkinter.PhotoImage can load PGM natively, which lets masks reach the
    screen without adding a Pillow dependency just for this one
    conversion.
    """
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w = arr.shape[:2]
    header = f"P5\n{w} {h}\n255\n".encode("ascii")
    return header + arr.tobytes()


class SLMDisplay:
    """Shows a sequence of amplitude masks full-screen on the SLM's monitor.

    Displays via a borderless (overrideredirect) Tkinter window -- see
    the module docstring's "Why Tkinter" section for why this replaced an
    earlier OpenCV-window-based implementation.
    """

    def __init__(
        self,
        masks: Sequence[np.ndarray],
        monitor_index: Optional[int] = None,
        window_name: str = "SLM",
        expected_size: Optional[Tuple[int, int]] = (1920, 1080),
        mirrored: bool = False,
        master=None,
    ):
        """Prepare (but don't yet display) a sequence of masks.

        Args:
            masks: Mask images to show, in order, as a sequence of 2D
                numpy arrays (grayscale). Use load_masks_from_folder() to
                build this from a directory of image files.
            monitor_index: Which monitor (0-based, per
                screeninfo.get_monitors()) is the SLM. If None, open()
                auto-picks the first *non-primary* monitor it finds (or,
                if mirrored=True, the primary one instead) -- fine with
                exactly one external display connected, ambiguous with
                more than one. Takes priority over mirrored if both are
                set.
            window_name: Kept for API compatibility with the previous
                cv2-based implementation; a borderless window has no
                title bar to display it in, so this is currently unused.
            expected_size: (width, height) every mask must match -- the
                SLM's native panel resolution. Pass None to skip this
                check (e.g. if you intend to pre-scale masks yourself);
                not recommended for a hard-edged amplitude mask, since
                resizing can blur sharp edges into intermediate gray
                values that aren't a real amplitude level.
            mirrored: Set True if the SLM's display is currently set to
                MIRROR the primary screen (same image on both), rather
                than extend the desktop. This is a fallback for when
                extending genuinely isn't available -- with mirroring
                on, filling the *primary* screen fully fills the SLM too,
                since it shows an identical copy. When True, open()
                covers the primary monitor fully instead of hunting for
                a non-primary one. Prefer extend-desktop mode (the
                default) when you can, since mirroring forces both
                screens to the same resolution/aspect ratio, which may
                not match the SLM's native panel pixel-for-pixel.
            master: An existing tkinter widget (typically a Tk() root
                from a larger app, e.g. a control-panel GUI) to create
                this display as a child Toplevel of, instead of its own
                independent Tk() root. A process should generally only
                ever have one real Tk() root; pass the app's root here
                when SLMDisplay is used alongside other Tkinter UI so
                the SLM window becomes a Toplevel sharing that
                interpreter rather than a second, separate one. Leave as
                None for standalone use (e.g. the test_on_hardware_*.py
                scripts), where SLMDisplay creates and owns its own Tk().

        Raises:
            SLMError: if tkinter or opencv isn't installed, no masks were
                given, or a mask doesn't match expected_size.
        """
        if tk is None:
            raise SLMError(
                "tkinter is not available in this Python installation. "
                "It's normally part of the standard library, but some "
                "conda/pyenv builds omit the Tk bindings. Try `conda "
                "install -c conda-forge tk`, or rebuild this environment "
                "from a conda-forge Python (which normally bundles Tk "
                "support), then check with `python3 -c \"import "
                "tkinter\"`."
            )
        if cv2 is None:
            raise SLMError(
                "opencv-python(-headless) is not installed (only needed "
                "for loading mask images from files). Run: "
                "pip install opencv-python-headless"
            )
        if not masks:
            raise SLMError("At least one mask is required.")

        if expected_size is not None:
            want_w, want_h = expected_size
            for i, mask in enumerate(masks):
                h, w = mask.shape[:2]
                if (w, h) != (want_w, want_h):
                    raise SLMError(
                        f"Mask {i} is {w}x{h}, expected {want_w}x{want_h} "
                        f"(the SLM's native resolution). Pass "
                        f"expected_size=None to skip this check if that's "
                        f"intentional."
                    )

        self.masks = list(masks)
        self.expected_size = expected_size
        self.monitor_index = monitor_index
        self.window_name = window_name
        self.mirrored = mirrored
        self.master = master
        self._is_open = False
        self._current_index: Optional[int] = None
        self._stop_requested = False
        self._root = None
        self._label = None
        self._photos: List = []

    def add_mask(self, mask: np.ndarray) -> int:
        """Add one more mask after any already loaded, returning its index.

        Works whether or not open() has been called yet. If the display
        is already open, the new mask becomes immediately selectable via
        show_mask() -- no need to close and reopen. Validated against the
        same expected_size the constructor used (None skips the check,
        same as the constructor).

        Raises:
            SLMError: if the mask's size doesn't match expected_size.
        """
        if self.expected_size is not None:
            want_w, want_h = self.expected_size
            h, w = mask.shape[:2]
            if (w, h) != (want_w, want_h):
                raise SLMError(
                    f"New mask is {w}x{h}, expected {want_w}x{want_h} "
                    f"(the SLM's native resolution). Pass "
                    f"expected_size=None at construction to skip this "
                    f"check if that's intentional."
                )

        self.masks.append(mask)
        if self._is_open:
            self._photos.append(tk.PhotoImage(data=_ndarray_to_pgm_bytes(mask)))
        return len(self.masks) - 1

    def remove_mask(self, index: int):
        """Remove one mask by index (at least one mask must always remain).

        Works whether or not open() has been called yet. If the display
        is already open and the removed mask was the one on screen, falls
        back to showing mask 0 of what remains, rather than leaving a
        stale or missing image up; otherwise the display keeps showing
        whatever it was already showing (index-adjusted if the removed
        mask came before it in the list).

        Returns:
            The index now actually showing, if the display is open
            (already re-painted if it changed); None if not open yet.

        Raises:
            SLMError: if this would remove the last remaining mask, or
                index is out of range.
        """
        if len(self.masks) <= 1:
            raise SLMError("Can't remove the last remaining mask -- at least one is required.")
        if not (0 <= index < len(self.masks)):
            raise SLMError(f"Mask index {index} out of range (0-{len(self.masks) - 1}).")

        del self.masks[index]
        if not self._is_open:
            return None

        del self._photos[index]
        if self._current_index == index:
            new_index = 0
            self.show_mask(new_index)
        else:
            new_index = self._current_index
            if new_index is not None and new_index > index:
                new_index -= 1
                self._current_index = new_index
        return new_index

    def set_masks(self, new_masks) -> None:
        """Replace the whole mask list in one go -- e.g. after re-scanning
        a mask folder on disk for changes made outside this process
        (files added, removed, renamed, or dragged between folders in
        Finder), rather than tracking each change one at a time via
        add_mask()/remove_mask(). Requires at least one mask.

        If the display is already open, rebuilds the cached PhotoImages
        and keeps showing whatever index was already current if that
        index still exists in the new list (clamped to a valid index
        otherwise), so a routine refresh doesn't blank the screen or jump
        back to mask 0 unnecessarily.

        Raises:
            SLMError: if new_masks is empty, or (when expected_size is
                set) any mask doesn't match it.
        """
        if not new_masks:
            raise SLMError("At least one mask is required.")
        if self.expected_size is not None:
            want_w, want_h = self.expected_size
            for i, mask in enumerate(new_masks):
                h, w = mask.shape[:2]
                if (w, h) != (want_w, want_h):
                    raise SLMError(
                        f"Mask {i} is {w}x{h}, expected {want_w}x{want_h} "
                        f"(the SLM's native resolution)."
                    )

        self.masks = list(new_masks)
        if not self._is_open:
            return

        self._photos = [
            tk.PhotoImage(data=_ndarray_to_pgm_bytes(mask)) for mask in self.masks
        ]
        if self._current_index is None or self._current_index >= len(self.masks):
            self._current_index = 0
        self.show_mask(self._current_index)

    def _find_monitor(self):
        """Pick the target monitor per monitor_index/mirrored, same logic
        as before -- only how it's *used* (window backend) changed."""
        if get_monitors is None:
            raise SLMError("screeninfo is not installed. Run: pip install screeninfo")

        monitors = get_monitors()
        if not monitors:
            raise SLMError("No monitors detected.")

        if self.monitor_index is not None:
            if self.monitor_index >= len(monitors):
                raise SLMError(
                    f"monitor_index={self.monitor_index} but only "
                    f"{len(monitors)} monitor(s) detected."
                )
            return monitors[self.monitor_index]

        if self.mirrored:
            primary = [m for m in monitors if getattr(m, "is_primary", False)]
            if not primary:
                raise SLMError("No primary monitor detected (unexpected).")
            return primary[0]

        non_primary = [m for m in monitors if not getattr(m, "is_primary", False)]
        if not non_primary:
            raise SLMError(
                "No secondary monitor detected -- is the SLM plugged "
                "in and set to extend (not mirror) the desktop? Pass "
                "monitor_index explicitly to override this check, or "
                "mirrored=True if the SLM is set to mirror instead."
            )
        return non_primary[0]

    def open(self) -> None:
        """Find the SLM's monitor and create a borderless window on it.

        Raises:
            SLMError: if screeninfo isn't installed, or no suitable
                monitor is found.
        """
        monitor = self._find_monitor()

        self._root = tk.Toplevel(self.master) if self.master is not None else tk.Tk()

        # Force 1 point == 1 pixel. Applied on self._root.tk either way:
        # a Toplevel shares its master's Tcl interpreter, so this sets
        # scaling for the whole app (control-panel GUI included) when
        # master is given, which is what we want -- consistent,
        # unscaled pixel math everywhere, not just in this window. Without this, Tk can inherit a DPI
        # "points per pixel" scaling factor (often 2.0) from the primary
        # Retina display and apply it globally, even to a window living
        # on a different, non-Retina monitor -- which shrinks the
        # PhotoImage's rendered size relative to the window's own
        # geometry (also specified in points) and left an even gap of
        # blank window around the mask on all sides. This is the same
        # flavor of points-vs-pixels mismatch that made
        # cv2.getWindowImageRect() unusable earlier, just showing up in
        # a different place now that the backend has changed.
        self._root.tk.call("tk", "scaling", 1.0)

        self._root.overrideredirect(True)  # no title bar, no border at all
        self._root.geometry(
            f"{monitor.width}x{monitor.height}+{monitor.x}+{monitor.y}"
        )
        self._root.configure(background="black")
        try:
            self._root.attributes("-topmost", True)
        except Exception:
            pass  # not critical if the window manager doesn't support this

        self._photos = [
            # No format= given: Tk auto-detects PGM from the "P5" magic
            # number in the header. (Explicitly passing format="PGM"
            # raises TclError -- Tk's built-in netpbm reader identifies
            # itself as "PPM" and covers PBM/PGM/PPM together, but
            # auto-detection sidesteps needing to know that.)
            tk.PhotoImage(data=_ndarray_to_pgm_bytes(mask))
            for mask in self.masks
        ]

        self._label = tk.Label(
            self._root,
            image=self._photos[0],
            bd=0,
            highlightthickness=0,
            background="black",
        )
        self._label.pack(fill="both", expand=True)

        # Esc or 'q' stops a running run_sequence() early, same as before.
        self._root.bind("<Escape>", lambda _e: self.stop_sequence())
        self._root.bind("q", lambda _e: self.stop_sequence())

        self._root.update_idletasks()
        self._root.update()
        self._root.focus_force()

        self._is_open = True
        self._current_index = 0

        print(
            f"[SLM] Requested monitor at ({monitor.x}, {monitor.y}), size "
            f"{monitor.width}x{monitor.height}. Borderless window geometry "
            f"now reports: {self._root.geometry()!r}."
        )

    def show_mask(self, index: int) -> None:
        """Display masks[index] immediately, replacing whatever was shown before.

        Raises:
            SLMError: if open() hasn't been called yet, or index is out of range.
        """
        if not self._is_open:
            raise SLMError("Display is not open. Call open() first.")
        if not (0 <= index < len(self.masks)):
            raise SLMError(f"Mask index {index} out of range (0-{len(self.masks) - 1}).")

        self._label.configure(image=self._photos[index])
        self._label.image = self._photos[index]  # keep a live reference
        self._root.update_idletasks()
        self._root.update()
        self._current_index = index

    def run_sequence(
        self,
        interval_s: Union[float, Sequence[float]],
        loop: bool = True,
        cycles: Optional[int] = None,
    ) -> None:
        """Show all masks in order, advancing at set intervals.

        This is a BLOCKING call -- it owns the calling thread until the
        sequence ends (cycles reached) or stop_sequence() flags it to stop,
        because Tkinter's GUI calls must run on the main thread. If the
        display needs to run in the background, run this from its own
        process rather than a Python thread.

        Args:
            interval_s: Seconds to show each mask before advancing to the
                next. Either one number applied to every mask, or a
                sequence of per-mask durations (must be the same length as
                masks) if different masks need different dwell times.
            loop: If True (default), keep cycling through the mask list
                (forever, or `cycles` times if given). If False, show each
                mask once and stop.
            cycles: If set, stop after this many full passes through the
                mask list, regardless of `loop`.

        Raises:
            SLMError: if open() hasn't been called yet, or interval_s is a
                sequence with the wrong length.
        """
        if not self._is_open:
            raise SLMError("Display is not open. Call open() first.")

        if isinstance(interval_s, (int, float)):
            intervals_s = [max(0.001, float(interval_s))] * len(self.masks)
        else:
            intervals = list(interval_s)
            if len(intervals) != len(self.masks):
                raise SLMError(
                    f"interval_s has {len(intervals)} entries but there are "
                    f"{len(self.masks)} masks -- provide one number for all "
                    f"masks, or one entry per mask."
                )
            intervals_s = [max(0.001, float(s)) for s in intervals]

        self._stop_requested = False
        completed_cycles = 0
        poll_s = 0.03  # how often to pump the Tk event loop while waiting

        while not self._stop_requested:
            for i in range(len(self.masks)):
                if self._stop_requested:
                    break
                self.show_mask(i)
                elapsed = 0.0
                while elapsed < intervals_s[i] and not self._stop_requested:
                    self._root.update()  # keeps the window responsive to Esc/q
                    time.sleep(poll_s)
                    elapsed += poll_s

            completed_cycles += 1
            if not loop or (cycles is not None and completed_cycles >= cycles):
                break

    def stop_sequence(self) -> None:
        """Signal a running run_sequence() to stop after its current mask."""
        self._stop_requested = True

    def close(self) -> None:
        """Close the display window.

        Safe to call multiple times / after a failed open().
        """
        if self._is_open:
            try:
                self._root.destroy()
            except Exception:
                pass
            self._is_open = False
        self._current_index = None
        self._root = None
        self._label = None
        self._photos = []

    def __enter__(self):
        """`with SLMDisplay(masks) as slm:` shortcut for open()."""
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        """`with` block exit: signal any running sequence to stop, then close()."""
        self.stop_sequence()
        self.close()

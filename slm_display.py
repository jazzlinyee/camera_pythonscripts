"""
Display control for the HOLOEYE amplitude SLM (HES 7020-1 6001).

Unlike the two cameras in this project, this SLM has no vendor SDK / frame-
grab API to call into (see project notes). It's addressed purely as a
second monitor over HDMI: whatever image is shown in a fullscreen window
on that display *is* the mask pattern. This module treats it exactly that
way -- open a fullscreen window on the SLM's monitor, then control which
mask image is showing and for how long.

Setup:
    pip install opencv-python      # NOT opencv-python-headless -- see note below
    pip install screeninfo         # for finding the SLM's monitor geometry

    IMPORTANT -- opencv-python vs opencv-python-headless:
    blackfly_camera.py / basler_camera.py only ever *save* image files, so
    they work fine with opencv-python-headless (no GUI backend). This
    module needs to *display* images in a real window, which headless
    builds cannot do at all (cv2.namedWindow/imshow raise "function not
    implemented" errors). opencv-python and opencv-python-headless both
    install as the `cv2` module and having both installed at once causes
    conflicts, so if this environment currently has opencv-python-headless
    installed for the cameras:
        pip uninstall opencv-python-headless
        pip install opencv-python
    (The camera scripts' save functions work identically with either
    package -- only this module actually needs the GUI-enabled one.)

    Physically: plug the SLM's driver unit into an HDMI output as an
    EXTENDED desktop display, not a mirrored one, and power it on. Confirm
    your OS's display settings show it as a separate ~1920x1080 monitor
    before running anything here.

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
        raise SLMError("opencv-python is not installed. Run: pip install opencv-python")

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


class SLMDisplay:
    """Shows a sequence of amplitude masks full-screen on the SLM's monitor."""

    def __init__(
        self,
        masks: Sequence[np.ndarray],
        monitor_index: Optional[int] = None,
        window_name: str = "SLM",
        expected_size: Optional[Tuple[int, int]] = (1920, 1080),
    ):
        """Prepare (but don't yet display) a sequence of masks.

        Args:
            masks: Mask images to show, in order, as a sequence of 2D
                numpy arrays (grayscale). Use load_masks_from_folder() to
                build this from a directory of image files.
            monitor_index: Which monitor (0-based, per
                screeninfo.get_monitors()) is the SLM. If None, open()
                auto-picks the first *non-primary* monitor it finds --
                fine with exactly one external display connected,
                ambiguous with more than one.
            window_name: OpenCV window title -- irrelevant once
                fullscreen, only visible briefly while the window is created.
            expected_size: (width, height) every mask must match -- the
                SLM's native panel resolution. Pass None to skip this
                check (e.g. if you intend to pre-scale masks yourself);
                not recommended for a hard-edged amplitude mask, since
                resizing can blur sharp edges into intermediate gray
                values that aren't a real amplitude level.

        Raises:
            SLMError: if opencv isn't installed, no masks were given, or
                a mask doesn't match expected_size.
        """
        if cv2 is None:
            raise SLMError("opencv-python is not installed. Run: pip install opencv-python")
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
        self.monitor_index = monitor_index
        self.window_name = window_name
        self._is_open = False
        self._current_index: Optional[int] = None
        self._stop_requested = False

    def open(self) -> None:
        """Find the SLM's monitor and create a fullscreen window on it.

        Raises:
            SLMError: if screeninfo isn't installed, or no suitable
                monitor is found.
        """
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
            monitor = monitors[self.monitor_index]
        else:
            # Default: first non-primary monitor -- the SLM is virtually
            # always the *extended* display, not the laptop's own screen.
            non_primary = [m for m in monitors if not getattr(m, "is_primary", False)]
            if not non_primary:
                raise SLMError(
                    "No secondary monitor detected -- is the SLM plugged "
                    "in and set to extend (not mirror) the desktop? Pass "
                    "monitor_index explicitly to override this check."
                )
            monitor = non_primary[0]

        # macOS's native OpenCV fullscreen mode (WINDOW_FULLSCREEN) is
        # unreliable in practice -- it can leave the window permanently
        # stuck on a blank/gray transitional frame no matter how much the
        # event loop is pumped afterward. Instead, size and position a
        # normal window to exactly cover the SLM's monitor, skipping the
        # native fullscreen transition entirely. The only cosmetic cost is
        # a thin title bar; it doesn't affect the image data reaching the
        # SLM panel itself.
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.waitKey(1)  # let the window fully materialize before moving it

        # Some window managers only loosely honor the first geometry
        # request (especially across displays with an unusual relative
        # offset), so reassert position+size a few times with the event
        # loop pumped in between, rather than trusting a single call.
        # Sized to 1512x982 (Rika's primary screen's own resolution) rather
        # than the SLM's native 1920x1080, so the window fits entirely
        # on-screen and can be seen/dragged/resized in full while cross-
        # monitor placement is still being sorted out. The mask image
        # itself is still 1920x1080 -- cv2 scales it down to fit this
        # window, so proportions are preserved, just not pixel-exact to
        # the SLM's native resolution. Once the window reliably lands on
        # the SLM's own monitor, switch this back to monitor.width/height
        # so each mask pixel maps 1:1 to a panel pixel.
        window_w, window_h = 1512, 982
        for _ in range(5):
            cv2.moveWindow(self.window_name, monitor.x, monitor.y)
            cv2.resizeWindow(self.window_name, window_w, window_h)
            cv2.waitKey(50)

        cv2.imshow(self.window_name, self.masks[0])
        for _ in range(15):
            cv2.waitKey(30)

        self._is_open = True
        self._current_index = 0

        # Report where the window actually ended up vs. where it was
        # asked to go -- if these don't match, positioning isn't landing
        # correctly on this machine and needs a different fix.
        try:
            actual_rect = cv2.getWindowImageRect(self.window_name)
            print(
                f"[SLM] Requested monitor at ({monitor.x}, {monitor.y}), "
                f"window size {window_w}x{window_h}. Window now "
                f"reports rect (x, y, w, h) = {actual_rect}."
            )
        except Exception:
            pass

    def show_mask(self, index: int) -> None:
        """Display masks[index] immediately, replacing whatever was shown before.

        Raises:
            SLMError: if open() hasn't been called yet, or index is out of range.
        """
        if not self._is_open:
            raise SLMError("Display is not open. Call open() first.")
        if not (0 <= index < len(self.masks)):
            raise SLMError(f"Mask index {index} out of range (0-{len(self.masks) - 1}).")

        cv2.imshow(self.window_name, self.masks[index])
        cv2.waitKey(1)  # pump the GUI event loop so the image actually paints
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
        because cv2's GUI calls must run on the main thread on some
        platforms (macOS in particular). If the display needs to run in
        the background, run this from its own process rather than a
        Python thread.

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
            intervals_ms = [max(1, int(interval_s * 1000))] * len(self.masks)
        else:
            intervals = list(interval_s)
            if len(intervals) != len(self.masks):
                raise SLMError(
                    f"interval_s has {len(intervals)} entries but there are "
                    f"{len(self.masks)} masks -- provide one number for all "
                    f"masks, or one entry per mask."
                )
            intervals_ms = [max(1, int(s * 1000)) for s in intervals]

        self._stop_requested = False
        completed_cycles = 0

        while not self._stop_requested:
            for i in range(len(self.masks)):
                if self._stop_requested:
                    break
                self.show_mask(i)
                # waitKey() both waits out the interval AND keeps the GUI
                # responsive; pressing Esc/q doubles as a manual stop.
                key = cv2.waitKey(intervals_ms[i]) & 0xFF
                if key in (27, ord("q")):
                    self._stop_requested = True

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
            cv2.destroyWindow(self.window_name)
            self._is_open = False
        self._current_index = None

    def __enter__(self):
        """`with SLMDisplay(masks) as slm:` shortcut for open()."""
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        """`with` block exit: signal any running sequence to stop, then close()."""
        self.stop_sequence()
        self.close()

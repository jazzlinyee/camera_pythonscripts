"""
Wrapper for the FLIR/Teledyne Blackfly S BFS-U3-16S2C (USB3, color).

Setup (one-time, on the machine the camera is physically connected to):
    1. Download and install the Spinnaker SDK for your OS from Teledyne's
       site (requires a free account):
       https://www.teledynevisionsolutions.com/support/support-center/software-firmware-downloads/
    2. From the SAME download page, get the "Spinnaker Python" package
       (PySpin) matching your OS and Python version exactly
       (e.g. spinnaker_python-4.x.x.x-cp310-cp310-linux_x86_64), then:
           pip install <the wheel file you downloaded>
    3. On Linux, the Spinnaker SDK installer sets up a udev rule so the
       camera is usable without sudo. If you get permission errors, re-run
       the installer's configure script or check that rule is in place.

Capture policy (set after the 2026-09 meeting):
    Frames are saved RAW -- the native Bayer mosaic straight off the
    sensor, with on-camera gamma correction and auto white balance both
    turned off, and the red/blue white-balance ratios set to 1.0 (neutral;
    added 2026-10-28 -- switching auto white balance off only freezes them).
    Run check_blackfly_raw.py with the camera connected to read the
    settings back from the camera. There is no demosaic/color-conversion step in capture()
    on purpose: converting to a viewable color image is exactly the kind
    of processing this policy is meant to avoid. Frames are still written
    out as .png (PNG happily stores a single-channel 8-bit array; it just
    won't look like a normal color photo when opened directly -- that's
    expected for raw sensor data).

    NOTE: "PixelFormat_BayerRG8" below is this sensor's commonly documented
    native CFA order, but hasn't been confirmed against real hardware in
    this project yet (no Blackfly emulator exists to test against -- see
    test_blackfly_camera.py for the PySpin-mocked alternative). Check the
    PixelFormat node's available entries in SpinView once the camera is in
    hand, and swap the value below if the real order differs.

    If a demosaiced (but still gamma/white-balance-free) color image turns
    out to be what's actually wanted instead of the raw mosaic, re-add a
    `.Convert(PySpin.PixelFormat_BGR8, PySpin.HQ_LINEAR)` call in capture()
    before saving -- the Gamma/BalanceWhiteAuto settings below still apply
    either way.

Usage:
    from blackfly_camera import BlackflyCamera

    cam = BlackflyCamera(serial="12345678")  # or None to grab the first camera found
    with cam:
        cam.capture("frame_0001.png")
        cam.set_exposure_time(20000)  # adjust exposure without stopping the stream
        cam.capture("frame_0002.png")

    # or manually, without the context manager:
    cam = BlackflyCamera()
    cam.open()
    cam.start()
    cam.capture("frame.png")
    cam.stop()
    cam.close()

Testing without a physical camera:
    Spinnaker/PySpin has no official emulated-camera mode, so there's
    nothing to actually run this against without the real hardware. See
    test_blackfly_camera.py for a test suite that mocks PySpin entirely
    and verifies this file's control flow (open/start/capture/stop/close,
    the exposure-time controller, and the gamma/white-balance/pixel-format
    policy) with no SDK or camera required.
"""

from pathlib import Path
from typing import Optional, Union

from camera_base import Camera, CameraError

try:
    import PySpin
except ImportError:
    PySpin = None


class BlackflyCamera(Camera):
    """Controls one FLIR Blackfly S camera via the Spinnaker SDK (PySpin)."""

    def __init__(
        self,
        serial: Optional[str] = None,
        name: str = "BFS-U3-16S2C",
        exposure_us: Optional[float] = None,
        gain_db: Optional[float] = None,
    ):
        """Configure (but don't yet connect to) a Blackfly camera.

        Args:
            serial: Serial number of the camera to use. If None, open()
                will grab whichever Spinnaker-compatible camera it finds
                first -- fine with only one Blackfly plugged in, ambiguous
                with more than one.
            name: Human-readable label, purely for your own logging/prints.
            exposure_us: Manual exposure time in microseconds, applied at
                open(). If None, auto-exposure is left on initially -- use
                set_exposure_time() later to switch to manual. Either way,
                set_exposure_time() can change it again at any point after
                open().
            gain_db: Manual gain in dB. If None, auto-gain is left on.

        Raises:
            CameraError: if PySpin isn't installed.
        """
        super().__init__(name)
        if PySpin is None:
            raise CameraError(
                "PySpin is not installed. Install the Spinnaker SDK and the "
                "matching PySpin wheel from Teledyne's site (see module docstring)."
            )
        self.serial = serial
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self._system = None
        self._cam = None

    def open(self) -> None:
        """Connect to the camera (by serial, or the first one found) and configure it.

        Raises:
            CameraError: if no Spinnaker-compatible cameras are detected,
                or the requested serial isn't among them.
        """
        self._system = PySpin.System.GetInstance()
        cam_list = self._system.GetCameras()
        try:
            if cam_list.GetSize() == 0:
                raise CameraError("No Spinnaker-compatible cameras detected.")

            if self.serial:
                cam = cam_list.GetBySerial(self.serial)
                if cam is None:
                    raise CameraError(f"No camera found with serial {self.serial}.")
                self._cam = cam
            else:
                self._cam = cam_list.GetByIndex(0)

            self._cam.Init()
            self._configure()
            self._is_open = True
        finally:
            # GetBySerial/GetByIndex return references that stay valid after
            # the list itself is cleared, so this is safe.
            cam_list.Clear()

    def _configure(self) -> None:
        """Apply acquisition mode, the raw/gamma/white-balance policy, and
        optional exposure/gain. Called by open().
        """
        nodemap = self._cam.GetNodeMap()

        acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        if PySpin.IsWritable(acq_mode):
            continuous = acq_mode.GetEntryByName("Continuous")
            acq_mode.SetIntValue(continuous.GetValue())

        # Force "newest frame only" buffering on the transport-layer stream,
        # matching BaslerCamera's GrabStrategy_LatestImageOnly. Without this,
        # Spinnaker's default buffer handling can leave stale, already-
        # acquired frames queued ahead of GetNextImage() -- so calling
        # capture() right after set_exposure_time() could silently return an
        # old frame from before the exposure change, making it look like
        # changing exposure did nothing even though it worked correctly on
        # the camera itself. Must be set before BeginAcquisition().
        tl_stream_nodemap = self._cam.GetTLStreamNodeMap()
        handling_mode = PySpin.CEnumerationPtr(
            tl_stream_nodemap.GetNode("StreamBufferHandlingMode")
        )
        if PySpin.IsWritable(handling_mode):
            newest_only = handling_mode.GetEntryByName("NewestOnly")
            handling_mode.SetIntValue(newest_only.GetValue())

        # Lab policy (2026-09 meeting): raw sensor data, no on-camera
        # color/tone processing.
        if PySpin.IsAvailable(self._cam.GammaEnable) and PySpin.IsWritable(self._cam.GammaEnable):
            self._cam.GammaEnable.SetValue(False)

        if PySpin.IsAvailable(self._cam.BalanceWhiteAuto) and PySpin.IsWritable(
            self._cam.BalanceWhiteAuto
        ):
            self._cam.BalanceWhiteAuto.SetValue(PySpin.BalanceWhiteAuto_Off)

        # Turning auto white balance off only FREEZES the red/blue balance ratios at
        # whatever they last were (e.g. from an earlier auto run or SpinView); it does
        # not reset them. Set both to 1.0 (neutral) so no colour gain is applied.
        # Added 2026-10-28. Check the result with raw_settings() / check_blackfly_raw.py.
        self.balance_ratio_note = self._set_neutral_balance()

        if PySpin.IsAvailable(self._cam.PixelFormat) and PySpin.IsWritable(self._cam.PixelFormat):
            # Native Bayer mosaic, undemosaiced -- the rawest form the
            # sensor can hand back. Must be set before BeginAcquisition().
            self._cam.PixelFormat.SetValue(PySpin.PixelFormat_BayerRG8)

        if self.exposure_us is not None:
            self._cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            self._cam.ExposureTime.SetValue(self.exposure_us)

        if self.gain_db is not None:
            self._cam.GainAuto.SetValue(PySpin.GainAuto_Off)
            self._cam.Gain.SetValue(self.gain_db)

    def start(self) -> None:
        """Begin acquisition ("turn on"). Must be called after open().

        Raises:
            CameraError: if open() hasn't been called yet.
        """
        if not self._is_open:
            raise CameraError("Camera is not open. Call open() first.")
        self._cam.BeginAcquisition()
        self._is_streaming = True

    def stop(self) -> None:
        """End acquisition ("turn off"). No-op if not currently streaming."""
        if self._is_streaming:
            self._cam.EndAcquisition()
            self._is_streaming = False

    def set_exposure_time(self, exposure_us: float) -> None:
        """Set exposure time in microseconds, switching off auto-exposure.

        Can be called any time after open(), including while streaming --
        use this to adjust exposure without stopping/restarting acquisition.

        Raises:
            CameraError: if open() hasn't been called yet.
        """
        if not self._is_open:
            raise CameraError("Camera is not open. Call open() first.")
        self._cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        self._cam.ExposureTime.SetValue(exposure_us)
        self.exposure_us = exposure_us

    def _set_neutral_balance(self) -> str:
        """Set BalanceRatio to 1.0 for Red and Blue. Never raises: returns a note
        saying what happened, so a camera without these nodes still opens."""
        try:
            sel_node, ratio_node = self._cam.BalanceRatioSelector, self._cam.BalanceRatio
            if not (PySpin.IsAvailable(sel_node) and PySpin.IsWritable(sel_node)):
                return "BalanceRatioSelector not available - ratios left as they were"
            done = []
            for name in ("Red", "Blue"):
                sel_node.SetValue(getattr(PySpin, f"BalanceRatioSelector_{name}"))
                if PySpin.IsWritable(ratio_node):
                    ratio_node.SetValue(1.0)
                    done.append(name)
            return f"BalanceRatio set to 1.0 for {', '.join(done) or 'none'}"
        except Exception as e:  # noqa: BLE001 - keep open() working on any firmware
            return f"could not set BalanceRatio ({e})"

    def raw_settings(self) -> dict:
        """Read back the camera settings that decide whether frames are raw:
        pixel format, gamma, white balance (auto + both ratios), ISP and ADC depth.
        Values that this camera doesn't expose come back as 'n/a'."""
        if not self._is_open:
            raise CameraError("Camera is not open. Call open() first.")
        cam = self._cam

        def enum(node):
            try:
                return node.GetCurrentEntry().GetSymbolic()
            except Exception:  # noqa: BLE001
                return "n/a"

        def value(node):
            try:
                return node.GetValue()
            except Exception:  # noqa: BLE001
                return "n/a"

        out = {
            "PixelFormat": enum(cam.PixelFormat),
            "GammaEnable": value(cam.GammaEnable),
            "BalanceWhiteAuto": enum(cam.BalanceWhiteAuto),
        }
        for name in ("Red", "Blue"):
            try:
                cam.BalanceRatioSelector.SetValue(getattr(PySpin, f"BalanceRatioSelector_{name}"))
                out[f"BalanceRatio_{name}"] = cam.BalanceRatio.GetValue()
            except Exception:  # noqa: BLE001
                out[f"BalanceRatio_{name}"] = "n/a"
        for key in ("IspEnable", "AdcBitDepth", "BlackLevel", "ExposureAuto", "GainAuto", "Gain"):
            node = getattr(cam, key, None)
            out[key] = "n/a" if node is None else (value(node) if key in ("IspEnable", "BlackLevel", "Gain") else enum(node))
        out["note"] = getattr(self, "balance_ratio_note", "")
        return out

    def get_exposure_time(self) -> float:
        """Return the camera's current exposure time, in microseconds.

        Raises:
            CameraError: if open() hasn't been called yet.
        """
        if not self._is_open:
            raise CameraError("Camera is not open. Call open() first.")
        return self._cam.ExposureTime.GetValue()

    def capture(self, save_path: Union[str, Path]) -> Path:
        """Grab the next available frame and save it to save_path.

        Waits up to 5 seconds for a frame. Saves the RAW sensor readout --
        the native Bayer mosaic, with no demosaic/color conversion and no
        gamma or white-balance correction (both disabled in _configure())
        -- as an 8-bit single-channel PNG.

        Raises:
            CameraError: if start() hasn't been called yet, or the grabbed
                frame is incomplete/corrupted.

        Returns:
            The save_path, as a Path object.
        """
        if not self._is_streaming:
            raise CameraError("Camera is not streaming. Call start() first.")

        save_path = Path(save_path)
        image_result = self._cam.GetNextImage(5000)  # 5s timeout
        try:
            if image_result.IsIncomplete():
                raise CameraError(f"Incomplete image: {image_result.GetImageStatus()}")
            # No .Convert() call here on purpose -- that step is what would
            # demosaic/color-correct the frame. Skipping it is what keeps
            # this a raw capture (see module docstring).
            img_array = image_result.GetNDArray()
            self._save_array(img_array, save_path)
            return save_path
        finally:
            image_result.Release()

    def grab_preview(self, timeout_ms: int = 20, downsample: int = 1):
        """Grab the newest frame for a LIVE VIEW, without saving anything.

        Added 2026-10-14 for the GUI's live view; capture() is unchanged.
        Returns the raw 8-bit Bayer mosaic as a 2-D numpy array (same data
        capture() would save, no demosaic here), or None if no new frame
        arrived within `timeout_ms` (normal when the exposure is longer
        than the timeout; just try again). `downsample` is accepted for
        symmetry with BaslerCamera.grab_preview() but ignored: striding a
        Bayer mosaic would mix up its colour filter pattern, so shrink the
        frame AFTER demosaicing instead.

        Raises:
            CameraError: if start() hasn't been called yet, or the camera
                reports a real (non-timeout) error.
        """
        if not self._is_streaming:
            raise CameraError("Camera is not streaming. Call start() first.")
        try:
            image_result = self._cam.GetNextImage(timeout_ms)
        except PySpin.SpinnakerException as e:
            # "No frame yet" is NOT an error here. Spinnaker reports it as
            # SPINNAKER_ERR_TIMEOUT (-1011) with the message "Failed
            # waiting for EventData on NEW_BUFFER_DATA event" -- the word
            # "timeout" is NOT in that text, so match on the code/event
            # name too. (Matching only "timeout" made the live view stop
            # as soon as the exposure outlasted the poll interval,
            # ~50 ms, 2026-10-14.)
            msg = str(e).lower()
            code = getattr(e, "errorcode", None)
            if (
                code == getattr(PySpin, "SPINNAKER_ERR_TIMEOUT", -1011)
                or "timeout" in msg
                or "timed out" in msg
                or "new_buffer_data" in msg
                or "-1011" in msg
            ):
                return None
            raise CameraError(f"Live grab failed: {e}") from e
        try:
            if image_result.IsIncomplete():
                return None
            return image_result.GetNDArray().copy()
        finally:
            image_result.Release()

    @staticmethod
    def _save_array(img_array, save_path: Path) -> None:
        """Write a numpy image array to disk, preferring OpenCV over Pillow."""
        try:
            import cv2
            cv2.imwrite(str(save_path), img_array)
        except ImportError:
            from PIL import Image
            Image.fromarray(img_array).save(save_path)

    def close(self) -> None:
        """Stop streaming if needed, release the camera and the Spinnaker System.

        Safe to call multiple times / after a failed open().
        """
        if self._cam is not None:
            try:
                if self._is_streaming:
                    self.stop()
                self._cam.DeInit()
            finally:
                del self._cam
                self._cam = None
        if self._system is not None:
            self._system.ReleaseInstance()
            self._system = None
        self._is_open = False

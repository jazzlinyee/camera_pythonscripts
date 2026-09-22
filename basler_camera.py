"""
Wrapper for the Basler ace acA5472-17um (USB3, monochrome).

Setup (one-time, on the machine the camera is physically connected to):
    pip install pypylon

    pypylon bundles the pylon runtime, so on most systems no separate SDK
    install is needed. If you hit a "no transport layer" / device-not-found
    error, install the full pylon Camera Software Suite from Basler's site
    instead: https://www.baslerweb.com/en/software/pylon/

    Saving frames uses OpenCV if available, falling back to Pillow:
        pip install opencv-python-headless
        # or
        pip install pillow

Capture policy (set after the 2026-09 meeting):
    Frames are captured at the sensor's true 12-bit depth instead of the
    previous 8-bit default. The camera's PixelFormat is set to "Mono12"
    and the pylon ImageFormatConverter outputs Mono16 (a 16-bit container
    holding the real 0-4095 range), LSB-aligned so the saved pixel values
    are the literal 12-bit sensor counts rather than left-shifted to fill
    16 bits. Frames are written out as .tiff (a format that preserves
    16-bit single-channel data losslessly -- .png would also work, but
    .tiff is what's expected here).

Usage:
    from basler_camera import BaslerCamera

    cam = BaslerCamera(serial="22222222")  # or None to grab the first camera found
    with cam:
        cam.capture("frame_0001.tiff")
        cam.set_exposure_time(20000)  # adjust exposure without stopping the grab loop
        cam.capture("frame_0002.tiff")

Testing without a physical camera:
    pylon has a built-in Camera Emulation transport layer -- set the
    PYLON_CAMEMU environment variable to a number BEFORE this module is
    imported, and pylon will enumerate that many virtual cameras streaming
    synthetic test images. See test_basler_emulated.py for a ready-to-run
    example (no real Basler hardware required). Note: the emulator may not
    support every PixelFormat a real ace camera does -- if it rejects
    "Mono12", that's an emulator limitation, not a bug in this file.
"""

from pathlib import Path
from typing import Optional, Union

from camera_base import Camera, CameraError

try:
    from pypylon import pylon, genicam
except ImportError:
    pylon = None
    genicam = None


class BaslerCamera(Camera):
    """Controls one Basler ace camera via the pylon SDK (pypylon)."""

    def __init__(
        self,
        serial: Optional[str] = None,
        name: str = "acA5472-17um",
        exposure_us: Optional[float] = None,
        gain_db: Optional[float] = None,
    ):
        """Configure (but don't yet connect to) a Basler camera.

        Args:
            serial: Serial number of the camera to use. If None, open()
                will grab whichever device it finds first -- fine with only
                one Basler plugged in, ambiguous with more than one.
            name: Human-readable label, purely for your own logging/prints.
            exposure_us: Manual exposure time in microseconds, applied at
                open(). If None, auto-exposure is left on initially -- use
                set_exposure_time() later to switch to manual. Either way,
                set_exposure_time() can change it again at any point after
                open().
            gain_db: Manual gain in dB. If None, auto-gain is left on.

        Raises:
            CameraError: if pypylon isn't installed.
        """
        super().__init__(name)
        if pylon is None:
            raise CameraError("pypylon is not installed. Run: pip install pypylon")
        self.serial = serial
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self._cam = None
        self._converter = None

    def open(self) -> None:
        """Connect to the camera (by serial, or the first one found) and configure it.

        Raises:
            CameraError: if no cameras are detected, or the requested
                serial isn't among them.
        """
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if not devices:
            raise CameraError("No Basler cameras detected.")

        if self.serial:
            device = next((d for d in devices if d.GetSerialNumber() == self.serial), None)
            if device is None:
                raise CameraError(f"No camera found with serial {self.serial}.")
        else:
            device = devices[0]

        self._cam = pylon.InstantCamera(tl_factory.CreateDevice(device))
        self._cam.Open()
        self._configure()

        # Frames come off the sensor as Mono12 (see _configure()); this
        # converter hands them back as Mono16 so numpy/TIFF tooling can
        # work with them directly, keeping the true 0-4095 pixel values
        # (LsbAligned) rather than left-shifting into the full 16-bit range.
        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_Mono16
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_LsbAligned

        self._is_open = True

    def _configure(self) -> None:
        """Apply the 12-bit pixel-depth policy and optional exposure/gain
        settings. Called by open().

        Raises:
            CameraError: if the camera rejects a setting (wrapped GenICam error).
        """
        try:
            # Lab policy (2026-09 meeting): read out true 12-bit sensor
            # data instead of the previous 8-bit default.
            self._cam.PixelFormat.SetValue("Mono12")

            if self.exposure_us is not None:
                self._cam.ExposureAuto.SetValue("Off")
                self._cam.ExposureTime.SetValue(self.exposure_us)
            if self.gain_db is not None:
                self._cam.GainAuto.SetValue("Off")
                self._cam.Gain.SetValue(self.gain_db)
        except genicam.GenericException as e:
            raise CameraError(f"Failed to configure camera: {e}") from e

    def start(self) -> None:
        """Begin grabbing ("turn on"). Must be called after open().

        Uses LatestImageOnly so unread frames don't pile up in a backlog --
        capture() always gets the most recent one.

        Raises:
            CameraError: if open() hasn't been called yet.
        """
        if not self._is_open:
            raise CameraError("Camera is not open. Call open() first.")
        self._cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self._is_streaming = True

    def stop(self) -> None:
        """Stop grabbing ("turn off"). No-op if not currently streaming."""
        if self._is_streaming:
            self._cam.StopGrabbing()
            self._is_streaming = False

    def set_exposure_time(self, exposure_us: float) -> None:
        """Set exposure time in microseconds, switching off auto-exposure.

        Can be called any time after open(), including while grabbing --
        use this to adjust exposure without stopping/restarting the grab loop.

        Raises:
            CameraError: if open() hasn't been called yet, or the camera
                rejects the value.
        """
        if not self._is_open:
            raise CameraError("Camera is not open. Call open() first.")
        try:
            self._cam.ExposureAuto.SetValue("Off")
            self._cam.ExposureTime.SetValue(exposure_us)
        except genicam.GenericException as e:
            raise CameraError(f"Failed to set exposure time: {e}") from e
        self.exposure_us = exposure_us

    def get_exposure_time(self) -> float:
        """Return the camera's current exposure time, in microseconds.

        Raises:
            CameraError: if open() hasn't been called yet.
        """
        if not self._is_open:
            raise CameraError("Camera is not open. Call open() first.")
        return self._cam.ExposureTime.GetValue()

    def capture(self, save_path: Union[str, Path], save_preview: bool = True) -> Path:
        """Grab the next available frame and save it to save_path.

        Waits up to 5 seconds for a frame. The raw frame is converted to
        Mono16 (holding the true 12-bit sensor values, 0-4095) before
        saving -- pass a .tiff path to preserve that depth losslessly.

        Args:
            save_path: Where to save the raw 12-bit .tiff (or whatever
                format/extension you pass).
            save_preview: If True (default), also saves a brightness-
                stretched 8-bit .png next to save_path (same name with
                "_preview" appended before the extension). Ordinary image
                viewers display 16-bit files against the full 0-65535
                range, so the true 12-bit data (0-4095) can look solid
                black even when well-exposed; this preview exists purely
                so the capture can be checked by eye without extra tools.
                It is for looking at only -- always do real analysis on
                the raw file, since the preview's values are rescaled and
                no longer the true measured intensity.

        Raises:
            CameraError: if start() hasn't been called yet, or the grab fails.

        Returns:
            The save_path, as a Path object.
        """
        if not self._is_streaming:
            raise CameraError("Camera is not streaming. Call start() first.")

        save_path = Path(save_path)
        grab_result = self._cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        try:
            if not grab_result.GrabSucceeded():
                raise CameraError(f"Grab failed: {grab_result.ErrorDescription}")
            image = self._converter.Convert(grab_result)
            img_array = image.GetArray()
            self._save_array(img_array, save_path)
            if save_preview:
                self._save_preview(img_array, save_path)
            return save_path
        finally:
            grab_result.Release()

    @staticmethod
    def _save_array(img_array, save_path: Path) -> None:
        """Write a numpy image array to disk, preferring OpenCV over Pillow."""
        try:
            import cv2
            cv2.imwrite(str(save_path), img_array)
        except ImportError:
            from PIL import Image
            Image.fromarray(img_array).save(save_path)

    @staticmethod
    def _save_preview(img_array, save_path: Path) -> None:
        """Save a brightness-stretched 8-bit .png preview next to save_path.

        The raw file holds true 12-bit values (0-4095) inside a 16-bit
        container -- correct for analysis, but it can look solid black in
        ordinary viewers since they display 16-bit data against the full
        0-65535 range. This preview rescales 0-4095 to 0-255 purely so the
        capture can be checked by eye; it changes nothing about the raw
        file and isn't used for anything downstream.
        """
        import numpy as np

        preview_path = save_path.with_name(save_path.stem + "_preview.png")
        stretched = np.clip(img_array, 0, 4095).astype(np.float32) / 4095.0 * 255
        stretched = stretched.astype(np.uint8)

        try:
            import cv2
            cv2.imwrite(str(preview_path), stretched)
        except ImportError:
            from PIL import Image
            Image.fromarray(stretched).save(preview_path)

    def close(self) -> None:
        """Stop grabbing if needed and close the camera handle.

        Safe to call multiple times / after a failed open().
        """
        if self._cam is not None:
            try:
                if self._is_streaming:
                    self.stop()
                self._cam.Close()
            finally:
                self._cam = None
        self._is_open = False

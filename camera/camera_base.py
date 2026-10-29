"""
Common interface for camera control across vendors.

Every camera wrapper in this project implements the same operations:
    open()                   -- connect to the camera and configure it (does NOT start streaming)
    start()                   -- begin acquisition ("turn on" in the streaming sense)
    stop()                    -- end acquisition ("turn off" in the streaming sense)
    capture(path)             -- grab a single frame and save it to disk
    set_exposure_time(us)     -- change exposure time at runtime (switches auto-exposure off)
    get_exposure_time()       -- read back the camera's current exposure time
    close()                   -- release the camera / SDK resources

Note on "power on/off":
USB3 Vision cameras (both the FLIR Blackfly S and the Basler ace used here)
draw their power from the USB3 connection itself. There is no software
command that cuts power to the sensor while it stays plugged in -- the SDK
only starts/stops the acquisition *stream*. If you need the camera to be
physically powered down between sessions, that has to be done externally
(a switched USB hub, a relay board, etc.).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


class CameraError(RuntimeError):
    """Raised for any camera-related failure (connect, configure, capture, etc.)."""


class Camera(ABC):
    """Abstract base class every vendor-specific camera wrapper implements.

    Subclasses (BlackflyCamera, BaslerCamera) provide the actual SDK calls;
    this class only defines the shared method names/signatures and the
    open->start->capture->stop->close lifecycle, plus `with` support.
    """

    def __init__(self, name: str):
        """Store the camera's display name and initialize state flags.

        Does NOT touch hardware -- call open() to actually connect.
        """
        self.name = name
        self._is_open = False       # True once open() has succeeded
        self._is_streaming = False  # True between start() and stop()

    @abstractmethod
    def open(self) -> None:
        """Connect to the physical camera and configure it.

        Must be called once before start()/capture(). Does NOT begin
        streaming -- it only establishes the connection and applies
        settings like exposure/gain.
        """

    @abstractmethod
    def start(self) -> None:
        """Begin image acquisition ("turn the camera on").

        Must be called after open() and before capture(). Safe to call
        capture() repeatedly after a single start() -- you don't need to
        start/stop between frames.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop image acquisition ("turn the camera off").

        Safe to call even if the camera isn't currently streaming (it's a
        no-op in that case). Does not close/release the camera -- call
        close() for that.
        """

    @abstractmethod
    def capture(self, save_path: Union[str, Path]) -> Path:
        """Grab a single frame and save it to save_path.

        Requires start() to have been called first. Returns the path the
        image was saved to (as a Path object).
        """

    @abstractmethod
    def set_exposure_time(self, exposure_us: float) -> None:
        """Set exposure time in microseconds, switching off auto-exposure.

        This is the exposure-time controller: call it any time after
        open() -- including while streaming, for both cameras in this
        project -- to change exposure without stopping/restarting
        acquisition.
        """

    @abstractmethod
    def get_exposure_time(self) -> float:
        """Return the camera's current exposure time, in microseconds."""

    @abstractmethod
    def close(self) -> None:
        """Release the camera and any SDK resources.

        Stops streaming first if needed. After close(), open() must be
        called again before the camera can be used.
        """

    def __enter__(self):
        """`with cam:` shortcut for open() + start().

        Lets you write: `with BlackflyCamera(...) as cam: cam.capture(...)`
        """
        self.open()
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        """`with` block exit: always stop() then close(), even on error."""
        try:
            self.stop()
        finally:
            self.close()

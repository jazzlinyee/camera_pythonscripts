# Camera Control Scripts

Python wrappers for controlling two lab cameras through one shared interface.

## Files

| File | Purpose |
|---|---|
| `camera_base.py` | Shared abstract `Camera` interface (`open/start/stop/capture/set_exposure_time/get_exposure_time/close`) both wrappers implement. |
| `blackfly_camera.py` | `BlackflyCamera` — wraps the FLIR Blackfly S via PySpin (Spinnaker SDK). |
| `basler_camera.py` | `BaslerCamera` — wraps the Basler ace via pypylon (Pylon SDK). |
| `example_capture.py` | Demo: opens both real cameras, captures 5 frames from each, and demonstrates the exposure-time controller. |
| `test_blackfly_camera.py` | Unit tests with PySpin mocked out — no SDK or hardware needed. |
| `test_basler_emulated.py` | Smoke test against pylon's built-in virtual/emulated camera — no real Basler hardware needed. |

## Hardware

| | FLIR Blackfly S | Basler ace |
|---|---|---|
| Model | BFS-U3-16S2C-CS | acA5472-17um |
| Sensor | Sony IMX273, 1.6MP, color | Sony IMX183, 20MP, monochrome |
| Interface | USB3 Vision | USB3 Vision |
| SDK | Spinnaker / PySpin | Pylon / pypylon |
| Product page | [Edmund Optics](https://www.edmundoptics.com/p/BFS-U3-16S2C-CS-USB3-Blackflyreg-S-Color-Camera/40164/) | [Edmund Optics](https://www.edmundoptics.com/p/basler-ace-aca5472-17um-usb-30-monochrome-camera/40361/) |

Both are USB3 Vision cameras powered over the USB3 cable itself — there's no
software command to cut power while they're plugged in. "Turn on/off" in
this codebase means start/stop the acquisition *stream* (`start()`/`stop()`),
not power. True power-cycling needs external hardware (a switched USB hub,
a relay board, etc.).

## Capture policy (set after the 2026-09 meeting)

| | Blackfly S | Basler ace |
|---|---|---|
| Pixel data | Raw native Bayer mosaic — **no demosaic/color conversion** | True 12-bit sensor readout (`Mono12` → `Mono16`, LSB-aligned so values are literally 0–4095) |
| On-camera processing | Gamma correction **off**, auto white balance **off** | — |
| File format | `.png` (holds the raw single-channel 8-bit mosaic — will not look like a normal color photo when opened, that's expected) | `.tiff` (preserves 16-bit single-channel data losslessly) |

**Open question on the Blackfly side:** `PixelFormat_BayerRG8` is this
sensor's commonly documented native CFA order, but it hasn't been
confirmed against real hardware yet (there's no Blackfly emulator to test
against). Check the `PixelFormat` node's available entries in SpinView
once the camera is connected, and update `blackfly_camera.py`'s
`_configure()` if the real order differs. If what's actually wanted turns
out to be a demosaiced-but-still-uncorrected *color* image rather than the
raw mosaic, that's a one-line change — see the note in that file's module
docstring.

All three settings above are configured in each wrapper's `_configure()`,
so nothing about how you call `open()`/`capture()` changed.

## Exposure-time controller

Both cameras now expose the same two methods for changing exposure at
runtime, without stopping/restarting acquisition:

```python
cam.set_exposure_time(20000)   # microseconds; also switches auto-exposure off
current = cam.get_exposure_time()
```

`exposure_us` passed to the constructor still sets the *initial* exposure
at `open()` time — `set_exposure_time()` is for changing it afterward,
mid-session.

## Platform support — this is NOT Windows-only

Both SDKs run on Windows, Linux, and macOS:

- **pypylon (Basler)** ships prebuilt `pip`-installable wheels for Windows,
  Linux (x86_64 and ARM/aarch64), and macOS (both Intel and Apple Silicon),
  Python 3.9–3.14. `pip install pypylon` just works on any of these.
- **Spinnaker/PySpin (FLIR)** officially supports Windows and Linux most
  fully; macOS is supported too, but through a separate/newer install path
  with narrower camera-family and OS-version coverage. If you're setting
  this up on a Mac, check Teledyne's current macOS Spinnaker SDK notes
  before assuming full parity with the Windows/Linux steps below.

## Setup — once a camera is physically connected

### FLIR Blackfly S
1. Plug into a USB3 (SuperSpeed) port — USB2 ports don't have enough bandwidth.
2. Install the full Spinnaker SDK for your OS from Teledyne's download page
   (free account required):
   https://www.teledynevisionsolutions.com/support/support-center/software-firmware-downloads/
   This includes **SpinView**, a GUI viewer — use it first to confirm the
   camera is detected, preview a live feed, and check the `PixelFormat`
   node's available entries (see the open question above) before touching
   any code.
3. From the same download page, get the **PySpin wheel** matching your OS
   and Python version *exactly* (e.g.
   `spinnaker_python-4.x.x.x-cp310-cp310-linux_x86_64`), then:
   `pip install <wheel file>`
4. Linux only: the Spinnaker installer sets up a udev rule so the camera
   works without `sudo`. Permission errors on `open()` usually mean this
   step needs re-running.
5. Note the camera's **serial number** (on the housing, or in SpinView).

### Basler ace
1. Plug into a USB3 port.
2. `pip install pypylon` (bundles the pylon runtime — usually no separate
   install needed).
3. If `EnumerateDevices()` comes back empty, install the full
   [pylon Camera Software Suite](https://www.baslerweb.com/en/software/pylon/)
   instead — it also includes **pylon Viewer**, the GUI equivalent of SpinView.
4. `pip install opencv-python-headless` (or `pillow`) for saving frames.
5. Note the camera's serial number.

### Run it
Fill in both serial numbers in `example_capture.py` (or leave `serial=None`
if only one of each model is plugged in), then:
```bash
python3 example_capture.py
```
This grabs 5 frames from each camera into `captures/` (Blackfly as raw
`.png`, Basler as 12-bit `.tiff`), changing exposure partway through to
demonstrate the controller. Open a couple of the saved images to
sanity-check exposure/gain before doing anything more involved — remember
the Blackfly PNGs are raw Bayer data, so they'll look like a faint
mosaic/grid pattern rather than a normal color photo; that's expected.

### Troubleshooting
- **"No cameras detected"** — check the cable/port is actually USB3, and
  that the *SDK* (not just the pip package) is installed.
- **Permission errors (Linux)** — re-run the Spinnaker udev config step.
- **`CameraError` from `_configure()`** — usually an exposure/gain value
  outside the sensor's valid range, or (Basler) a `PixelFormat`/`Mono12`
  rejection — see the emulator note in `test_basler_emulated.py`.

## Testing without a physical camera

- `python3 -m unittest test_blackfly_camera -v` — mocks PySpin entirely,
  runs with plain Python 3 + numpy, no SDK or hardware required. Covers
  open/start/capture/stop/close, the exposure-time controller, and the
  gamma/white-balance/pixel-format policy. 19/19 passing as of this change.
- `python3 test_basler_emulated.py` — requires `pip install pypylon` (and
  cv2 or pillow), uses pylon's built-in Camera Emulation (`PYLON_CAMEMU`
  env var) to run against a virtual camera. The emulator may not support
  `Mono12` depending on your pylon version — see that file's docstring.

## API quick reference

Every camera exposes the same methods (see `camera_base.py` and each
wrapper's own docstrings for full detail):

```python
cam = BlackflyCamera(serial="12345678", exposure_us=10000, gain_db=0)
with cam:                        # open() + start()
    cam.capture("frame.png")     # grab + save one frame
    cam.set_exposure_time(20000) # adjust exposure without stopping the stream
    cam.capture("frame2.png")
# stop() + close() happen automatically on exit
```

| Method | Does |
|---|---|
| `open()` | Connect + configure. Does not start streaming. |
| `start()` | Begin acquisition ("turn on"). |
| `capture(path)` | Grab one frame, save it, return the path. |
| `set_exposure_time(us)` | Change exposure at runtime; switches auto-exposure off. |
| `get_exposure_time()` | Read back the current exposure time, in microseconds. |
| `stop()` | End acquisition ("turn off"). Safe to call repeatedly. |
| `close()` | Release the camera/SDK handle. |

## SLM display (`slm_display.py`)

Controls the HOLOEYE amplitude SLM (HES 7020-1 6001) — see project notes
for why this doesn't fit the `Camera` pattern: it has no vendor SDK and is
addressed purely as a second monitor over HDMI (whatever's shown fullscreen
on that display *is* the mask).

| File | Purpose |
|---|---|
| `slm_display.py` | `SLMDisplay` class + `load_masks_from_folder()` helper. |
| `test_slm_display.py` | Mocked tests (cv2 GUI calls + screeninfo) — no display or SDK required. |
| `example_slm_sequence.py` | Demo using generated checkerboard patterns (no real mask files needed to try it). |

**What it does:**
- Loads **multiple masks** (grayscale images, one per pattern) either from a folder (`load_masks_from_folder()`) or as numpy arrays directly.
- `show_mask(index)` — display one mask immediately, under manual control.
- `run_sequence(interval_s, loop=True, cycles=None)` — automatically **cycles through all masks at a set interval** (one number for all masks, or a list of per-mask durations). Blocking call by design — see the "why blocking" note below. Esc/`q` on the window, or `stop_sequence()` called from elsewhere (e.g. a signal handler), stops it early.

**Setup — important packaging conflict:**
The camera scripts use `opencv-python-headless` (fine, since they only *save* files). This module needs to *display* a window, which the headless package cannot do at all. `opencv-python` and `opencv-python-headless` both provide the `cv2` module and **cannot coexist** — if headless is currently installed for the cameras:
```bash
pip uninstall opencv-python-headless
pip install opencv-python
pip install screeninfo
```
The camera scripts' save calls work identically with the non-headless package, so this switch doesn't break them.

**Why `run_sequence()` blocks the calling thread:** cv2's GUI calls must run on the main thread on some platforms (macOS in particular — relevant since this project's dev machine is a Mac). Rather than risk a background-thread GUI crash, `run_sequence()` owns the thread until it's done; run it from its own process if it needs to happen alongside other work.

**Not yet implemented:** the SLM's USB connection (calibration + hardware trigger-sync output, per the HOLOEYE product page) isn't used here — this module only drives the HDMI/display side. Hardware-triggered/camera-synced mask changes (rather than software-timed) would be a separate piece of work on the USB side.

**Testing without a physical SLM:** `python3 -m unittest test_slm_display -v` — mocks cv2's GUI calls and screeninfo's monitor detection, so it runs with no display attached and regardless of which opencv package is installed. 22/22 passing. `load_masks_from_folder()` is tested for real (it only calls `cv2.imread`, which works fine headless).

## Field-testing on real hardware

Two interactive scripts, meant for hands-on bring-up in the lab rather than
automated testing — capture/display, look at the result, adjust, repeat,
all without editing and re-running code each time.

- **`test_on_hardware_cameras.py`** — pick a camera (Blackfly/Basler),
  optional serial, then a `c` / `e <us>` / `g` / `q` prompt loop to
  capture, set exposure, read exposure back, and quit.
- **`test_on_hardware_slm.py [masks_folder]`** — opens the SLM display
  (real masks from a folder, or 4 generated test patterns if you don't
  have real ones ready yet), then an `m <index>` / `r <interval> [cycles]`
  / `q` prompt loop to show individual masks or run a timed sequence.

See each file's own docstring for the full command reference.

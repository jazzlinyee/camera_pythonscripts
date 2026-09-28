# Camera + SLM Control Scripts

Python wrappers and interactive tools for controlling two lab cameras and
a HOLOEYE amplitude SLM through one shared setup.

## Files

| File | Purpose |
|---|---|
| `camera_base.py` | Shared abstract `Camera` interface (`open/start/stop/capture/set_exposure_time/get_exposure_time/close`) both camera wrappers implement. |
| `blackfly_camera.py` | `BlackflyCamera` — wraps the FLIR Blackfly S via PySpin (Spinnaker SDK). |
| `basler_camera.py` | `BaslerCamera` — wraps the Basler ace via pypylon (Pylon SDK). |
| `slm_display.py` | `SLMDisplay` class (Tkinter-based) + `load_masks_from_folder()` helper for the HOLOEYE SLM. |
| `preview_utils.py` | Shared helper (`array_to_photoimage`) that converts a captured/mask array into a Tkinter-displayable thumbnail, used by the GUI. |
| `view_tiff.py` | Standalone viewer/diagnostic for a saved Basler `.tiff` — prints true min/max/mean and auto-stretches it for viewing regardless of alignment convention. |
| `gui_test_camera_slm.py` | **Main interactive tool.** Point-and-click GUI: open/control both cameras, manage SLM mask folders, capture, and run automated Auto Cycle sequences. See its own section below. |
| `test_on_hardware_cameras.py` | REPL: pick a camera, then `c`/`e <us>`/`g`/`q` to capture / set exposure / read exposure / quit. |
| `test_on_hardware_slm.py` | REPL: `m <index>` / `r <interval> [cycles]` / `q` to show a mask, run a timed sequence, or quit. |
| `test_on_hardware_camera_slm.py` | REPL combining the two above into one session. |
| `example_capture.py` | Demo: opens both real cameras, captures 5 frames from each, demonstrates the exposure-time controller. |
| `example_slm_sequence.py` | Demo: cycles generated checkerboard patterns on the SLM — no real mask files needed. |
| `test_blackfly_camera.py` | Unit tests with PySpin mocked out — no SDK or hardware needed. |
| `test_basler_emulated.py` | Smoke test against pylon's built-in virtual/emulated camera — no real Basler hardware needed. |
| `test_slm_display.py` | Mocked SLM tests — **currently out of date**, still mocks the old cv2-window backend (see Known issues below). |
| `mask_folders.json` | Auto-generated — remembers which mask folders you've registered in the GUI across restarts. Not something you edit by hand. |
| `test_captures/` | Auto-generated output folder for captured frames. |
| `test_patterns/` | Auto-generated folder of built-in test masks (checkerboard/white/black/stripes), created the first time the GUI runs without a mask folder specified. |

## Hardware

| | FLIR Blackfly S | Basler ace | HOLOEYE SLM |
|---|---|---|---|
| Model | BFS-U3-16S2C-CS | acA5472-17um | HES 7020-1 6001 |
| Sensor / panel | Sony IMX273, 1.6MP, color | Sony IMX183, 20MP, monochrome | 1920×1080, 8.0 µm pixel pitch, amplitude-only |
| Interface | USB3 Vision | USB3 Vision | HDMI (as an extended-desktop monitor) |
| SDK | Spinnaker / PySpin | Pylon / pypylon | None — addressed as a plain display, no vendor SDK |
| Product page | [Edmund Optics](https://www.edmundoptics.com/p/BFS-U3-16S2C-CS-USB3-Blackflyreg-S-Color-Camera/40164/) | [Edmund Optics](https://www.edmundoptics.com/p/basler-ace-aca5472-17um-usb-30-monochrome-camera/40361/) | [HOLOEYE product line](https://holoeye.com/products/spatial-light-modulators/amplitude-slms/) |

Both cameras are USB3 Vision devices powered over the USB3 cable itself —
there's no software command to cut power while they're plugged in.
"Turn on/off" in this codebase means start/stop the acquisition *stream*
(`start()`/`stop()`), not power. True power-cycling needs external hardware
(a switched USB hub, a relay board, etc.).

The SLM is an *amplitude modulator*, not a light source — it only
attenuates light already passing through it from an external source. It
has no vendor SDK; this project drives it purely by showing a fullscreen
image on it as if it were a second monitor.

---

## Getting started from scratch

This walks through everything needed to go from a fresh machine to a
working setup: get the code, set up Python, install both camera SDKs,
connect the hardware, and confirm it all actually works. Written from a
macOS bring-up (this project's dev machine is a Mac), with Linux/Windows
notes called out where they differ.

### 1. Get the code

```bash
git clone git@github.com:jazzlinyee/camera_pythonscripts.git
cd camera_pythonscripts
```

(Use the HTTPS remote instead if you haven't got an SSH key set up with
GitHub: `git clone https://github.com/jazzlinyee/camera_pythonscripts.git`.)

### 2. Set up a Python environment

**Use Python 3.12 specifically.** The FLIR Spinnaker SDK ships PySpin as
prebuilt wheels tied to exact Python minor versions, and as of the
current Spinnaker release those only go up to `cp312` (Python 3.12) — a
newer interpreter (3.13, 3.14, ...) will not have a matching PySpin wheel
available at all.

Using conda (recommended — easiest way to pin the interpreter version
precisely):
```bash
conda create -n camera python=3.12
conda activate camera
```

Using a plain venv instead, as long as your system already has a 3.12
interpreter available (`python3.12 -m venv .venv`):
```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Either way, confirm before continuing:
```bash
python3 --version                  # should print 3.12.x
python3 -c "import tkinter; print('tkinter OK')"
```
`tkinter` is part of the standard library, but some conda/pyenv Python
builds omit the Tk bindings. If the import fails, try
`conda install -c conda-forge tk`, or rebuild the environment from a
conda-forge Python (which normally bundles Tk support) rather than
trying to add Tk to an existing env after the fact.

### 3. Install the pure-Python dependencies

```bash
pip install numpy opencv-python-headless screeninfo
```
(No `requirements.txt` exists in this repo yet — this is the full list.
`opencv-python-headless` is sufficient: the SLM display itself is drawn
with Tkinter, not cv2's own window, so cv2 is only used here to *load*
mask/capture image files off disk. You do not need the full
non-headless `opencv-python` package.)

### 4. Install the FLIR Blackfly S software (Spinnaker + PySpin)

1. Create a free account and download the **Spinnaker SDK** for your OS
   from Teledyne's site:
   https://www.teledynevisionsolutions.com/support/support-center/software-firmware-downloads/
   This includes **SpinView**, a GUI viewer — install it and use it
   first, before writing or running any code, to confirm the camera is
   detected and to preview a live feed.
2. From the same download page, get the **Spinnaker Python (PySpin)**
   wheel that matches your OS, architecture, **and Python 3.12**
   exactly (the filename encodes all three, e.g. something like
   `spinnaker_python-4.x.x.x-cp312-cp312-macosx_...` on a Mac, or
   `...-cp312-cp312-linux_x86_64` on Linux).
3. **macOS-specific gotcha**: the Spinnaker installer typically places
   the wheel under `/Applications/Spinnaker/PySpin`, which isn't
   writable by a normal (non-sudo) account, and `pip install` from
   there will fail with a permissions error. Copy the `.whl` file
   somewhere in your own home directory first (e.g.
   `~/Downloads/pyspin_extract/`), then install from there:
   ```bash
   pip install ~/Downloads/pyspin_extract/spinnaker_python-*.whl
   ```
4. **macOS-specific native library errors**: if `import PySpin` fails
   with missing-library errors (`libomp.dylib`, `libusb-1.0.0.dylib`,
   `libswscale.7.dylib`, or similar), install their Homebrew equivalents:
   ```bash
   brew install libomp libusb ffmpeg@6
   ```
5. **Linux only**: the Spinnaker installer sets up a udev rule so the
   camera works without `sudo`. Permission errors on `open()` usually
   mean this step needs re-running.
6. Confirm the install:
   ```bash
   python3 -c "import PySpin; print('PySpin OK')"
   ```
7. Note the camera's **serial number** (printed on the housing, or
   visible in SpinView) — you'll be prompted for it (or can leave it
   blank if only one Blackfly is plugged in) by the scripts below.

`macOS` support for Spinnaker/PySpin is real but newer and narrower than
Windows/Linux — if something above doesn't match what you see, check
Teledyne's current macOS Spinnaker release notes rather than assuming
full parity with the Windows/Linux path.

### 5. Install the Basler ace software (pypylon)

```bash
pip install pypylon
```
This bundles the pylon runtime, so on most systems no separate SDK
install is needed. If `EnumerateDevices()` comes back empty later,
install the full [pylon Camera Software Suite](https://www.baslerweb.com/en/software/pylon/)
instead — it also includes **pylon Viewer**, the GUI equivalent of
SpinView, useful for the same "check it's detected before writing code"
step. Note the camera's serial number too.

### 6. Connect the hardware

- **Both cameras**: plug into genuinely USB3 (SuperSpeed) ports — not
  just a USB3-shaped port on a hub/adapter that's actually only wired for
  USB2 underneath. A "device not found" or bandwidth-related error after
  everything above installed cleanly is usually this — try a different
  physical port or a known-good adapter/hub.
- **SLM**: plug the SLM driver unit into an HDMI output as an
  **extended** desktop display (not mirrored), and power it on. Confirm
  your OS's display settings show it as its own ~1920×1080 monitor
  before running anything. If extending genuinely isn't available in
  your setup, `--mirrored` (see the GUI section below) targets the
  primary monitor instead — that's a fallback, not the normal path.

### 7. Verify everything works

Work through these roughly in order — earlier ones need no hardware at
all, later ones need progressively more of it connected.

1. **Imports sane, no hardware needed:**
   ```bash
   python3 -c "import numpy, cv2, screeninfo, tkinter; print('deps OK')"
   python3 -c "import PySpin; print('PySpin OK')"
   python3 -c "import pypylon.pylon; print('pypylon OK')"
   ```
2. **Unit tests, no hardware needed** (mocked SDKs):
   ```bash
   python3 -m unittest test_blackfly_camera -v
   ```
   Should show all tests passing.
3. **Basler emulator smoke test, no real Basler needed:**
   ```bash
   python3 test_basler_emulated.py
   ```
4. **Blackfly + Basler on real hardware:**
   ```bash
   python3 test_on_hardware_cameras.py
   ```
   Pick each camera in turn, `c` to capture, `e <microseconds>` to change
   exposure, `g` to read it back. Open a couple of the files written to
   `test_captures/` and sanity-check them (remember: Blackfly `.png`s are
   raw Bayer mosaic data — a faint grid pattern, not a normal color
   photo; Basler `.tiff`s are MSB-aligned 16-bit — see Capture policy
   below for both).
5. **SLM on real hardware:**
   ```bash
   python3 test_on_hardware_slm.py
   ```
   Confirms the window lands exactly on the SLM's monitor, full-screen,
   and that `m 0`/`m 1`/etc. switch between the built-in test patterns.
6. **Everything together, the actual deliverable:**
   ```bash
   python3 gui_test_camera_slm.py
   ```
   Open both cameras, open the SLM, add a folder of masks (or just use
   the auto-created `test_patterns/`), capture from each camera
   individually and with "Capture Both", and try a short Auto Cycle run.
   See the GUI section below for the full feature rundown.

If every step above works, the environment is fully set up.

---

## Capture policy (set after the 2026-09 meeting, revised 2026-09-22)

| | Blackfly S | Basler ace |
|---|---|---|
| Pixel data | Raw native Bayer mosaic — **no demosaic/color conversion** | True 12-bit sensor readout (`Mono12` → `Mono16`) |
| Bit alignment | N/A (8-bit raw mosaic) | **MSB-aligned** (`true_value * 16`) — see note below |
| On-camera processing | Gamma correction **off**, auto white balance **off** | — |
| File format | `.png` (raw single-channel 8-bit mosaic — will look like a faint mosaic/grid, not a normal color photo; that's expected) | `.tiff` (16-bit single-channel, losslessly preserves the 12-bit reading) |

**Basler alignment note:** values in the saved `.tiff` are the true
12-bit reading multiplied by 16 (e.g. a real reading of 188 is stored as
3008), so that the file displays at roughly correct brightness in an
ordinary viewer without extra stretching. **Any code computing real
intensities from these files must divide by 16 (or bit-shift right by
4) first**, or readings will be 16x too high. `view_tiff.py` prints both
the raw and true values and auto-detects older (pre-2026-09-22)
LSB-aligned files too, if you ever need to open one of those.

**Open question on the Blackfly side:** `PixelFormat_BayerRG8` is this
sensor's commonly documented native CFA order, but hasn't been confirmed
against real hardware yet. Check the `PixelFormat` node's available
entries in SpinView once the camera is connected, and update
`blackfly_camera.py`'s `_configure()` if the real order differs. If a
demosaiced-but-uncorrected *color* image turns out to be what's actually
wanted instead of the raw mosaic, that's a one-line change — see the
note in that file's module docstring.

All settings above are configured in each wrapper's `_configure()`, so
nothing about how you call `open()`/`capture()` changes.

## Exposure-time controller

Both cameras expose the same two methods for changing exposure at
runtime, without stopping/restarting acquisition:

```python
cam.set_exposure_time(20000)   # microseconds; also switches auto-exposure off
current = cam.get_exposure_time()
```

`exposure_us` passed to the constructor still sets the *initial* exposure
at `open()` time — `set_exposure_time()` is for changing it afterward,
mid-session. Because of how each SDK buffers frames, a capture requested
immediately after changing exposure can still reflect the old value for
one frame — both `gui_test_camera_slm.py` and the REPL scripts already
handle this (grab-and-discard one throwaway frame after every change);
any new code that changes exposure and captures back-to-back in a tight
loop should do the same.

## Platform support — this is NOT Windows-only

Both camera SDKs run on Windows, Linux, and macOS:

- **pypylon (Basler)** ships prebuilt `pip`-installable wheels for
  Windows, Linux (x86_64 and ARM/aarch64), and macOS (both Intel and
  Apple Silicon). `pip install pypylon` just works on any of these.
- **Spinnaker/PySpin (FLIR)** officially supports Windows and Linux most
  fully; macOS is supported too, but through a narrower install path —
  see step 4 above.

## SLM display (`slm_display.py`)

Controls the HOLOEYE amplitude SLM by treating it as a plain second
monitor over HDMI — whatever's shown fullscreen on that display *is* the
mask. No vendor SDK is used; the SLM's USB connection (calibration +
hardware trigger-sync) is intentionally out of scope so far.

**Display backend is Tkinter, not cv2** (changed 2026-09-22, after cv2's
own window proved unreliable for exact fullscreen placement on macOS —
see the "Why Tkinter" note in the module's own docstring for the full
story). `cv2` is still used, but only by `load_masks_from_folder()` to
*read* mask image files off disk — `opencv-python-headless` is enough,
no display-capable opencv build is required.

**What it offers:**
- Load masks from a folder or as raw numpy arrays; each is validated
  against the panel's native 1920×1080 resolution.
- `show_mask(index)` — display one mask immediately, non-blocking.
- `run_sequence(interval_s, loop=True, cycles=None)` — cycle through all
  loaded masks at a set interval, blocking the calling thread by design
  (cv2/Tk GUI calls need the main thread on macOS) — run it from its own
  process if it needs to happen alongside other work.
- `add_mask()` / `remove_mask()` / `set_masks()` — change what an
  *already-open* display shows without closing and reopening the window.
  `set_masks()` (wholesale replace) is what the GUI actually uses; the
  other two are additive public API for anyone who wants single-mask
  granularity.

**A dim, non-black "black" mask capture is normal**: real transmissive
LC amplitude panels have finite contrast ratio, so a "black" mask
attenuates rather than fully blocks light, and captures also pick up
ambient light the panel can't touch. This is expected panel behavior,
not a bug.

**Mask-switch settle time is real, not a bug**: the LC panel doesn't
switch instantaneously — especially transitioning toward black — so
capturing immediately after a mask switch can show the previous mask
faintly ghosted in. Pause briefly (the GUI's Auto Cycle "settle" delay
exists specifically for this) before capturing after a switch.

## GUI control panel (`gui_test_camera_slm.py`)

The main interactive tool — a single window covering both cameras and
the SLM together, built on top of the same wrapper classes described
above (no changes to their actual hardware-control logic).

```bash
python3 gui_test_camera_slm.py [--mirrored] [masks_folder]
```
`--mirrored` targets the primary monitor for the SLM instead of the
first extended display (see slm_display.py's note above). An optional
positional argument registers that folder as a mask folder on startup;
otherwise the built-in test patterns are auto-generated into
`test_patterns/` and registered instead.

**Camera panel**: pick Blackfly or Basler, optional serial, open/close,
set/read exposure, capture. Both cameras have independent slots — either
or both can be open at once ("Capture Both" takes one shot from each,
back-to-back, explicitly *not* hardware-synced between the two).

**SLM panel / mask folders**: masks are managed entirely as real folders
on disk, not inside the app. "Add Folder..." registers a folder (via a
native folder picker); "Remove Folder" un-registers one without
touching its files; "Refresh" re-reads every registered folder's
current contents. To add, rename, or delete individual masks, do it
directly in Finder/File Explorer, then hit Refresh — mask names always
come from the filename, with no naming step in the app. Masks display in
a collapsible, multi-select tree grouped by folder; double-clicking one
shows it live on an already-open SLM window. The list of registered
folders is remembered across restarts in `mask_folders.json` (next to
the script) — a folder that's been moved or deleted meanwhile is skipped
and dropped from the list rather than erroring.

**Auto Cycle panel**: automatically switches masks and captures on a
non-blocking timer. "Switch every (ms)" sets the full cycle period,
"Settle before capture (ms)" sets the pause after each switch before
capturing (see the settle-time note above). "Loop through" picks one
registered folder or "All masks" across every folder; folders are
re-scanned fresh each time you hit Start. "Capture both cameras" makes
each tick capture from both cameras instead of just the selected one.
Optionally, "Vary exposure each switch" steps exposure in lockstep with
the mask switches (independently wrapping, so a single-mask loop with
this on is just a plain exposure bracket), in one of two mutually
exclusive modes:
- **Start/step/end sweep** — three number fields (microseconds), stepped
  evenly.
- **Loaded exposure list** ("Load Exposure List..." button) — a plain
  text file, one microsecond value per line (blank lines and `#`
  comments ignored, invalid lines skipped with a count logged). Takes
  precedence over the sweep while loaded; "Clear" drops it and
  re-enables the sweep fields.

**Previews and log**: live preview of the current mask plus the most
recent Blackfly/Basler captures side by side, and a scrolling log panel
at the bottom recording every action.

Single-Tk-root architecture: the main window owns the one `tk.Tk()`; the
SLM's display window is a child `tk.Toplevel` of it, not a second
independent root.

## Field-testing / REPL scripts

Interactive scripts for hands-on bring-up at the bench — capture/display,
look at the result, adjust, repeat — without editing and re-running code
each time.

- **`test_on_hardware_cameras.py`** — pick a camera (Blackfly/Basler),
  optional serial, then a `c`/`e <us>`/`g`/`q` prompt loop.
- **`test_on_hardware_slm.py [--mirrored] [masks_folder]`** — opens the
  SLM display (real masks from a folder, or 4 generated test patterns),
  then an `m <index>`/`r <interval> [cycles]`/`q` prompt loop.
- **`test_on_hardware_camera_slm.py [--mirrored] [masks_folder]`** —
  combines the two into one session (`m`/`c`/`e`/`g`/`q` together).

See each file's own docstring for its full command reference.

## Testing without a physical camera or SLM

- `python3 -m unittest test_blackfly_camera -v` — mocks PySpin entirely,
  runs with plain Python 3 + numpy, no SDK or hardware required. Covers
  open/start/capture/stop/close, the exposure-time controller, and the
  gamma/white-balance/pixel-format policy.
- `python3 test_basler_emulated.py` — uses pylon's built-in Camera
  Emulation (`PYLON_CAMEMU` env var) to run against a virtual camera. The
  emulator may not support `Mono12` depending on your pylon version — see
  that file's docstring.

### Known issues / not yet done

- `test_slm_display.py` predates the 2026-09-22 Tkinter rewrite and still
  mocks the old cv2-window backend — it tests an API that no longer
  exists and needs a rewrite before it's trustworthy again.
- No automated test coverage yet exists for the GUI-layer features
  (Auto Cycle, folder-based masks, dual-camera capture, the exposure
  sweep/list) — they currently only have real-hardware manual
  confirmation, not unit tests.
- The Blackfly `PixelFormat_BayerRG8` assumption above is still
  unconfirmed against real SpinView output.
- The SLM's exact settle time hasn't been characterized in milliseconds
  (the Auto Cycle's "settle" field is a user-tuned guess, not a measured
  value).

## Troubleshooting

- **"No cameras detected"** — check the cable/port is actually USB3
  (SuperSpeed), and that the *SDK* (not just the pip package) is
  installed; try SpinView/pylon Viewer directly to isolate whether it's
  a Python-layer or hardware/driver-layer problem.
- **Permission errors on Linux** — re-run the Spinnaker udev config step.
- **`pip install` of the PySpin wheel fails on macOS with a permissions
  error** — you're likely installing directly from
  `/Applications/Spinnaker/PySpin`; copy the wheel into your own home
  directory first (see step 4 above).
- **`import PySpin` fails with a missing `.dylib` error on macOS** —
  `brew install libomp libusb ffmpeg@6` (see step 4 above).
- **PySpin wheel install fails / no matching wheel found** — your
  Python interpreter isn't 3.12; Spinnaker's current wheels don't go
  past `cp312`. Recreate your environment pinned to Python 3.12.
- **`CameraError` from `_configure()`** — usually an exposure/gain value
  outside the sensor's valid range, or (Basler) a `PixelFormat`/`Mono12`
  rejection — see the emulator note in `test_basler_emulated.py`.
- **Basler `.tiff` looks solid black in a normal viewer** — check
  whether it's an older, pre-2026-09-22 LSB-aligned capture; new
  captures are MSB-aligned and should display at roughly correct
  brightness on their own. `view_tiff.py` auto-detects and handles both.
- **A mask switch still shows the old mask's ghost in the next
  capture** — this is the LC panel's physical settle time, not a bug;
  increase the Auto Cycle "settle" delay (or pause longer manually
  before capturing).
- **SLM window doesn't cover the panel edge-to-edge / shows on the
  wrong monitor** — confirm your OS display settings actually show the
  SLM as an *extended* desktop monitor (not mirrored) before running
  anything; use `--mirrored` only if extending genuinely isn't possible
  in your setup.

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

```python
from slm_display import SLMDisplay, load_masks_from_folder

masks = load_masks_from_folder("masks/")   # sorted list of grayscale arrays
with SLMDisplay(masks) as slm:
    slm.show_mask(0)                       # display one mask, under your own control
    ...
    slm.run_sequence(interval_s=2.0, cycles=3)   # or cycle through all of them automatically
```

## Version control

`.gitignore` excludes OS/build cruft (`.DS_Store`, `__pycache__/`,
`*.pyc`), generated output (`test_captures/`, `test_patterns/`), and
per-machine app state (`mask_folders.json`) — none of that is meant to
be committed. If you're seeing those as tracked/modified in `git status`
anyway, they were committed before the `.gitignore` was added; running
`git rm -r --cached <path>` on them (without `-f`, so the real files on
disk are untouched) removes them from tracking going forward.

# Hyperspectral pipeline: calibrate → capture → cube

One tool that runs the whole coded-aperture pipeline, either from the command line or from
a GUI with a live camera view. It drives the SLM and cameras through the lab's
`camera_pythonscripts`, and processes data with the lab's own notebooks
(`graycode`, `interpolation_wavelengths`, `Reverse_lookup.py`, `meas2hyper`).

**The lab's files are never modified.** Notebooks are read, their hard-coded paths are
swapped in memory, and the patched copy is run. The camera code is imported without writing
`.pyc` files. Everything the pipeline makes goes into `pipeline/sessions/`.

## Folder layout

```
comp vision lab/                  (any name; these folders must sit side by side)
  calibration/                    lab notebooks (git: raalkubaisy/calibration)   (read only)
  camera_pythonscripts/           SLM + camera classes (git: jazzlinyee/camera_pythonscripts)
    pipeline/                     this tool (part of the camera_pythonscripts repo)
      masks/scan_masks/           black + 56 scan masks + scan_plan.csv (copied from slm_scan/)
      masks/test_masks/mask_00.png  the single mask for sparse measurements
      masks/slm_scan.py           the script that made the scan masks
      sessions/<session>/         NOT in git (data; see .gitignore)
        session.json              settings: exposures, lasers, settle time, ...
        calibration/              captures, decoded maps, reverse_lookup.npz (the table), check.json
        scenes/<scene>/           captures/, masks/, manifest.json, cubes/cube_combined.npz, preview.png
```

The paths to the lab folders are set in `settings.py` (`"lab"`) and can be overridden per session
in `session.json`, e.g. with absolute paths if your folders are arranged differently.

### On another computer
1. Clone `camera_pythonscripts` (it includes `pipeline/` and its masks) and, next to it,
   the `calibration` repo.
2. Create the two environments below.
3. `cd camera_pythonscripts/pipeline`, then `python pipeline.py new <session>` and set
   `"processing_python"` in its session.json. Sessions (data) are not in git: copy a session
   folder over if you need its calibration table.

A **session** = one optical setup. Calibrate once per session; capture as many scenes as you
like. If the optics haven't moved, a new session can reuse another session's calibration.

## Setup

Two Python environments are involved:

| Used for | Environment | Needs |
|---|---|---|
| Hardware + GUI (incl. your camera GUI) | the `camera` conda env from `camera_pythonscripts` (Python 3.12) | pypylon, PySpin, screeninfo, tkinter, numpy, opencv-python-headless |
| Processing (notebooks, cubes) | your Anaconda base (where the notebooks already ran) | opencv-contrib-python, scipy, numpy, matplotlib, nbclient, ipykernel |

```
python -m pip install nbclient ipykernel      # in the processing environment, once
```

In `session.json`, set `"processing_python"` to the processing environment's Python
(e.g. `/opt/anaconda3/bin/python`) so the GUI uses it for processing steps.

## Using it

### GUI
```
conda activate camera
cd "comp vision lab/camera_pythonscripts/pipeline"
python pipeline.py gui
```
This opens **one window** with three tabs:

- **1 Calibrate**: one row per laser with its wavelength, exposure, status and its own
  **Capture** button (any order; re-capturing one laser keeps the others). Then **B. Process**
  and **C. Re-run check**. The dark panel at the bottom shows every captured frame, the
  processing steps and one line per table band with time left; it is also saved to
  `sessions/<session>/calibration/calibration_log.txt`. Calibration output is not repeated in
  the pipeline log at the bottom of the window.
- **2 Capture**: **your camera GUI** (`camera_pythonscripts/gui_test_camera_slm.py`,
  unchanged), with its camera panel, SLM mask-folder tree, Auto Cycle, previews, live views and
  log. A pipeline bar on top has the scene name, exposure, a Blackfly checkbox and
  **Capture scene** (all-zero + 56 masks + all-zero). The Gray-code `patterns` and the
  `scan_masks` folders are already in its folder list.
- **3 Cube**: select a scene, press **Make cube**, then **Load cube in viewer** and click the
  image to see spectra.

Steps:
1. **New session…** (or pick one), then **Settings file** to set `processing_python`.
2. Top bar: **Open** the SLM, the Basler (and the Blackfly if you want RGB), or **Open all**.
   These use the camera GUI's own open/close, so the 2 Capture tab shows the same state. An SLM
   opened here starts on the white pattern (added as an extra last mask; the camera GUI's mask
   list is unchanged).
3. **1 Calibrate**, for each laser: turn it on, press **Set exposure** (white pattern on the SLM,
   Live window opens), type exposures until the readout says "good" (bright parts ~70-80%, 0%
   saturated), then press its **Capture** button. The orientation-check frames are taken with
   the laser chosen under "Orientation-check frames with".
4. **2 Capture**: name the scene and press **Capture scene**, in the same sitting with nothing
   moved.
5. Later: **B. Process** on the Calibrate tab, then **3 Cube**.

**Live window ⧉** (top bar) opens a separate, resizable live view. *True brightness* shows what
a saved capture really looks like; *Auto-contrast* stretches it like the camera GUI's live
pane, which is easier to see but hides under-exposure. The readout gives how full the sensor
is ("bright parts" should be about 70–80%) and how much is saturated (shown in red).

While a pipeline capture runs, the camera GUI's live views pause. Its "Current mask" and
preview panes show each mask and frame. Afterwards its masks, live views and exposure are put
back the way they were. Don't use its own Capture or Auto Cycle during a pipeline capture.
Progress and warnings appear in the pipeline log at the bottom of the window.

When run this way, the camera GUI keeps its remembered mask folders and default
`test_captures/` in `pipeline/labgui_state/`, so nothing in `camera_pythonscripts` is written.
Starting it the usual way (`python gui_test_camera_slm.py` from its own folder) is unaffected.

`python pipeline.py gui --standalone` opens the pipeline without your camera GUI. It connects to
the hardware itself and has its own simple Live tab.

### Command line
```
python pipeline.py new 101426                      # then edit sessions/101426/session.json
python pipeline.py calibrate capture 101426        # at the lab: all lasers, asks you to switch
python pipeline.py calibrate capture 101426 --laser Blue   # just one laser
python pipeline.py calibrate process 101426        # ~2 h, then runs the check
python pipeline.py capture 101426 scene1           # at the lab
python pipeline.py cube    101426 scene1           # ~10-20 min
python pipeline.py status  101426
```

## Fully sampled cube (`cube_full.npz`)

The 56 scan masks together open every SLM pixel, so the scan is complete. The NaN holes in
`cube_combined.npz` (about 15–20%, all around the edge) come from the reverse-lookup table,
which has no entry where calibration saw no light. `full_cube.py` makes a cube with a value at
every pixel and band. **Make cube** also writes it automatically (`"cube": {"full": false}`
in session.json turns it off).

1. **Measured (`source = 2`)**: the same maths as meas2hyper, with the calibrated table
   (verified identical to the notebook's output).
2. **Extended table (`source = 1`)**: where the table is NaN, a smooth polynomial fit of the
   table (camera x and y versus SLM x and y, per band) is extended past its edge. Values are
   read from the captures wherever that lands on the sensor. Fit error is about 3 camera px,
   or about half a band.
3. **Filled (`source = 0`)**: light that lands off the sensor can't be measured. Each of these
   takes the value of the nearest measured pixel in the same band.

```
python full_cube.py sessions/101426/scenes/scene1                     # a pipeline scene
python full_cube.py --cubes ../../first_test_cubes                        # merge+fill existing cube_NN.npz
python full_cube.py --zero Z.tiff --captures "caps/*.tiff" --masks "masks/mask_*.png" \
                    --table reverse_lookup.npz --out cube_full.npz     # any captures
```
Output: `cube` (no NaN), `wavelengths`, `source` (per pixel and band), `coverage`, plus
`*_preview.png` and `*_source.png` (white = measured, grey = extended, black = filled).
Use `source` to mask out filled values in any analysis.

## Safety checks built in

- **Orientation check.** After calibration, and again before every cube, it compares how
  the smears actually move when the mask moves against what the table predicts. It catches a
  left-right or up-down flip, or a shifted table, which is what made the 062926 table unusable
  for the October captures. A failed check stops the cube (`--force` overrides).
- **No mixed frames.** After each mask change it waits the settle time, then keeps only a
  frame whose exposure started after that.
- **Saturation warnings** for any frame with clipped pixels.
- **Zero drift.** It compares the all-zero frames from the start and end of a scene capture
  (light level changes).
- **Lab-code guard.** If a lab notebook changes so that the expected file paths can't be found
  exactly, it stops instead of guessing.

## Testing without the lab

`python pipeline.py new test --sim` makes a session that uses a built-in simulator (virtual
SLM, prism and camera). Every stage, the GUI and the checks work the same way. Set
`"sim": {"flip_slm_x": true}` in its session.json to reproduce a left-right flip.

## Notes

- **File formats.** Calibration captures are saved as 8-bit PNG (top 8 of 12 bits), which is
  what `graycode.ipynb` expects. Scene captures are raw 12-bit values in 16-bit TIFF. This is
  independent of `MSB_ALIGN_OUTPUT` in `basler_camera.py`, because frames are read with
  `grab_preview()`, which always returns raw values.
- **QE correction.** `meas2hyper.ipynb` multiplies by the QE curve, and the pipeline runs it
  as written. If the lab changes this, the pipeline picks up the change automatically.
- **Gray-code patterns** are read from `calibration/patterns/` (made by graycode.ipynb's first cell).

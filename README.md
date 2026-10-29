# camera_pythonscripts

Control and processing code for the coded-aperture hyperspectral setup
(HOLOEYE SLM + Basler acA5472 behind the prism + Blackfly RGB guide camera).

```
camera_pythonscripts/
  pipeline/    the main tool: calibrate -> capture -> cube, GUI + command line
               (see pipeline/README.md)
  camera/      SLM + camera drivers and the camera GUI. The pipeline uses these:
               camera_base, basler_camera, blackfly_camera, slm_display,
               gui_test_camera_slm (+ preview_utils, test_on_hardware_slm),
               check_blackfly_raw.py (checks the Blackfly delivers raw frames)
               (see camera/README.md for installing the camera SDKs)
    extras/    examples, unit tests and the TIFF viewer - not needed to run anything
```

The calibration notebooks live in a separate repo (raalkubaisy/calibration), which must sit
next to this one:

```
<lab folder>/
  calibration/            (raalkubaisy/calibration)
  camera_pythonscripts/   (this repo)
```

## Start

```
conda activate camera
cd camera_pythonscripts/pipeline
python pipeline.py gui <session>
```

The camera GUI on its own: `cd camera_pythonscripts/camera` then `python gui_test_camera_slm.py`.

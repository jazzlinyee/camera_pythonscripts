"""
Interactive combined SLM + camera test -- lets you switch which mask the
SLM is showing and capture with a camera from the same running session,
without needing two separate terminals.

This is for hands-on bring-up/troubleshooting (e.g. checking focus and
alignment by switching mask patterns and capturing repeatedly), not a
scripted/automated test. It reuses pick_camera() from
test_on_hardware_cameras.py and make_test_masks() from
test_on_hardware_slm.py so all three test scripts stay in sync (same
camera-picking prompt, same built-in test patterns).

Run:
    python3 test_on_hardware_camera_slm.py [masks_folder]

If masks_folder is given, loads real masks from it (see
load_masks_from_folder()'s docstring for naming/sizing rules). If
omitted, falls back to the same generated test patterns as
test_on_hardware_slm.py (currently checkerboard/white/black/stripes).

Commands at the prompt:
    m <index>     show one mask on the SLM immediately
    c             capture a frame with the camera -- saves to
                  test_captures/, same as test_on_hardware_cameras.py
                  (Basler captures also get an auto-generated
                  brightness-stretched _preview.png alongside the raw
                  12-bit .tiff)
    e <value>     set exposure time in microseconds, e.g. "e 15000"
    g             print the camera's current exposure time
    q             stop/close both the camera and the SLM display, and quit

Note: SLMDisplay.run_sequence()'s automatic mask-cycling isn't offered
here -- it's a blocking call that would prevent interleaving captures.
Use "m <index>" to switch masks manually between captures instead.
"""

import sys
from pathlib import Path

from camera_base import CameraError
from slm_display import SLMDisplay, SLMError, load_masks_from_folder
from test_on_hardware_cameras import pick_camera
from test_on_hardware_slm import make_test_masks

OUTPUT_DIR = Path("test_captures")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    if len(sys.argv) > 1:
        print(f"Loading masks from {sys.argv[1]}...")
        try:
            masks = load_masks_from_folder(sys.argv[1])
        except SLMError as e:
            print(f"Could not load masks: {e}")
            sys.exit(1)
    else:
        print("No folder given -- using the generated test patterns.")
        masks = make_test_masks()
    print(f"{len(masks)} mask(s) loaded.")

    cam, label, ext = pick_camera()

    print(f"Opening {label} camera...")
    try:
        cam.open()
        cam.start()
    except CameraError as e:
        print(f"Failed to open/start camera: {e}")
        sys.exit(1)

    try:
        slm = SLMDisplay(masks)
        slm.open()
    except SLMError as e:
        print(f"Failed to open SLM display: {e}")
        cam.stop()
        cam.close()
        sys.exit(1)

    print(
        "SLM window opened -- check it actually landed on the SLM's "
        "screen, not your laptop display, before continuing."
    )
    print(f"Connected to {label}. Initial exposure: {cam.get_exposure_time()} us")
    print("Commands: m <index>, c=capture, e <us>=set exposure, g=get exposure, q=quit\n")

    shot = 0
    try:
        while True:
            cmd = input("> ").strip()
            if not cmd:
                continue

            if cmd == "q":
                break

            elif cmd.startswith("m "):
                try:
                    slm.show_mask(int(cmd.split(" ", 1)[1]))
                except (ValueError, SLMError) as e:
                    print(f"Could not show mask: {e}")

            elif cmd == "c":
                path = OUTPUT_DIR / f"{label}_{shot:03d}{ext}"
                try:
                    saved = cam.capture(path)
                    print(f"Saved {saved} -- open it to check exposure/focus/framing.")
                    if label == "basler":
                        preview = saved.with_name(saved.stem + "_preview.png")
                        print(f"  (also saved a normal-looking preview: {preview})")
                    shot += 1
                except CameraError as e:
                    print(f"Capture failed: {e}")

            elif cmd == "g":
                try:
                    print(f"Current exposure: {cam.get_exposure_time()} us")
                except CameraError as e:
                    print(f"Could not read exposure: {e}")

            elif cmd.startswith("e "):
                try:
                    value = float(cmd.split(" ", 1)[1])
                    cam.set_exposure_time(value)
                    print(f"Exposure set to {value} us")
                except (ValueError, CameraError) as e:
                    print(f"Could not set exposure: {e}")

            else:
                print("Unknown command. Use: m <index>, c, e <us>, g, or q")
    finally:
        print("Stopping and closing camera and SLM display...")
        cam.stop()
        cam.close()
        slm.close()


if __name__ == "__main__":
    main()

"""
Interactive smoke test for either camera on real hardware -- for hands-on
bring-up in the lab, not a scripted/automated test.

Run:
    python3 test_on_hardware_cameras.py

Then follow the prompts (pick camera, enter serial or leave blank to
auto-detect), and at the command prompt:
    c            capture a frame -- open the saved file to check exposure/
                 focus/framing
    e <value>    set exposure time in microseconds, e.g. "e 15000"
    g            print the camera's current exposure time
    q            stop streaming, close the camera, and quit

This is meant to be used interactively: capture, look at the result, adjust
exposure with "e <value>", capture again -- repeat until it looks right,
without editing and re-running a script each time.
"""

import sys
from pathlib import Path

from basler_camera import BaslerCamera
from blackfly_camera import BlackflyCamera
from camera_base import CameraError

OUTPUT_DIR = Path("test_captures")


def pick_camera():
    print("Which camera do you want to test?")
    print("  1) Blackfly S")
    print("  2) Basler ace")
    choice = input("> ").strip()

    serial = input("Serial number (leave blank to auto-detect): ").strip() or None

    if choice == "1":
        return BlackflyCamera(serial=serial), "blackfly", ".png"
    elif choice == "2":
        return BaslerCamera(serial=serial), "basler", ".tiff"
    else:
        print("Not a valid choice (enter 1 or 2).")
        sys.exit(1)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    cam, label, ext = pick_camera()

    print(f"Opening {label} camera...")
    try:
        cam.open()
        cam.start()
    except CameraError as e:
        print(f"Failed to open/start camera: {e}")
        sys.exit(1)

    print(f"Connected. Initial exposure: {cam.get_exposure_time()} us")
    print("Commands: c=capture, e <us>=set exposure, g=get exposure, q=quit\n")

    shot = 0
    try:
        while True:
            cmd = input("> ").strip()
            if not cmd:
                continue

            if cmd == "q":
                break

            elif cmd == "c":
                path = OUTPUT_DIR / f"{label}_{shot:03d}{ext}"
                try:
                    saved = cam.capture(path)
                    print(f"Saved {saved} -- open it to check exposure/focus/framing.")
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
                print("Unknown command. Use: c, e <us>, g, or q")
    finally:
        print("Stopping and closing camera...")
        cam.stop()
        cam.close()


if __name__ == "__main__":
    main()

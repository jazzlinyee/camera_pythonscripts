"""
Interactive smoke test for the SLM on real hardware -- for hands-on
bring-up in the lab, not a scripted/automated test.

Confirms the display window actually lands on the SLM's monitor (not your
laptop screen) and lets you try different masks/intervals interactively
before committing to a real experiment.

Requires the non-headless opencv-python + screeninfo (see slm_display.py's
module docstring for the package-swap steps) and the SLM connected as an
extended-desktop display (not mirrored).

Run:
    python3 test_on_hardware_slm.py [masks_folder]

If masks_folder is given, loads real masks from it (see
load_masks_from_folder()'s docstring for naming/sizing rules). If omitted,
falls back to 4 generated test patterns (checkerboard, all-white,
all-black, vertical stripes) so you can confirm the display itself
works before you have real mask files ready.

Commands at the prompt:
    m <index>          show one mask immediately
    r <interval> [n]   run all masks in sequence, <interval> seconds each,
                        for n full cycles (omit n to loop until stopped --
                        Esc or 'q' ON THE SLM WINDOW stops it early)
    q                  stop and close
"""

import sys

import numpy as np

from slm_display import SLMDisplay, SLMError, load_masks_from_folder

WIDTH, HEIGHT = 1920, 1080


def make_test_masks():
    """4 simple synthetic patterns so the display can be checked with no real masks yet."""
    black = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    white = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)

    checkerboard = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    block = 64
    for row in range(0, HEIGHT, block):
        for col in range(0, WIDTH, block):
            if ((row // block) + (col // block)) % 2 == 0:
                checkerboard[row : row + block, col : col + block] = 255

    stripes = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    stripes[:, ::20] = 255

    return [checkerboard, white, black, stripes]


def main():
    if len(sys.argv) > 1:
        print(f"Loading masks from {sys.argv[1]}...")
        try:
            masks = load_masks_from_folder(sys.argv[1])
        except SLMError as e:
            print(f"Could not load masks: {e}")
            sys.exit(1)
    else:
        print(
            "No folder given -- using 4 generated test patterns "
            "(checkerboard, white, black, stripes)."
        )
        masks = make_test_masks()

    print(f"{len(masks)} mask(s) loaded.")

    try:
        slm = SLMDisplay(masks)
        slm.open()
    except SLMError as e:
        print(f"Failed to open display: {e}")
        sys.exit(1)

    print(
        "Window opened -- check it actually landed on the SLM's screen, "
        "not your laptop display, before continuing."
    )
    print("Commands: m <index>, r <interval> [cycles], q\n")

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

            elif cmd.startswith("r "):
                parts = cmd.split(" ")
                try:
                    interval = float(parts[1])
                    cycles = int(parts[2]) if len(parts) > 2 else None
                    print("Running -- Esc or 'q' on the SLM window stops early.")
                    slm.run_sequence(interval_s=interval, cycles=cycles)
                except (ValueError, SLMError) as e:
                    print(f"Could not run sequence: {e}")

            else:
                print("Unknown command. Use: m <index>, r <interval> [cycles], or q")
    finally:
        print("Closing display...")
        slm.close()


if __name__ == "__main__":
    main()

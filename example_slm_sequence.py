"""
Example: display a handful of synthetic masks on the SLM, cycling through
them at a fixed interval.

Run with:
    python3 example_slm_sequence.py

Requirements:
    - opencv-python (the GUI-enabled package, NOT opencv-python-headless
      -- see slm_display.py's module docstring for why these conflict)
    - screeninfo
    - The SLM's HDMI output connected and set to extend (not mirror) the
      desktop.

This can't be run in a headless environment (no display attached) or with
opencv-python-headless installed -- both will raise a clear SLMError
rather than hanging, so you'll know immediately if either is the problem.

This uses generated checkerboard/gradient patterns rather than real mask
files so it's runnable the moment the SLM is connected, with no need to
have real mask images ready yet. Swap in load_masks_from_folder("masks/")
once you have actual mask files.
"""

import numpy as np

from slm_display import SLMDisplay

WIDTH, HEIGHT = 1920, 1080


def make_demo_masks():
    """Build 3 simple synthetic 1920x1080 grayscale masks to display."""
    all_off = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    all_on = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)

    checkerboard = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    block = 64
    for row in range(0, HEIGHT, block):
        for col in range(0, WIDTH, block):
            if ((row // block) + (col // block)) % 2 == 0:
                checkerboard[row : row + block, col : col + block] = 255

    return [all_off, checkerboard, all_on]


def main():
    masks = make_demo_masks()

    with SLMDisplay(masks) as slm:
        print(f"Displaying {len(masks)} masks, 2s each, 2 full cycles.")
        print("Press Esc or 'q' on the SLM window to stop early.")
        slm.run_sequence(interval_s=2.0, cycles=2)

    print("Done.")


if __name__ == "__main__":
    main()

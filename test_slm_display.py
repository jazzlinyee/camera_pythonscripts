"""
Unit tests for slm_display.py with cv2's GUI calls and screeninfo mocked out.

Two separate concerns are tested separately:
  - load_masks_from_folder() only ever calls cv2.imread(), which works
    fine even with opencv-python-headless installed (it's not a GUI call)
    -- so those tests use the REAL cv2, writing and reading real small PNGs.
  - SLMDisplay's open()/show_mask()/run_sequence()/close() call cv2's
    window functions (namedWindow, imshow, waitKey, ...), which do NOT
    work headless and also need an actual attached monitor -- those tests
    patch `cv2` and `screeninfo.get_monitors` entirely, so they run
    without a display, without screeninfo installed, and regardless of
    which opencv package is installed.

Run:
    python3 -m unittest test_slm_display.py -v
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import slm_display
from slm_display import SLMDisplay, SLMError, load_masks_from_folder


# ---------------------------------------------------------------------------
# load_masks_from_folder -- uses the real, installed cv2 (headless-safe calls only)
# ---------------------------------------------------------------------------


class LoadMasksFromFolderTests(unittest.TestCase):
    def test_loads_sorted_grayscale_masks(self):
        import cv2  # the real module -- imread/imwrite work under headless too

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            expected = []
            for i, value in enumerate([10, 200, 50]):
                arr = np.full((4, 4), value, dtype=np.uint8)
                cv2.imwrite(str(tmp_path / f"mask_{i:03d}.png"), arr)
                expected.append(value)

            masks = load_masks_from_folder(tmp_path)

            self.assertEqual(len(masks), 3)
            # Sorted by filename -> values should come back in the order written.
            for mask, value in zip(masks, expected):
                self.assertTrue((mask == value).all())

    def test_raises_when_no_files_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SLMError):
                load_masks_from_folder(tmp, pattern="*.png")


# ---------------------------------------------------------------------------
# SLMDisplay -- cv2 GUI + screeninfo mocked, no display/hardware required
# ---------------------------------------------------------------------------


def make_fake_cv2():
    fake = MagicMock(name="cv2")
    fake.WND_PROP_FULLSCREEN = "WND_PROP_FULLSCREEN"
    fake.WINDOW_FULLSCREEN = "WINDOW_FULLSCREEN"
    # Default: "no key pressed" on every waitKey call unless a test overrides it.
    fake.waitKey.return_value = -1
    return fake


def make_monitor(x, y, is_primary):
    return SimpleNamespace(x=x, y=y, is_primary=is_primary)


class SLMDisplayInitTests(unittest.TestCase):
    """__init__ only validates shapes -- no cv2 GUI calls involved, so no mocking needed."""

    def test_raises_on_empty_masks(self):
        with self.assertRaises(SLMError):
            SLMDisplay([])

    def test_raises_on_wrong_size_mask(self):
        bad_mask = np.zeros((10, 10), dtype=np.uint8)
        with self.assertRaises(SLMError):
            SLMDisplay([bad_mask])

    def test_accepts_correctly_sized_masks(self):
        mask = np.zeros((1080, 1920), dtype=np.uint8)  # (height, width)
        slm = SLMDisplay([mask])
        self.assertEqual(len(slm.masks), 1)

    def test_expected_size_none_skips_check(self):
        bad_mask = np.zeros((10, 10), dtype=np.uint8)
        slm = SLMDisplay([bad_mask], expected_size=None)
        self.assertEqual(len(slm.masks), 1)


class SLMDisplayGuiTests(unittest.TestCase):
    def setUp(self):
        self.masks = [np.full((1080, 1920), v, dtype=np.uint8) for v in (0, 128, 255)]

        self.fake_cv2 = make_fake_cv2()
        cv2_patcher = patch.object(slm_display, "cv2", self.fake_cv2)
        cv2_patcher.start()
        self.addCleanup(cv2_patcher.stop)

        self.fake_monitors = [
            make_monitor(0, 0, is_primary=True),
            make_monitor(1920, 0, is_primary=False),
        ]
        self.get_monitors_patcher = patch.object(
            slm_display, "get_monitors", return_value=self.fake_monitors
        )
        self.get_monitors_patcher.start()
        self.addCleanup(self.get_monitors_patcher.stop)

    def _open_display(self, **kwargs):
        slm = SLMDisplay(self.masks, **kwargs)
        slm.open()
        return slm

    # -- open() ----------------------------------------------------------

    def test_open_picks_first_non_primary_monitor_by_default(self):
        slm = self._open_display()
        self.assertTrue(slm._is_open)
        self.fake_cv2.moveWindow.assert_called_once_with(slm.window_name, 1920, 0)
        self.fake_cv2.setWindowProperty.assert_called_once_with(
            slm.window_name, "WND_PROP_FULLSCREEN", "WINDOW_FULLSCREEN"
        )

    def test_open_honors_explicit_monitor_index(self):
        slm = self._open_display(monitor_index=0)
        self.fake_cv2.moveWindow.assert_called_once_with(slm.window_name, 0, 0)

    def test_open_raises_when_no_secondary_monitor(self):
        self.get_monitors_patcher.stop()
        patch.object(
            slm_display, "get_monitors", return_value=[make_monitor(0, 0, True)]
        ).start()
        slm = SLMDisplay(self.masks)
        with self.assertRaises(SLMError):
            slm.open()

    def test_open_raises_when_no_monitors_at_all(self):
        self.get_monitors_patcher.stop()
        patch.object(slm_display, "get_monitors", return_value=[]).start()
        slm = SLMDisplay(self.masks)
        with self.assertRaises(SLMError):
            slm.open()

    # -- show_mask() -------------------------------------------------------

    def test_show_mask_before_open_raises(self):
        slm = SLMDisplay(self.masks)
        with self.assertRaises(SLMError):
            slm.show_mask(0)

    def test_show_mask_displays_correct_array(self):
        slm = self._open_display()
        slm.show_mask(1)
        self.fake_cv2.imshow.assert_called_once_with(slm.window_name, self.masks[1])
        self.assertEqual(slm._current_index, 1)

    def test_show_mask_out_of_range_raises(self):
        slm = self._open_display()
        with self.assertRaises(SLMError):
            slm.show_mask(99)

    # -- run_sequence() ------------------------------------------------------

    def test_run_sequence_shows_every_mask_each_cycle(self):
        slm = self._open_display()
        slm.run_sequence(interval_s=0.001, cycles=2)
        # 3 masks x 2 cycles = 6 imshow calls
        self.assertEqual(self.fake_cv2.imshow.call_count, 6)

    def test_run_sequence_once_through_when_loop_false(self):
        slm = self._open_display()
        slm.run_sequence(interval_s=0.001, loop=False)
        self.assertEqual(self.fake_cv2.imshow.call_count, 3)

    def test_run_sequence_rejects_mismatched_interval_list(self):
        slm = self._open_display()
        with self.assertRaises(SLMError):
            slm.run_sequence(interval_s=[1.0, 2.0])  # only 2 entries for 3 masks

    def test_run_sequence_per_mask_intervals(self):
        slm = self._open_display()
        slm.run_sequence(interval_s=[0.001, 0.002, 0.003], loop=False)
        # Each mask triggers two waitKey calls: show_mask()'s own paint-pump
        # waitKey(1), then run_sequence()'s interval wait. The interval
        # calls are every other entry, starting at index 1.
        called_ms = [call.args[0] for call in self.fake_cv2.waitKey.call_args_list]
        interval_calls = called_ms[1::2]
        self.assertEqual(interval_calls, [1, 2, 3])

    def test_run_sequence_stops_on_q_keypress(self):
        slm = self._open_display()
        # First waitKey (inside show_mask's paint-pump call) returns "no key",
        # second waitKey (the real interval wait) returns 'q'.
        self.fake_cv2.waitKey.side_effect = [-1, ord("q"), -1, -1, -1, -1]
        slm.run_sequence(interval_s=0.001, cycles=5)
        # Should have stopped after the very first mask, not run all 5 cycles.
        self.assertEqual(self.fake_cv2.imshow.call_count, 1)

    def test_stop_sequence_sets_flag(self):
        # run_sequence() resets the stop flag at the start of each call (so
        # a stale stop from a previous run doesn't block a fresh one) --
        # meaning stop_sequence() only has an effect on a run already in
        # progress (e.g. from a signal handler firing mid-loop), which is
        # what the keypress-triggered stop test above already exercises
        # end-to-end. This test just checks the flag itself gets set,
        # without calling the blocking run_sequence() at all.
        slm = self._open_display()
        self.assertFalse(slm._stop_requested)
        slm.stop_sequence()
        self.assertTrue(slm._stop_requested)

    # -- close() / context manager -----------------------------------------

    def test_close_destroys_window(self):
        slm = self._open_display()
        slm.close()
        self.fake_cv2.destroyWindow.assert_called_once_with(slm.window_name)
        self.assertFalse(slm._is_open)

    def test_close_is_safe_to_call_twice(self):
        slm = self._open_display()
        slm.close()
        slm.close()
        self.fake_cv2.destroyWindow.assert_called_once()

    def test_context_manager_opens_and_closes(self):
        with SLMDisplay(self.masks) as slm:
            self.assertTrue(slm._is_open)
            slm.show_mask(0)
        self.fake_cv2.destroyWindow.assert_called_once()


if __name__ == "__main__":
    unittest.main()

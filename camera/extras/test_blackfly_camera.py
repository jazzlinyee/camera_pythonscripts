"""
Unit tests for BlackflyCamera with PySpin fully mocked out.

These tests do NOT require the Spinnaker SDK, PySpin, or a physical
Blackfly camera to be installed/connected -- they run with plain Python 3
and the standard library (plus numpy, for a couple of realistic array
round-trips). They patch the `PySpin` name inside blackfly_camera.py with
a fake object that mimics just enough of the real PySpin API surface to
exercise BlackflyCamera's control flow (open -> start -> capture -> stop
-> close), the exposure-time controller, the raw/gamma/white-balance
capture policy, and error handling.

Run:
    python3 -m unittest test_blackfly_camera.py -v
    # or, if pytest is installed:
    python3 -m pytest test_blackfly_camera.py -v
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # the drivers are in camera/, one folder up


import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import blackfly_camera
from camera_base import CameraError


def make_fake_pyspin():
    """Build a MagicMock that stands in for the PySpin module."""
    fake = MagicMock(name="PySpin")

    # Enum constants used as plain sentinels -- their actual values don't
    # matter, only that they're distinct and consistently returned.
    fake.ExposureAuto_Off = "ExposureAuto_Off"
    fake.GainAuto_Off = "GainAuto_Off"
    fake.BalanceWhiteAuto_Off = "BalanceWhiteAuto_Off"
    fake.PixelFormat_BayerRG8 = "PixelFormat_BayerRG8"
    fake.BalanceRatioSelector_Red = "BalanceRatioSelector_Red"
    fake.BalanceRatioSelector_Blue = "BalanceRatioSelector_Blue"
    fake.BalanceWhiteAuto_Continuous = "BalanceWhiteAuto_Continuous"

    fake.IsWritable.return_value = True
    fake.IsAvailable.return_value = True
    return fake


class BlackflyCameraTests(unittest.TestCase):
    def setUp(self):
        self.fake_pyspin = make_fake_pyspin()
        patcher = patch.object(blackfly_camera, "PySpin", self.fake_pyspin)
        patcher.start()
        self.addCleanup(patcher.stop)

        # Fake system / camera-list / camera objects, wired the way the
        # real Spinnaker API returns them.
        self.fake_system = MagicMock(name="System")
        self.fake_pyspin.System.GetInstance.return_value = self.fake_system

        self.fake_cam = MagicMock(name="Camera")
        self.fake_cam_list = MagicMock(name="CameraList")
        self.fake_cam_list.GetSize.return_value = 1
        self.fake_cam_list.GetByIndex.return_value = self.fake_cam
        self.fake_cam_list.GetBySerial.return_value = self.fake_cam
        self.fake_system.GetCameras.return_value = self.fake_cam_list

        acq_mode_node = MagicMock(name="AcquisitionModeNode")
        acq_mode_node.GetEntryByName.return_value.GetValue.return_value = 1
        self.fake_cam.GetNodeMap.return_value.GetNode.return_value = acq_mode_node
        self.fake_pyspin.CEnumerationPtr.return_value = acq_mode_node

    def _open_camera(self, **kwargs):
        cam = blackfly_camera.BlackflyCamera(**kwargs)
        cam.open()
        return cam

    # -- open() --------------------------------------------------------

    def test_open_picks_first_camera_when_no_serial(self):
        cam = self._open_camera()
        self.assertTrue(cam._is_open)
        self.fake_cam_list.GetByIndex.assert_called_once_with(0)
        self.fake_cam.Init.assert_called_once()
        self.fake_cam_list.Clear.assert_called_once()

    def test_open_picks_by_serial_when_given(self):
        self._open_camera(serial="12345678")
        self.fake_cam_list.GetBySerial.assert_called_once_with("12345678")

    def test_open_raises_when_no_cameras_found(self):
        self.fake_cam_list.GetSize.return_value = 0
        cam = blackfly_camera.BlackflyCamera()
        with self.assertRaises(CameraError):
            cam.open()

    def test_open_raises_when_serial_not_found(self):
        self.fake_cam_list.GetBySerial.return_value = None
        cam = blackfly_camera.BlackflyCamera(serial="nope")
        with self.assertRaises(CameraError):
            cam.open()

    # -- _configure() (called from open()) ------------------------------

    def test_configure_disables_gamma_and_white_balance_sets_raw_format(self):
        self._open_camera()
        self.fake_cam.GammaEnable.SetValue.assert_called_once_with(False)
        self.fake_cam.BalanceWhiteAuto.SetValue.assert_called_once_with(
            "BalanceWhiteAuto_Off"
        )
        self.fake_cam.PixelFormat.SetValue.assert_called_once_with(
            "PixelFormat_BayerRG8"
        )

    def test_configure_sets_white_balance_ratios_to_one(self):
        # Added 2026-10-28: auto white balance off only freezes the ratios,
        # so both are set to a neutral 1.0 explicitly.
        cam = self._open_camera()
        self.assertEqual(
            [c.args[0] for c in self.fake_cam.BalanceRatioSelector.SetValue.call_args_list],
            ["BalanceRatioSelector_Red", "BalanceRatioSelector_Blue"],
        )
        self.assertEqual(
            [c.args[0] for c in self.fake_cam.BalanceRatio.SetValue.call_args_list], [1.0, 1.0]
        )
        self.assertIn("1.0 for Red, Blue", cam.balance_ratio_note)

    def test_open_still_works_if_balance_ratio_cannot_be_set(self):
        self.fake_cam.BalanceRatioSelector.SetValue.side_effect = RuntimeError("no such node")
        cam = self._open_camera()
        self.assertIn("could not set BalanceRatio", cam.balance_ratio_note)
        self.fake_cam.PixelFormat.SetValue.assert_called_once_with("PixelFormat_BayerRG8")

    def test_raw_settings_reads_back_camera_values(self):
        cam = self._open_camera()
        self.fake_cam.PixelFormat.GetCurrentEntry.return_value.GetSymbolic.return_value = "BayerRG8"
        self.fake_cam.GammaEnable.GetValue.return_value = False
        self.fake_cam.BalanceRatio.GetValue.return_value = 1.0
        s = cam.raw_settings()
        self.assertEqual(s["PixelFormat"], "BayerRG8")
        self.assertIs(s["GammaEnable"], False)
        self.assertEqual((s["BalanceRatio_Red"], s["BalanceRatio_Blue"]), (1.0, 1.0))

    def test_set_gamma_turns_it_on_and_off(self):
        cam = self._open_camera()
        self.fake_cam.GammaEnable.SetValue.reset_mock()
        cam.set_gamma(True)
        cam.set_gamma(False)
        self.assertEqual([c.args[0] for c in self.fake_cam.GammaEnable.SetValue.call_args_list], [True, False])

    def test_set_white_balance_on_is_auto_and_off_resets_ratios(self):
        cam = self._open_camera()
        self.fake_cam.BalanceWhiteAuto.SetValue.reset_mock()
        self.fake_cam.BalanceRatio.SetValue.reset_mock()
        cam.set_white_balance(True)
        self.fake_cam.BalanceWhiteAuto.SetValue.assert_called_with("BalanceWhiteAuto_Continuous")
        self.fake_cam.BalanceRatio.SetValue.assert_not_called()
        cam.set_white_balance(False)
        self.fake_cam.BalanceWhiteAuto.SetValue.assert_called_with("BalanceWhiteAuto_Off")
        self.assertEqual([c.args[0] for c in self.fake_cam.BalanceRatio.SetValue.call_args_list], [1.0, 1.0])

    def test_gamma_and_white_balance_need_an_open_camera(self):
        cam = blackfly_camera.BlackflyCamera()
        with self.assertRaises(CameraError):
            cam.set_gamma(False)
        with self.assertRaises(CameraError):
            cam.set_white_balance(False)

    def test_configure_sets_exposure_and_gain_when_given(self):
        self._open_camera(exposure_us=8000, gain_db=3.5)
        self.fake_cam.ExposureAuto.SetValue.assert_called_once_with("ExposureAuto_Off")
        self.fake_cam.ExposureTime.SetValue.assert_called_once_with(8000)
        self.fake_cam.GainAuto.SetValue.assert_called_once_with("GainAuto_Off")
        self.fake_cam.Gain.SetValue.assert_called_once_with(3.5)

    def test_configure_skips_exposure_gain_when_not_given(self):
        self._open_camera()
        self.fake_cam.ExposureAuto.SetValue.assert_not_called()
        self.fake_cam.GainAuto.SetValue.assert_not_called()

    # -- start() / stop() ------------------------------------------------

    def test_start_begins_acquisition(self):
        cam = self._open_camera()
        cam.start()
        self.fake_cam.BeginAcquisition.assert_called_once()
        self.assertTrue(cam._is_streaming)

    def test_start_before_open_raises(self):
        cam = blackfly_camera.BlackflyCamera()
        with self.assertRaises(CameraError):
            cam.start()

    def test_stop_ends_acquisition_only_if_streaming(self):
        cam = self._open_camera()
        cam.start()
        cam.stop()
        self.fake_cam.EndAcquisition.assert_called_once()
        self.assertFalse(cam._is_streaming)

        # Calling stop() again should be a no-op, not a second EndAcquisition call.
        cam.stop()
        self.fake_cam.EndAcquisition.assert_called_once()

    # -- exposure-time controller ----------------------------------------

    def test_set_exposure_time_updates_camera_and_state(self):
        cam = self._open_camera()
        cam.set_exposure_time(12345)
        self.fake_cam.ExposureAuto.SetValue.assert_called_with("ExposureAuto_Off")
        self.fake_cam.ExposureTime.SetValue.assert_called_with(12345)
        self.assertEqual(cam.exposure_us, 12345)

    def test_set_exposure_time_before_open_raises(self):
        cam = blackfly_camera.BlackflyCamera()
        with self.assertRaises(CameraError):
            cam.set_exposure_time(1000)

    def test_get_exposure_time_reads_camera_value(self):
        cam = self._open_camera()
        self.fake_cam.ExposureTime.GetValue.return_value = 9999.0
        self.assertEqual(cam.get_exposure_time(), 9999.0)

    def test_get_exposure_time_before_open_raises(self):
        cam = blackfly_camera.BlackflyCamera()
        with self.assertRaises(CameraError):
            cam.get_exposure_time()

    # -- capture() ---------------------------------------------------------

    def test_capture_before_start_raises(self):
        cam = self._open_camera()
        with self.assertRaises(CameraError):
            cam.capture("frame.png")

    def test_capture_saves_raw_array_with_no_conversion(self):
        cam = self._open_camera()
        cam.start()

        fake_image = MagicMock(name="ImageResult")
        fake_image.IsIncomplete.return_value = False
        fake_image.GetNDArray.return_value = np.zeros((4, 4), dtype=np.uint8)
        self.fake_cam.GetNextImage.return_value = fake_image

        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "frame.png"
            result_path = cam.capture(save_path)

            self.fake_cam.GetNextImage.assert_called_once_with(5000)
            fake_image.Convert.assert_not_called()  # raw passthrough, no demosaic
            fake_image.Release.assert_called_once()
            self.assertEqual(result_path, save_path)
            self.assertTrue(save_path.exists())

    def test_capture_raises_on_incomplete_image_and_still_releases(self):
        cam = self._open_camera()
        cam.start()

        fake_image = MagicMock(name="ImageResult")
        fake_image.IsIncomplete.return_value = True
        fake_image.GetImageStatus.return_value = "some_error_status"
        self.fake_cam.GetNextImage.return_value = fake_image

        with self.assertRaises(CameraError):
            cam.capture("frame.png")
        fake_image.Release.assert_called_once()

    # -- close() / context manager ---------------------------------------

    def test_close_deinits_camera_and_releases_system(self):
        cam = self._open_camera()
        cam.start()
        cam.close()

        self.fake_cam.EndAcquisition.assert_called_once()  # stop() called from close()
        self.fake_cam.DeInit.assert_called_once()
        self.fake_system.ReleaseInstance.assert_called_once()
        self.assertFalse(cam._is_open)

    def test_context_manager_full_lifecycle(self):
        fake_image = MagicMock(name="ImageResult")
        fake_image.IsIncomplete.return_value = False
        fake_image.GetNDArray.return_value = np.zeros((4, 4), dtype=np.uint8)
        self.fake_cam.GetNextImage.return_value = fake_image

        with tempfile.TemporaryDirectory() as tmp:
            with blackfly_camera.BlackflyCamera() as cam:
                cam.capture(Path(tmp) / "frame.png")

        self.fake_cam.Init.assert_called_once()
        self.fake_cam.BeginAcquisition.assert_called_once()
        self.fake_cam.EndAcquisition.assert_called_once()
        self.fake_cam.DeInit.assert_called_once()
        self.fake_system.ReleaseInstance.assert_called_once()


if __name__ == "__main__":
    unittest.main()

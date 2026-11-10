"""
Unit tests for basler_camera.py's settings: auto gain is always turned off at open, and
raw_settings() reads the camera's settings back. pypylon is replaced by a fake, so no
camera or pylon install is needed. Run from camera/extras/:

    python -m unittest test_basler_camera
"""

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # camera/


class FakeGenericException(Exception):
    pass


def install_fake_pypylon():
    pkg = types.ModuleType("pypylon")
    pkg.pylon = mock.MagicMock(name="pylon")
    pkg.genicam = types.SimpleNamespace(GenericException=FakeGenericException)
    sys.modules["pypylon"] = pkg
    sys.modules["pypylon.pylon"] = pkg.pylon
    sys.modules["pypylon.genicam"] = pkg.genicam


install_fake_pypylon()
sys.modules.pop("basler_camera", None)
import basler_camera  # noqa: E402


class FakeNode:
    def __init__(self, value, maximum=None):
        self.value = value
        self.maximum = maximum
        self.set_calls = []

    def GetMax(self):
        if self.maximum is None:
            raise AttributeError("no maximum")
        return self.maximum

    def GetValue(self):
        return self.value

    def SetValue(self, v):
        self.set_calls.append(v)
        self.value = v


class FakeCam:
    """Only the nodes listed exist; any other attribute raises, like a missing GenICam node."""

    def __init__(self, **nodes):
        self.__dict__["_nodes"] = {k: (v if isinstance(v, FakeNode) else FakeNode(v))
                                   for k, v in nodes.items()}

    def __getattr__(self, name):
        try:
            return self.__dict__["_nodes"][name]
        except KeyError:
            raise AttributeError(name) from None

    def GetDeviceInfo(self):
        return types.SimpleNamespace(GetModelName=lambda: "acA5472-17um",
                                     GetSerialNumber=lambda: "12345678")


def full_cam(**over):
    nodes = dict(PixelFormat="Mono8", ExposureAuto="Off", ExposureTime=10000.0,
                 GainAuto="Continuous", Gain=6.0, BlackLevel=0.0,
                 Width=FakeNode(2000, maximum=5472), Height=FakeNode(1000, maximum=3648),
                 OffsetX=120, OffsetY=64, BinningHorizontal=2, BinningVertical=2,
                 CenterX=True, CenterY=False,
                 ReverseX=False, ReverseY=False, DeviceTemperature=41.5,
                 AcquisitionFrameRateEnable=False, AcquisitionFrameRate=17.0,
                 ResultingFrameRate=17.9)
    nodes.update(over)
    return FakeCam(**nodes)


def make(cam, **kw):
    b = basler_camera.BaslerCamera(**kw)
    b._cam = cam
    return b


class GainTests(unittest.TestCase):
    def test_default_turns_auto_gain_off_at_zero_db(self):
        cam = full_cam()
        make(cam)._configure()
        self.assertEqual(cam.GainAuto.set_calls, ["Off"])
        self.assertEqual(cam.Gain.set_calls, [0.0])
        self.assertEqual(cam.PixelFormat.value, "Mono12")

    def test_given_gain_is_used(self):
        cam = full_cam()
        make(cam, gain_db=3.5)._configure()
        self.assertEqual(cam.Gain.set_calls, [3.5])

    def test_gain_none_means_zero_db(self):
        cam = full_cam()
        make(cam, gain_db=None)._configure()
        self.assertEqual(cam.GainAuto.set_calls, ["Off"])
        self.assertEqual(cam.Gain.set_calls, [0.0])

    def test_missing_gain_nodes_do_not_stop_open(self):
        cam = FakeCam(PixelFormat="Mono8")
        b = make(cam)
        b._configure()                       # must not raise
        self.assertIn("GainAuto", b.gain_note)
        self.assertIn("Gain", b.gain_note)

    def test_exposure_untouched_when_not_given(self):
        cam = full_cam()
        make(cam)._configure()
        self.assertEqual(cam.ExposureAuto.set_calls, [])


class FullSensorTests(unittest.TestCase):
    def test_open_resets_a_cropped_offset_roi_to_full_sensor(self):
        cam = full_cam()
        b = make(cam)
        b._configure()
        self.assertEqual((cam.OffsetX.value, cam.OffsetY.value), (0, 0))
        self.assertEqual((cam.Width.value, cam.Height.value), (5472, 3648))
        self.assertEqual((cam.BinningHorizontal.value, cam.BinningVertical.value), (1, 1))
        self.assertEqual(cam.CenterX.value, False)
        self.assertEqual(b.roi_note, "")

    def test_offsets_set_before_width(self):
        order = []
        cam = full_cam()
        for name in ("OffsetX", "Width"):
            node = getattr(cam, name)
            orig = node.SetValue
            node.SetValue = (lambda o, n: (lambda v: (order.append(n), o(v))))(orig, name)
        make(cam)._configure()
        self.assertEqual(order, ["OffsetX", "Width"])

    def test_missing_roi_nodes_do_not_stop_open(self):
        b = make(FakeCam(PixelFormat="Mono8"))
        b._configure()                       # must not raise; missing nodes are skipped
        self.assertEqual(b.roi_note, "")


class RawSettingsTests(unittest.TestCase):
    def test_reads_settings_back(self):
        cam = full_cam()
        b = make(cam)
        b._configure()
        b._is_open = True
        s = b.raw_settings()
        self.assertEqual(s["Model"], "acA5472-17um")
        self.assertEqual(s["Serial"], "12345678")
        self.assertEqual(s["PixelFormat"], "Mono12")
        self.assertEqual(s["GainAuto"], "Off")
        self.assertEqual(s["Gain"], 0.0)
        self.assertEqual(s["ExposureTime"], 10000.0)
        self.assertEqual(s["DeviceTemperature"], 41.5)
        self.assertEqual((s["OffsetX"], s["OffsetY"], s["Width"], s["Height"]), (0, 0, 5472, 3648))
        self.assertIn("x16", s["SavedAs"])
        self.assertEqual(s["note"], "")

    def test_missing_nodes_are_na(self):
        b = make(FakeCam(PixelFormat="Mono12"))
        b._is_open = True
        s = b.raw_settings()
        self.assertEqual(s["Gamma"], "n/a")
        self.assertEqual(s["DeviceTemperature"], "n/a")

    def test_needs_open_camera(self):
        b = make(full_cam())
        with self.assertRaises(basler_camera.CameraError):
            b.raw_settings()


if __name__ == "__main__":
    unittest.main()

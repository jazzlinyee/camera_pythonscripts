"""
Simulated SLM + prism + cameras, for testing the pipeline without the lab.

Turn on with  "hardware": "sim"  in session.json. Optional "sim" settings:
    "sim": {"scale": 0.6, "flip_slm_x": false, "noise": 4.0}
flip_slm_x mirrors what the SLM shows (reproduces the left-right flip problem, so the
orientation check can be tested). The camera is smaller than the real Basler
(912 x 1368) so a full calibration runs in minutes.

Optics model: SLM pixel (x, y) at wavelength L lands on camera pixel
    u = scale*x + 60 + D(L),   v = scale_y*y + 110 + 3*((L-550)/150)^2
with D(L) following the N-BK7 refractive index (the same model the interpolation
notebook fits), so the lab notebooks can calibrate it exactly.
"""

import time

import cv2
import numpy as np

SLM_H, SLM_W = 1080, 1920
CAM_H, CAM_W = 912, 1368
WL = np.arange(400, 701, 10).astype(float)
LASERS = {"Blue": 405.0, "Green": 532.0, "Red": 635.0}


def n_bk7(lam_nm):
    L2 = (np.asarray(lam_nm, float) / 1000.0) ** 2
    return np.sqrt(1 + 1.03961212 * L2 / (L2 - 0.00600069867) + 0.231792344 * L2 / (L2 - 0.0200179144)
                   + 1.01046945 * L2 / (L2 - 103.560653))


class Optics:
    def __init__(self, scale=0.6, disp_px=40.0):
        self.sx, self.sy = scale, scale * 1.02
        self.disp_px = disp_px
        self.n400, self.n700 = n_bk7(400), n_bk7(700)
        vv, uu = np.mgrid[0:CAM_H, 0:CAM_W].astype(np.float32)
        self.uu, self.vv = uu, vv

    def D(self, lam):  # dispersion offset in camera px (decreasing n -> increasing u)
        return self.disp_px * (self.n400 - n_bk7(lam)) / (self.n400 - self.n700)

    def tilt(self, lam):
        return 3.0 * ((lam - 550.0) / 150.0) ** 2

    def cam_to_slm(self, lam):
        """For every camera pixel, the SLM (x, y) it sees at wavelength lam."""
        x = (self.uu - 60.0 - self.D(lam)) / self.sx
        y = (self.vv - 110.0 - self.tilt(lam)) / self.sy
        return x.astype(np.float32), y.astype(np.float32)


class Scene:
    """A synthetic hyperspectral scene on the SLM grid: smooth patches with distinct spectra."""

    def __init__(self, seed=1):
        rng = np.random.default_rng(seed)
        self.spectra = []
        for c in [450, 520, 580, 630, 680, 500]:
            s = np.exp(-0.5 * ((WL - c) / 35.0) ** 2) * 0.8 + 0.2 * rng.random()
            self.spectra.append(s)
        self.spectra = np.array(self.spectra, np.float32)              # (6, 31)
        yy, xx = np.mgrid[0:SLM_H, 0:SLM_W]
        self.labels = ((xx // 320) + 3 * (yy // 540)).astype(np.int32) % len(self.spectra)
        bright = 0.6 + 0.4 * np.cos(xx / 400.0) * np.cos(yy / 300.0)
        self.bright = bright.astype(np.float32)

    def band(self, b):
        return self.spectra[self.labels, b] * self.bright             # (SLM_H, SLM_W)

    def truth(self, y, x):
        return self.spectra[self.labels[y, x]] * self.bright[y, x]


class SimRig:
    def __init__(self, session, log=print):
        self.session = session
        self.log = log
        sim = session.settings.get("sim", {})
        self.optics = Optics(scale=sim.get("scale", 0.6))
        self.flip = bool(sim.get("flip_slm_x", False))
        self.noise = float(sim.get("noise", 4.0))
        self.scene = Scene()
        self.masks, self.cur = [np.zeros((SLM_H, SLM_W), np.uint8)], 0
        self.exposure_us = 100000.0
        self.light = "scene"            # "scene" or a laser folder name ("Blue"/"Green"/"Red")
        self.rng = np.random.default_rng(0)
        self._maps = {}
        self._cache = None

    # interface ----------------------------------------------------------------
    def open(self, blackfly=True):
        self.log("Simulator: SLM + cameras ready (no hardware).")

    def close(self):
        pass

    def load_masks(self, masks):
        self.masks = [np.asarray(m) for m in masks]
        self.cur = 0
        self._cache = None

    def show(self, i):
        self.cur = i
        self._cache = None

    def set_exposure(self, us):
        self.exposure_us = float(us)
        self._cache = None

    def set_light(self, light):
        """Simulator only: which light is on (the real rig asks the user to switch lasers)."""
        self.light = light
        self._cache = None

    def grab_basler(self):
        time.sleep(0.01)
        return self._render()

    def grab_blackfly(self):
        rgb = np.stack([self.scene.band(b) for b in (25, 15, 5)], -1)  # B,G,R order for BGR
        rgb = cv2.resize(rgb, (720, 540)) * 200
        bayer = np.zeros((540, 720), np.float32)                       # RG Bayer mosaic
        bayer[0::2, 0::2] = rgb[0::2, 0::2, 2]
        bayer[0::2, 1::2] = rgb[0::2, 1::2, 1]
        bayer[1::2, 0::2] = rgb[1::2, 0::2, 1]
        bayer[1::2, 1::2] = rgb[1::2, 1::2, 0]
        return bayer.clip(0, 255).astype(np.uint8)

    def live_frame(self, downsample=8):
        return self._render()[::downsample, ::downsample]

    # rendering ------------------------------------------------------------------
    def _map(self, lam):
        if lam not in self._maps:
            self._maps[lam] = self.optics.cam_to_slm(lam)
        return self._maps[lam]

    def _displayed(self):
        m = self.masks[self.cur].astype(np.float32) / 255.0
        return m[:, ::-1].copy() if self.flip else m

    def _sample(self, img, lam):
        x, y = self._map(lam)
        return cv2.remap(img, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _render(self):
        if self._cache is None:
            m = self._displayed()
            k = self.exposure_us / 100000.0
            if self.light in LASERS:
                img = 2500.0 * k * self._sample(m, LASERS[self.light]) + 60.0
            else:
                img = np.zeros((CAM_H, CAM_W), np.float32)
                for b, lam in enumerate(WL):
                    img += self._sample(m * self.scene.band(b), lam)
                img = 900.0 * k * img
                # reflection: a blurred, unmasked image of the scene (present even with an all-zero mask)
                refl = cv2.GaussianBlur(self._sample(self.scene.band(15), 550.0), (0, 0), 3)
                img += 1200.0 * k * refl + 80.0
            self._cache = img
        noisy = self._cache + self.rng.normal(0, self.noise, self._cache.shape)
        return noisy.clip(0, 4095).astype(np.uint16)

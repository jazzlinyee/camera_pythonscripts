"""
Hardware access for the pipeline: SLM + Basler (smear camera) + Blackfly (RGB guide camera).

RealRig drives the lab hardware through the lab's own camera_pythonscripts classes
(imported read-only). SimRig (simulator.py) has the same interface and needs no hardware.

Interface used by the stages and the GUI:
    rig.open(blackfly=True)          rig.close()
    rig.load_masks([arrays...])      rig.show(i)          # SLM; call from the main (Tk) thread
    rig.set_exposure(us)             rig.exposure_us
    rig.grab_basler()  -> uint16 raw 12-bit frame taken entirely AFTER the last show()
    rig.grab_blackfly()-> raw Bayer (RG) uint8 frame, same guarantee
    rig.live_frame(downsample) -> latest Basler frame (raw 12-bit) or None, for live view
"""

import threading
import time

import numpy as np

from labcode import import_camera_code


class RigError(RuntimeError):
    pass


def make_rig(session, tk_master=None, log=print):
    if session.settings.get("hardware") == "sim":
        from simulator import SimRig
        return SimRig(session, log=log)
    return RealRig(session, tk_master=tk_master, log=log)


class _Base:
    # Extra wait after the exposure, for readout/transfer before a frame is returned.
    READOUT_MARGIN_S = 0.15

    def __init__(self, session, log=print):
        self.session = session
        self.log = log
        self.settle_s = session.settings["settle_ms"] / 1000.0
        self.exposure_us = None
        self._t_show = 0.0
        self._lock = threading.Lock()

    def _wait_clean(self, grab_fn, exposure_us, timeout_s=None):
        """Return the first frame whose exposure began after the SLM had settled.

        A frame returned at time t started exposing at about t - exposure. Frames
        that might have started before (show time + settle) are thrown away -- this
        is the same stale-frame protection the lab GUI uses, made exposure-aware.
        """
        exp_s = (exposure_us or 0) / 1e6
        earliest_start = self._t_show + self.settle_s
        wait = earliest_start - time.time()
        if wait > 0:
            time.sleep(wait)
        deadline = time.time() + (timeout_s or (3 * exp_s + 10))
        discarded = 0
        while time.time() < deadline:
            frame = grab_fn(int(max(exp_s * 2 * 1000, 0) + 1000))
            if frame is None:
                continue
            t_done = time.time()
            if t_done - exp_s - self.READOUT_MARGIN_S >= earliest_start and discarded >= 1:
                return frame
            discarded += 1
        raise RigError("Timed out waiting for a camera frame.")


class RealRig(_Base):
    def __init__(self, session, tk_master=None, log=print):
        super().__init__(session, log)
        self.tk_master = tk_master
        self.mods = None
        self.slm = self.basler = self.blackfly = None
        self.blackfly_exposure_us = session.settings["blackfly"]["exposure_us"]

    # ------------------------------------------------------------- open/close
    def open(self, blackfly=True):
        s = self.session.settings
        self.mods = import_camera_code(self.session.lab_path("camera_code"))
        slm_mod = self.mods["slm_display"]
        black = np.zeros((1080, 1920), np.uint8)
        self.slm = slm_mod.SLMDisplay([black], monitor_index=s["slm"]["monitor_index"],
                                      mirrored=s["slm"]["mirrored"], master=self.tk_master)
        self.slm.open()
        self.log("SLM open.")
        B = self.mods["basler_camera"].BaslerCamera
        self.basler = B(serial=s["basler"]["serial"], gain_db=s["basler"]["gain_db"])
        self.basler.open()
        self.basler.start()
        self.log("Basler open.")
        if blackfly and s["blackfly"]["enabled"]:
            F = self.mods["blackfly_camera"].BlackflyCamera
            self.blackfly = F(serial=s["blackfly"]["serial"], gain_db=s["blackfly"]["gain_db"],
                              exposure_us=s["blackfly"]["exposure_us"])
            self.blackfly.open()
            self.blackfly.start()
            self.log("Blackfly open.")

    def close(self):
        for dev in (self.basler, self.blackfly):
            if dev is not None:
                try:
                    dev.stop()
                    dev.close()
                except Exception as e:  # noqa
                    self.log(f"(close: {e})")
        if self.slm is not None:
            try:
                self.slm.close()
            except Exception:
                pass
        self.slm = self.basler = self.blackfly = None

    # ------------------------------------------------------------- SLM
    def load_masks(self, masks):
        self.slm.set_masks([np.ascontiguousarray(m, dtype=np.uint8) for m in masks])

    def show(self, i):
        self.slm.show_mask(i)
        self._t_show = time.time()

    # ------------------------------------------------------------- cameras
    def set_exposure(self, us):
        with self._lock:
            self.basler.set_exposure_time(float(us))
            self.exposure_us = float(us)

    def grab_basler(self):
        with self._lock:
            return self._wait_clean(lambda t: self.basler.grab_preview(timeout_ms=t, downsample=1),
                                    self.exposure_us)

    def grab_blackfly(self):
        if self.blackfly is None:
            return None
        return self._wait_clean(lambda t: self.blackfly.grab_preview(timeout_ms=t, downsample=1),
                                self.blackfly_exposure_us)

    def live_frame(self, downsample=8):
        if not self._lock.acquire(blocking=False):
            return None          # a capture is using the camera
        try:
            t = int(min(2000, (self.exposure_us or 0) / 1000 * 1.5 + 50))   # long exposures need a longer wait
            return self.basler.grab_preview(timeout_ms=t, downsample=downsample)
        finally:
            self._lock.release()


class LabGuiRig(_Base):
    """Uses the SLM + cameras that are open in the lab's camera GUI
    (camera_pythonscripts/gui_test_camera_slm.py, App), so the pipeline runs inside
    the familiar window. The lab GUI keeps ownership of the devices: this never opens
    or closes them.

    During a pipeline sequence (begin() ... end(), both on the Tk thread):
      * the lab GUI's live views are paused (so two things never grab at once),
      * the current mask and each captured frame are shown in its preview panes,
      * afterwards its own mask list, live views and Basler exposure are restored.
    """

    def __init__(self, session, lab, labmod, log=print):
        super().__init__(session, log)
        self.lab, self.labmod = lab, labmod
        self._saved = None
        self._masks = []

    @property
    def basler(self):
        return self.lab.cams.get("basler")

    @property
    def blackfly(self):
        return self.lab.cams.get("blackfly")

    def problems(self):
        p = []
        if self.lab.slm is None:
            p.append("open the SLM (top bar: SLM → Open)")
        if self.basler is None:
            p.append("open the Basler (top bar: Basler → Open)")
        if getattr(self.lab, "_auto_running", False):
            p.append("stop Auto Cycle (2 Capture tab)")
        return p

    def open(self, blackfly=True):
        p = self.problems()
        if p:
            raise RigError("First " + "; ".join(p) + ".")
        self.exposure_us = float(self.basler.get_exposure_time())
        self.blackfly_exposure_us = float(self.blackfly.get_exposure_time()) if self.blackfly else None

    def close(self):
        pass                        # devices belong to the lab GUI

    # -- sequence bracket (Tk thread) ------------------------------------------
    def begin(self):
        p = self.problems()
        if p:
            raise RigError("First " + "; ".join(p) + ".")
        lab = self.lab
        self._saved = {"live": {k: bool(v.get()) for k, v in lab.live_vars.items()},
                       "exposure": float(self.basler.get_exposure_time())}
        for k, on in self._saved["live"].items():
            if on:
                lab._stop_live(k, reset_checkbox=True)
        lab.log("Pipeline sequence running - live views paused.")

    def end(self):
        lab, saved = self.lab, self._saved or {}
        try:
            if lab.slm is not None and getattr(lab, "_showing_white", False) and hasattr(lab, "show_white"):
                lab.show_white()                  # it was on white before the sequence
            elif lab.slm is not None and lab.masks:
                lab.slm.set_masks(lab.masks)
            lab._render_previews(["mask"])
        except Exception as e:  # noqa
            self.log(f"(restoring masks: {e})")
        try:
            if "exposure" in saved and self.basler is not None:
                self.basler.set_exposure_time(saved["exposure"])
                self.exposure_us = saved["exposure"]
                lab.refresh_exposure()
        except Exception as e:  # noqa
            self.log(f"(restoring exposure: {e})")
        for k, on in saved.get("live", {}).items():
            if on and lab.cams.get(k) is not None:
                lab.live_vars[k].set(True)
                lab._on_live_toggle(k)
        lab.log("Pipeline sequence finished - masks, exposure and live views restored.")
        self._saved = None

    # -- SLM (Tk thread) ---------------------------------------------------------
    def load_masks(self, masks):
        self._masks = [np.ascontiguousarray(m, dtype=np.uint8) for m in masks]
        self.lab.slm.set_masks(self._masks)

    def show(self, i):
        self.lab.slm.show_mask(i)
        self._t_show = time.time()
        try:
            w, h = self.lab.preview_sizes["mask"]
            ph = self.labmod.array_to_photoimage(self._masks[i], w, h)
            self.lab._mask_thumb_photo = ph
            self.lab.mask_preview_label.configure(image=ph)
        except Exception:
            pass

    # -- cameras (worker thread) --------------------------------------------------
    def set_exposure(self, us):
        with self._lock:
            self.basler.set_exposure_time(float(us))
            self.exposure_us = float(us)
        self.lab.root.after(0, self.lab.refresh_exposure)

    def _preview(self, choice, frame):
        """Show a frame the pipeline just grabbed in the camera GUI's capture pane
        (its live panes are hidden inside the pipeline)."""
        def show():
            try:
                if choice == "basler":
                    # true brightness, no stretching: raw 0-4095 -> 0-255 (what was really recorded)
                    f = np.clip(frame[::8, ::8], 0, 4095).astype(np.float32)
                    disp = (f * (255.0 / 4095.0)).astype(np.uint8)
                else:
                    disp = self.lab._live_display_frame(choice, frame)      # Bayer -> colour only
                ph = self.labmod.array_to_photoimage(disp, *self.lab.preview_sizes[choice])
                if choice == "basler":
                    self.lab._basler_thumb_photo = ph
                    self.lab.basler_preview_label.configure(image=ph)
                    self.lab.basler_path_label.configure(text="(pipeline capture - true brightness)")
                else:
                    self.lab._blackfly_thumb_photo = ph
                    self.lab.blackfly_preview_label.configure(image=ph)
                    self.lab.blackfly_path_label.configure(text="(pipeline capture)")
            except Exception:
                pass
        self.lab.root.after(0, show)

    def grab_basler(self):
        with self._lock:
            f = self._wait_clean(lambda t: self.basler.grab_preview(timeout_ms=t, downsample=1), self.exposure_us)
        self._preview("basler", f)
        return f

    def grab_blackfly(self):
        if self.blackfly is None:
            return None
        f = self._wait_clean(lambda t: self.blackfly.grab_preview(timeout_ms=t, downsample=1),
                             self.blackfly_exposure_us)
        self._preview("blackfly", f)
        return f

    def live_frame(self, downsample=8):
        return None                 # the lab GUI has its own live view


# ---------------------------------------------------------------- image helpers
def raw12_to_png8(raw):
    """12-bit sensor values -> 8-bit, i.e. the top 8 bits (what Mono8 readout gives)."""
    return (np.asarray(raw, dtype=np.uint16) >> 4).clip(0, 255).astype(np.uint8)


def saturation_fraction(raw):
    return float((np.asarray(raw) >= 4095).mean())

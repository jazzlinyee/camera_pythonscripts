"""
Pipeline GUI: run calibrate -> capture -> cube and see what's happening.

    python pipeline.py gui [session]

Tabs:  Live (camera view, show any mask, set exposure)  |  1 Calibrate  |  2 Capture  |  3 Cube (viewer)
Hardware steps run in this window; processing steps (notebooks, cubes) run as a separate
Python process so the window stays responsive, and their output streams into the log.
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

import cv2
import numpy as np

from settings import PIPELINE_DIR, Session, list_sessions

BG = "#1e1e1e"

# Text colours for light and dark mode (macOS dark mode makes the window dark, so fixed
# dark-grey text would disappear). App.__init__ picks one set; widgets use self.C[...].
_PALETTES = {
    "light": {"muted": "#555555", "text2": "#333333", "good": "#1f7a45", "warn": "#b35900", "bad": "#c62828"},
    "dark":  {"muted": "#b4b4b4", "text2": "#e0e0e0", "good": "#6fd39a", "warn": "#ffb454", "bad": "#ff7070"},
}


def _is_dark(root):
    """True when the window background is dark (macOS dark mode, or a dark Tk theme)."""
    try:
        return bool(int(root.tk.call("tk::unsupported::MacWindowStyle", "isdark", root)))
    except Exception:  # noqa - not macOS / older Tk
        pass
    try:
        bg = ttk.Style(root).lookup("TFrame", "background") or root.cget("background")
        r, g, b = root.winfo_rgb(bg)
        return (0.299 * r + 0.587 * g + 0.114 * b) / 65535 < 0.5
    except Exception:  # noqa
        return False


# --------------------------------------------------------------------------- image helpers
def to_photo(img, max_w, max_h):
    """uint8 gray (H,W) or RGB (H,W,3) -> tk.PhotoImage fitted into max_w x max_h. Returns (photo, scale)."""
    h, w = img.shape[:2]
    sc = min(max_w / w, max_h / h)
    nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_NEAREST)
    if small.ndim == 2:
        data = b"P5 %d %d 255\n" % (nw, nh) + small.tobytes()
    else:
        data = b"P6 %d %d 255\n" % (nw, nh) + np.ascontiguousarray(small).tobytes()
    return tk.PhotoImage(data=data), sc


def stretch(raw, saturate=4095):
    """12-bit frame -> uint8 RGB with a percentile stretch; saturated pixels in red."""
    r = raw.astype(np.float32)
    lo, hi = np.percentile(r, [1, 99.7])
    g = np.clip((r - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    rgb = np.stack([g, g, g], -1)
    rgb[raw >= saturate] = (255, 0, 0)
    return rgb


# --------------------------------------------------------------------------- UI bridge for worker threads
class GuiUI:
    def __init__(self, app):
        self.app = app
        self.stop_flag = threading.Event()

    def main(self, fn):
        done, box = threading.Event(), {}

        def run():
            try:
                box["v"] = fn()
            except Exception as e:  # noqa
                box["e"] = e
            done.set()
        self.app.root.after(0, run)
        done.wait()
        if "e" in box:
            raise box["e"]
        return box.get("v")

    def prompt(self, msg):
        return self.main(lambda: messagebox.askokcancel("Next step", msg, parent=self.app.root))

    def progress(self, i, n, text=""):
        self.app.q.put(("progress", i, n, text))

    def stopped(self):
        return self.stop_flag.is_set()


# --------------------------------------------------------------------------- app
class App:
    def __init__(self, root, session_name=None, embed_lab=False, mirrored=False):
        self.root = root
        self.lab = self.labmod = None                # the lab camera GUI, shown inside the Capture tab
        self.mirrored = mirrored
        if embed_lab:
            try:
                self.labmod, self.lab_state = _import_lab_gui(session_name)
            except Exception as e:  # noqa
                print(f"Could not load the lab camera GUI ({type(e).__name__}: {e}) - "
                      f"using the pipeline's own controls instead.")
                self.labmod = None
        root.title("Hyperspectral pipeline")
        self.C = _PALETTES["dark" if _is_dark(root) else "light"]
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"{min(1500, sw - 40)}x{sh - 90}+10+0" if self.labmod else "1280x860")
        self.q = queue.Queue()
        self.session = None
        self.rig = None
        self.busy = None              # description of the running job
        self.log_dest = "main"        # "cal" while a calibration job runs (output -> Calibrate tab panel)
        self.proc = None              # running subprocess
        self.ui = GuiUI(self)
        self.live_on = False
        self.cube = None
        self._photos = {}
        self._build()
        names = list_sessions()
        if session_name or names:
            self.load_session(session_name or names[-1])
        self.root.after(100, self._drain)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------ layout
    def _build(self):
        # top bar: left part (session etc.) and the hardware buttons; the hardware buttons
        # move to a second line when the window is too narrow for both
        self.topwrap = ttk.Frame(self.root, padding=6)
        self.topwrap.pack(fill="x")
        self.topwrap.grid_columnconfigure(1, weight=1)
        top = ttk.Frame(self.topwrap)
        top.grid(row=0, column=0, sticky="w")
        ttk.Label(top, text="Session:").pack(side="left")
        self.sess_var = tk.StringVar()
        self.sess_box = ttk.Combobox(top, textvariable=self.sess_var, values=list_sessions(), width=22, state="readonly")
        self.sess_box.pack(side="left", padx=4)
        self.sess_box.bind("<<ComboboxSelected>>", lambda e: self.load_session(self.sess_var.get()))
        ttk.Button(top, text="New session…", command=self.new_session).pack(side="left")
        ttk.Button(top, text="Settings file", command=self.open_settings).pack(side="left", padx=4)
        ttk.Button(top, text="Live window ⧉", command=self.open_live_window).pack(side="left", padx=(12, 4))
        self.hw_label = ttk.Label(top, text="", foreground=self.C["muted"])
        self.hw_label.pack(side="left", padx=10)
        self.disconnect_btn = ttk.Button(top, text="Disconnect", command=self.disconnect, state="disabled")
        self.connect_btn = ttk.Button(top, text="Connect hardware", command=self.connect)
        if self.labmod is None:
            self.disconnect_btn.pack(side="right")
            self.connect_btn.pack(side="right", padx=4)
        else:
            self._build_hw_bar(self.topwrap)
            self._top_left = top
            self.topwrap.bind("<Configure>", lambda e: self._place_hw_bar(), add="+")

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=6)
        if self.labmod is None:
            self._build_live()              # the camera GUI has its own live views
        else:
            self._build_live_stub()
        self._build_calibrate()
        self._build_capture()
        self._build_cube()
        if self.lab is None:          # (with the camera GUI, _build_capture_lab binds this)
            self.nb.bind("<<NotebookTabChanged>>", self._on_tab, add="+")

        # status bar + pipeline log are pinned to the bottom (packed before the tabs),
        # so a tall Capture tab (the camera GUI) can never push them off-screen
        logf = ttk.Frame(self.root, padding=(6, 0, 6, 6))
        logf.pack(side="bottom", fill="x", before=self.nb)
        bottom = ttk.Frame(self.root, padding=(6, 2))
        bottom.pack(side="bottom", fill="x", before=self.nb)
        self.status = ttk.Label(bottom, text="Ready")
        self.status.pack(side="left")
        self.stop_btn = ttk.Button(bottom, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="right")
        self.pbar = ttk.Progressbar(bottom, length=300)
        self.pbar.pack(side="right", padx=8)
        self.log_text = tk.Text(logf, height=4 if self.labmod is not None else 11, bg=BG, fg="#ddd", font=("TkFixedFont", 10), wrap="word")
        sb = ttk.Scrollbar(logf, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)
        for tag, col in (("ok", "#6c6"), ("bad", "#e66"), ("warn", "#fc6"), ("head", "#8cf")):
            self.log_text.tag_configure(tag, foreground=col)

    # ------------------------------------------------------------------ hardware (top bar)
    _HW = (("slm", "SLM"), ("basler", "Basler"), ("blackfly", "Blackfly"))

    def _build_hw_bar(self, top):
        """SLM / Basler / Blackfly Open-Close buttons. They call the camera GUI's own
        open/close functions, so the 2 Capture tab always shows the same state."""
        bar = ttk.Frame(top)
        bar.grid(row=0, column=2, sticky="e")
        self._hw_bar, self._hw_bar_wrapped = bar, False
        self.hw_widgets = {}
        for key, name in self._HW:
            ttk.Label(bar, text=name).pack(side="left", padx=(10, 2))
            dot = ttk.Label(bar, text="○", foreground=self.C["muted"])
            dot.pack(side="left")
            btn = ttk.Button(bar, text="Open", width=6, command=lambda k=key: self._hw_toggle(k))
            btn.pack(side="left", padx=(2, 0))
            self.hw_widgets[key] = (dot, btn)
            if key == "blackfly":            # its processing switches + what the camera reports
                self.bf_gamma_var, self.bf_wb_var = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
                self.bf_gamma_cb = ttk.Checkbutton(bar, text="Gamma", variable=self.bf_gamma_var,
                                                   command=self._bf_toggle_gamma)
                self.bf_gamma_cb.pack(side="left", padx=(8, 0))
                self.bf_wb_cb = ttk.Checkbutton(bar, text="White balance", variable=self.bf_wb_var,
                                                command=self._bf_toggle_wb)
                self.bf_wb_cb.pack(side="left", padx=(4, 0))
                self.bf_status = ttk.Label(bar, text="", foreground=self.C["muted"])
                self.bf_status.pack(side="left", padx=(6, 0))
        self.hw_all_btn = ttk.Button(bar, text="Open all", command=self._hw_open_all)
        self.hw_all_btn.pack(side="left", padx=(12, 0))
        self.root.after(500, self._hw_refresh)

    def _place_hw_bar(self):
        """Hardware buttons on the right of the top bar, or on their own line below it
        when the window is too narrow for both."""
        try:
            need = self._top_left.winfo_reqwidth() + self._hw_bar.winfo_reqwidth() + 30
            wrap = self.topwrap.winfo_width() < need
            if wrap != self._hw_bar_wrapped:
                self._hw_bar_wrapped = wrap
                if wrap:
                    self._hw_bar.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))
                else:
                    self._hw_bar.grid(row=0, column=2, columnspan=1, sticky="e", pady=0)
        except (tk.TclError, AttributeError):
            pass

    def _hw_is_open(self, key):
        if self.lab is None:
            return False
        return getattr(self.lab, "slm", None) is not None if key == "slm" else self.lab.cams.get(key) is not None

    def _hw_refresh(self):
        """Keep the top-bar buttons in step with the camera GUI (also if things are opened
        or closed in the 2 Capture tab), and lock them while a job runs."""
        if self.lab is None or not hasattr(self, "hw_widgets"):
            return
        sim = self.session is not None and self.session.settings.get("hardware") == "sim"
        for key, (dot, btn) in self.hw_widgets.items():
            is_open = self._hw_is_open(key)
            dot.configure(text="●" if is_open else "○", foreground=self.C["good"] if is_open else self.C["muted"])
            btn.configure(text="Close" if is_open else "Open",
                          state="disabled" if (self.busy or sim) else "normal")
        all_open = all(self._hw_is_open(k) for k, _ in self._HW)
        self.hw_all_btn.configure(state="disabled" if (self.busy or sim or all_open) else "normal")
        self._bf_refresh()
        try:
            self.root.after(1000, self._hw_refresh)
        except tk.TclError:
            pass

    # ---- Blackfly gamma / white balance (top bar)
    def _bf_cam(self):
        cam = self.lab.cams.get("blackfly") if self.lab is not None else None
        return cam if cam is not None and hasattr(cam, "raw_settings") else None

    @staticmethod
    def _bf_is_raw(s):
        ratios_ok = all(not isinstance(s.get(k), (int, float)) or abs(s[k] - 1.0) <= 1e-3
                        for k in ("BalanceRatio_Red", "BalanceRatio_Blue"))
        return s.get("GammaEnable") is not True and str(s.get("BalanceWhiteAuto", "Off")) in ("Off", "n/a") and ratios_ok

    def _bf_refresh(self):
        """Read gamma / white balance back from the Blackfly and show it (about once a second)."""
        cam = self._bf_cam()
        usable = cam is not None and not self.busy
        for cb in (self.bf_gamma_cb, self.bf_wb_cb):
            cb.state(["!disabled"] if usable else ["disabled"])
        if cam is None:
            self.bf_status.configure(text="")
            return
        if self.busy:
            return                          # don't talk to the camera while a capture uses it
        try:
            s = cam.raw_settings()
        except Exception as e:  # noqa
            self.bf_status.configure(text=f"(can't read settings: {e})", foreground=self.C["warn"])
            return
        gamma_on = s.get("GammaEnable") is True
        wb_on = str(s.get("BalanceWhiteAuto", "Off")) not in ("Off", "n/a")
        self.bf_gamma_var.set(gamma_on)
        self.bf_wb_var.set(wb_on)
        r = lambda k: f"{s[k]:.2f}" if isinstance(s.get(k), (int, float)) else "?"
        raw = self._bf_is_raw(s)
        self.bf_status.configure(
            text=f"camera: gamma {'ON' if gamma_on else 'off'} · WB {'ON' if wb_on else 'off'} · "
                 f"R {r('BalanceRatio_Red')} B {r('BalanceRatio_Blue')}" + ("" if raw else "  NOT raw"),
            foreground=self.C["good"] if raw else self.C["warn"])

    def _bf_toggle_gamma(self):
        cam, on = self._bf_cam(), self.bf_gamma_var.get()
        try:
            cam.set_gamma(on)
            self.log(f"Blackfly gamma correction {'ON - frames are not raw' if on else 'off (raw)'}.",
                     "warn" if on else "ok")
        except Exception as e:  # noqa
            self.bf_gamma_var.set(not on)
            messagebox.showinfo("Blackfly", f"Could not change gamma: {e}")
        self._bf_refresh()

    def _bf_toggle_wb(self):
        cam, on = self._bf_cam(), self.bf_wb_var.get()
        try:
            note = cam.set_white_balance(on)
            self.log(f"Blackfly white balance {'ON (automatic) - frames are not raw' if on else 'off (raw)'}: {note}.",
                     "warn" if on else "ok")
        except Exception as e:  # noqa
            self.bf_wb_var.set(not on)
            messagebox.showinfo("Blackfly", f"Could not change white balance: {e}")
        self._bf_refresh()

    def _hw_toggle(self, key):
        if self.busy:
            return
        if self._hw_is_open(key):
            self._hw_close(key)
        else:
            self._hw_open(key)
        self._hw_refresh_once()

    def _hw_refresh_once(self):
        for key, (dot, btn) in getattr(self, "hw_widgets", {}).items():
            is_open = self._hw_is_open(key)
            dot.configure(text="●" if is_open else "○", foreground=self.C["good"] if is_open else self.C["muted"])
            btn.configure(text="Close" if is_open else "Open")

    def _hw_open(self, key):
        lab = self.lab
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            if key == "slm":
                lab.open_slm()
                if lab.slm is None:
                    messagebox.showinfo("SLM", "The SLM did not open - see the camera GUI's log in the 2 Capture tab.")
                    return
                if not getattr(self.labmod, "SLM_OPENS_WHITE", False):   # older camera GUI
                    self._show_white_on_slm()
            else:
                prev = lab.cam_choice.get()
                lab.cam_choice.set(key)
                try:
                    lab.open_camera()
                finally:
                    lab.cam_choice.set(prev)
                if lab.cams.get(key) is None:
                    messagebox.showinfo(key.capitalize(), f"The {key.capitalize()} did not open - see the camera "
                                                          "GUI's log in the 2 Capture tab.")
        finally:
            self.root.configure(cursor="")

    def _hw_close(self, key):
        lab = self.lab
        if key == "slm":
            self._adj_stop(restore=False)
            lab.close_slm()
        else:
            prev = lab.cam_choice.get()
            lab.cam_choice.set(key)
            try:
                lab.close_camera()
            finally:
                lab.cam_choice.set(prev)

    def _hw_open_all(self):
        for key, _ in self._HW:
            if not self._hw_is_open(key) and not self.busy:
                self._hw_open(key)
        self._hw_refresh_once()

    def _white_pattern(self):
        white = None
        if self.session is not None:
            white = cv2.imread(str(self.session.lab_path("patterns") / "pat_white.png"), cv2.IMREAD_GRAYSCALE)
        return white if white is not None else np.full((1080, 1920), 255, np.uint8)

    def _show_white_on_slm(self):
        """Show white on a freshly opened SLM. White is added as an extra, last mask, so the
        camera GUI's own mask list (and its numbering) stays as it is."""
        lab = self.lab
        try:
            white = self._white_pattern()
            masks = list(lab.masks) + [white]
            lab.slm.set_masks(masks)
            lab.slm.show_mask(len(masks) - 1)
            w, h = lab.preview_sizes["mask"]
            lab._mask_thumb_photo = self.labmod.array_to_photoimage(white, w, h)
            lab.mask_preview_label.configure(image=lab._mask_thumb_photo)
            lab.log("SLM shows the white pattern (opened from the pipeline's top bar).")
        except Exception as e:  # noqa
            self.log(f"(could not show white on the SLM: {e})")

    def _build_live(self):
        f = ttk.Frame(self.nb, padding=6)
        self.nb.add(f, text="Live")
        self.live_canvas = tk.Canvas(f, bg="black", width=820, height=560, highlightthickness=0)
        self.live_canvas.pack(side="left", fill="both", expand=True)
        side = ttk.Frame(f, padding=(10, 0))
        side.pack(side="left", fill="y")
        self.live_btn = ttk.Button(side, text="Start live view", command=self.toggle_live)
        self.live_btn.pack(fill="x")
        self.live_stats = ttk.Label(side, text="", justify="left")
        self.live_stats.pack(anchor="w", pady=6)
        ttk.Separator(side).pack(fill="x", pady=6)
        ttk.Label(side, text="Basler exposure (µs)").pack(anchor="w")
        row = ttk.Frame(side); row.pack(fill="x")
        self.live_exp = tk.StringVar(value="100000")
        ttk.Entry(row, textvariable=self.live_exp, width=12).pack(side="left")
        ttk.Button(row, text="Apply", command=self.apply_exposure).pack(side="left", padx=4)
        ttk.Separator(side).pack(fill="x", pady=6)
        ttk.Label(side, text="Show on SLM").pack(anchor="w")
        self.mask_choice = tk.StringVar(value="black")
        self.mask_box = ttk.Combobox(side, textvariable=self.mask_choice, state="readonly", width=24)
        self.mask_box.pack(fill="x")
        ttk.Button(side, text="Show", command=self.show_selected_mask).pack(fill="x", pady=4)
        self.mask_canvas = tk.Canvas(side, bg="black", width=240, height=135, highlightthickness=0)
        self.mask_canvas.pack(pady=6)
        ttk.Label(side, text="Red = saturated pixels.\nUse this tab to set exposures\nbefore calibrating / capturing.",
                  foreground=self.C["muted"]).pack(anchor="w")

    def _build_live_stub(self):
        # Live tab widgets are referenced elsewhere; keep them, but unshown.
        holder = ttk.Frame(self.root)
        nb, self.nb = self.nb, ttk.Notebook(holder)
        self._build_live()
        self.nb = nb

    def _build_calibrate(self):
        f = ttk.Frame(self.nb, padding=10)
        self.nb.add(f, text="1  Calibrate")
        self.cal_tab = f
        f.columnconfigure(0, weight=1)
        ttk.Label(f, text="Once per optical setup. Nothing in the setup may move between calibration and scene "
                          "captures. Capture each laser with its own button, in any order.",
                  foreground=self.C["muted"]).grid(row=0, column=0, sticky="w")

        # one row per laser (built in load_session from session.json's lasers list)
        self.laser_frame = ttk.Frame(f)
        self.laser_frame.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.laser_rows = []

        # exposure adjuster: changes the Basler exposure live (watch the Live window)
        adj = ttk.Frame(f)
        adj.grid(row=2, column=0, sticky="we", pady=(8, 0))
        self.adj_laser = None
        self.adj_label = ttk.Label(adj, text="Exposure: press 'Set exposure' on a laser", width=34)
        self.adj_label.pack(side="left")
        self.adj_var = tk.StringVar()
        self.adj_entry = ttk.Entry(adj, textvariable=self.adj_var, width=11)
        self.adj_entry.pack(side="left", padx=(6, 0))
        self.adj_entry.bind("<Return>", lambda e: self._adj_apply_text())
        ttk.Label(adj, text="µs").pack(side="left", padx=(2, 6))
        self.adj_apply_btn = ttk.Button(adj, text="Apply", command=self._adj_apply_text)
        self.adj_apply_btn.pack(side="left")
        self.adj_done = ttk.Button(adj, text="Done", command=self._adj_stop)
        self.adj_done.pack(side="left", padx=(4, 0))
        self.adj_readout = ttk.Label(adj, text="", foreground=self.C["text2"])
        self.adj_readout.pack(side="left", padx=10)
        self._adj_widgets = (self.adj_entry, self.adj_apply_btn, self.adj_done)
        self._adj_enable(False)

        opts = ttk.Frame(f)
        opts.grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Label(opts, text="Orientation-check frames with:").pack(side="left")
        self.check_var = tk.StringVar()
        self.check_box = ttk.Combobox(opts, textvariable=self.check_var, width=8, state="readonly")
        self.check_box.pack(side="left", padx=(4, 16))
        ttk.Label(opts, text="Settle time (ms):").pack(side="left")
        self.settle_var = tk.StringVar()
        ttk.Entry(opts, textvariable=self.settle_var, width=7).pack(side="left", padx=(4, 16))
        self.flip_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Flip Gray-code patterns left-right (same for all lasers)",
                        variable=self.flip_var).pack(side="left")

        btns = ttk.Frame(f)
        btns.grid(row=4, column=0, sticky="we", pady=10)
        ttk.Button(btns, text="B. Process → table",
                   command=lambda: self.run_proc(["calibrate", "process", "{S}"], what="Calibrate process",
                                                 dest="cal")).pack(side="left")
        ttk.Button(btns, text="C. Re-run check",
                   command=lambda: self.run_proc(["calibrate", "check", "{S}"], what="Calibrate check",
                                                 dest="cal")).pack(side="left", padx=6)
        self.cal_summary = ttk.Label(btns, text="", foreground=self.C["text2"])
        self.cal_summary.pack(side="left", padx=12)

        # terminal-style output panel for calibration capture + processing
        term = ttk.Frame(f)
        term.grid(row=5, column=0, sticky="nsew")
        f.rowconfigure(5, weight=1)
        bar = ttk.Frame(term)
        bar.pack(fill="x")
        ttk.Label(bar, text="Calibration output").pack(side="left")
        ttk.Button(bar, text="Log file", command=self._open_cal_log).pack(side="right")
        ttk.Button(bar, text="Copy", command=self._copy_cal_term).pack(side="right", padx=4)
        ttk.Button(bar, text="Clear", command=lambda: self.cal_term.delete("1.0", "end")).pack(side="right")
        box = ttk.Frame(term)
        box.pack(fill="both", expand=True, pady=(4, 0))
        self.cal_term = tk.Text(box, height=14, bg="#0c0c0c", fg="#d0d0d0", insertbackground="#d0d0d0",
                                font=("TkFixedFont", 10), wrap="none", relief="flat", padx=6, pady=4)
        ysb = ttk.Scrollbar(box, command=self.cal_term.yview)
        xsb = ttk.Scrollbar(box, orient="horizontal", command=self.cal_term.xview)
        self.cal_term.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        ysb.pack(side="right", fill="y")
        xsb.pack(side="bottom", fill="x")
        self.cal_term.pack(fill="both", expand=True)
        for tag, col in (("ok", "#6c6"), ("bad", "#f66"), ("warn", "#fc6"), ("head", "#8cf"),
                         ("prompt", "#f9f"), ("dim", "#9a9a9a")):
            self.cal_term.tag_configure(tag, foreground=col)

    def _build_laser_rows(self):
        self._adj_stop()
        for w in self.laser_frame.winfo_children():
            w.destroy()
        self.laser_rows = []
        self._laser_names = {}
        for c, text in enumerate(("Laser", "nm", "Basler exposure (µs)", "Status", "", "")):
            ttk.Label(self.laser_frame, text=text, foreground=self.C["muted"]).grid(row=0, column=c, sticky="w", padx=(0, 10))
        for r, (folder, nm, exp) in enumerate(self.session.settings["calibration"]["lasers"], 1):
            nm_var, exp_var = tk.StringVar(value=f"{nm:g}"), tk.StringVar(value=f"{exp:.0f}")
            name = ttk.Label(self.laser_frame, text=folder, width=8)
            name.grid(row=r, column=0, sticky="w")
            ttk.Entry(self.laser_frame, textvariable=nm_var, width=6).grid(row=r, column=1, sticky="w", padx=(0, 10))
            ttk.Entry(self.laser_frame, textvariable=exp_var, width=12).grid(row=r, column=2, sticky="w", padx=(0, 10))
            status = ttk.Label(self.laser_frame, text="", width=58)
            status.grid(row=r, column=3, sticky="w", padx=(0, 10))
            ttk.Button(self.laser_frame, text="Set exposure",
                       command=lambda fo=folder: self._adj_start(fo)).grid(row=r, column=4, sticky="w", pady=1, padx=(0, 4))
            ttk.Button(self.laser_frame, text=f"Capture {folder}",
                       command=lambda fo=folder: self.run_cal_capture(fo)).grid(row=r, column=5, sticky="w", pady=1)
            self.laser_rows.append((folder, nm_var, exp_var, status))
            self._laser_names[folder] = name
        names = [r[0] for r in self.laser_rows]
        self.check_box.configure(values=names)
        cl = self.session.settings["calibration"].get("check_laser")
        self.check_var.set(cl if cl in names else (names[0] if names else ""))

    # ------------------------------------------------------------------ exposure adjuster
    _ADJ_MIN, _ADJ_MAX = 100.0, 10_000_000.0          # µs

    def _adj_enable(self, on):
        for w in self._adj_widgets:
            try:
                w.state(["!disabled"] if on else ["disabled"])
            except tk.TclError:
                pass

    def _adj_row(self, folder):
        return next((r for r in self.laser_rows if r[0] == folder), None)

    def _adj_start(self, folder):
        """'Set exposure' on a laser row: white pattern on the SLM, the Basler at this
        laser's exposure, and the Live window open so the change can be watched."""
        if not self.session or self.busy:
            if self.busy:
                messagebox.showinfo("Busy", f"Wait for '{self.busy}' to finish.")
            return
        if not self._need_rig():
            return
        if self.live_camera("basler") is None:
            messagebox.showinfo("Exposure", "Open the Basler first (top bar: Basler → Open)."); return
        try:
            white = self._white_pattern()
            self.rig.load_masks([white])
            self.rig.show(0)
        except Exception as e:  # noqa
            messagebox.showinfo("SLM", f"Could not show the white pattern on the SLM ({e}). "
                                       "Is the SLM open (top bar: SLM → Open)?"); return
        prev = self.adj_laser
        if prev and prev in self._laser_names:
            self._laser_names[prev].configure(text=prev)
        self.adj_laser = folder
        self._laser_names[folder].configure(text=f"▶ {folder}")
        self.adj_label.configure(text=f"Adjusting {folder} (SLM shows white):")
        self._adj_enable(True)
        self.open_live_window()
        lw = self.live_window
        lw.cam_var.set("basler")
        lw.mode_var.set("true")
        self._cal_note(f"Adjusting the {folder} exposure: white pattern on the SLM, Live window open.")
        row = self._adj_row(folder)
        try:
            us = float(row[2].get())
        except ValueError:
            us = 100000.0
        self._adj_apply(us)

    def _cal_note(self, text, tag="head"):
        """A line for the Calibrate tab's output panel (and its log file), outside a job."""
        dest, self.log_dest = self.log_dest, "cal"
        try:
            self._write_log(text, tag)
        finally:
            self.log_dest = dest

    def _adj_apply_text(self):
        try:
            self._adj_apply(float(self.adj_var.get()))
        except ValueError:
            messagebox.showinfo("Exposure", "Exposure must be a number (µs).")

    def _adj_apply(self, us):
        """Set the Basler exposure now and store it as this laser's calibration exposure."""
        if not self.adj_laser or self.busy:
            return
        us = float(min(max(us, self._ADJ_MIN), self._ADJ_MAX))
        self.adj_var.set(f"{us:.0f}")
        row = self._adj_row(self.adj_laser)
        if row:
            row[2].set(f"{us:.0f}")
        try:
            self.rig.set_exposure(us)
        except Exception as e:  # noqa
            self.adj_readout.configure(text=f"could not set exposure: {e}"); return
        self.save_form()
        self._cal_note(f"  {self.adj_laser} exposure -> {us:.0f} us ({us / 1e6:g} s)", None)
        self.adj_readout.configure(text=f"set to {us / 1e6:g} s - waiting for the next frame…",
                                   foreground=self.C["text2"])

    def _adj_stop(self, restore=True):
        """Leave exposure-adjust mode and give the SLM back to the camera GUI's masks."""
        folder, self.adj_laser = getattr(self, "adj_laser", None), None
        if folder is None:
            return
        if folder in getattr(self, "_laser_names", {}):
            try:
                self._laser_names[folder].configure(text=folder)
            except tk.TclError:
                pass
        self.adj_label.configure(text="Exposure: press 'Set exposure' on a laser")
        self.adj_readout.configure(text="")
        self._adj_enable(False)
        if restore and self.lab is not None and getattr(self.lab, "slm", None) is not None:
            try:
                if getattr(self.lab, "_showing_white", False) and hasattr(self.lab, "show_white"):
                    self.lab.show_white()
                elif self.lab.masks:
                    self.lab.slm.set_masks(self.lab.masks)
                self.lab._render_previews(["mask"])
            except Exception:  # noqa
                pass

    def _cal_log_path(self):
        return self.session.dir / "calibration" / "calibration_log.txt" if self.session else None

    def _open_cal_log(self):
        p = self._cal_log_path()
        if p is None or not p.is_file():
            messagebox.showinfo("Log file", "No calibration log yet for this session."); return
        subprocess.Popen(["open", "-t", str(p)] if sys.platform == "darwin" else ["xdg-open", str(p)])

    def _copy_cal_term(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.cal_term.get("1.0", "end-1c"))
        self.status.configure(text="Calibration output copied to the clipboard.")

    def _build_capture(self):
        if self.labmod is not None:
            return self._build_capture_lab()
        f = ttk.Frame(self.nb, padding=10)
        self.nb.add(f, text="2  Capture")
        ttk.Label(f, text="Captures an all-zero frame, the 56 scan masks, and another all-zero frame. Keep the scene still.",
                  foreground=self.C["muted"]).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(f, text="Scene name").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.scene_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.scene_var, width=24).grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Label(f, text="Basler exposure (µs)").grid(row=2, column=0, sticky="w")
        self.cap_exp = tk.StringVar()
        ttk.Entry(f, textvariable=self.cap_exp, width=12).grid(row=2, column=1, sticky="w")
        self.bf_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Also capture RGB guide camera (Blackfly)", variable=self.bf_var).grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Button(f, text="Capture scene", command=self.run_capture).grid(row=4, column=0, sticky="w", pady=10)
        self.cap_canvas = tk.Canvas(f, bg="black", width=640, height=430, highlightthickness=0)
        self.cap_canvas.grid(row=5, column=0, columnspan=4, sticky="nw")
        self.cap_info = ttk.Label(f, text="", foreground=self.C["muted"])
        self.cap_info.grid(row=6, column=0, columnspan=4, sticky="w")

    def _build_capture_lab(self):
        """2 Capture tab = the lab camera GUI, with a slim pipeline bar on top."""
        f = ttk.Frame(self.nb)
        self.nb.add(f, text="2  Capture")
        self.capture_tab = f
        bar = ttk.Frame(f, padding=(8, 6))
        bar.pack(fill="x")
        ttk.Label(bar, text="Pipeline scene capture:", font=("TkDefaultFont", 11, "bold")).pack(side="left")
        ttk.Label(bar, text="  Scene name").pack(side="left")
        self.scene_var = tk.StringVar()
        ttk.Entry(bar, textvariable=self.scene_var, width=16).pack(side="left", padx=4)
        ttk.Label(bar, text="Basler exposure (µs)").pack(side="left", padx=(8, 0))
        self.cap_exp = tk.StringVar()
        ttk.Entry(bar, textvariable=self.cap_exp, width=10).pack(side="left", padx=4)
        self.bf_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="+ Blackfly RGB", variable=self.bf_var).pack(side="left", padx=8)
        ttk.Button(bar, text="Capture scene (zero + 56 masks + zero)", command=self.run_capture).pack(side="left", padx=4)
        ttk.Button(bar, text="Live window ⧉", command=self.open_live_window).pack(side="left", padx=4)
        self.cap_info = ttk.Label(bar, text="", foreground=self.C["muted"])
        self.cap_info.pack(side="left", padx=8)
        self.cap_canvas = None
        ttk.Separator(f).pack(fill="x")
        # The camera GUI sits on a scrollable canvas: it always gets at least the size its
        # panels + previews + log need, and scrollbars appear when the window is smaller.
        area = ttk.Frame(f)
        area.pack(fill="both", expand=True)
        bg = ttk.Style(self.root).lookup("TFrame", "background") or self.root.cget("background")
        canvas = tk.Canvas(area, highlightthickness=0, bd=0, background=bg)
        self._cap_vsb = ttk.Scrollbar(area, orient="vertical", command=canvas.yview)
        self._cap_hsb = ttk.Scrollbar(area, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=self._cap_vsb.set, xscrollcommand=self._cap_hsb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        area.grid_rowconfigure(0, weight=1)
        area.grid_columnconfigure(0, weight=1)
        holder = _LabHost(canvas)
        self._cap_win = canvas.create_window(0, 0, window=holder, anchor="nw")
        self._cap_canvas, self._cap_holder = canvas, holder
        self.lab = self.labmod.App(holder, self.mirrored)
        self.lab.load_saved_folders()
        if not self.lab.mask_root_folders:
            self.lab.add_folder_path(self.labmod.ensure_test_pattern_folder())
        self.lab.log(f"Running inside the pipeline. Mask folders / test captures are kept in {self.lab_state} "
                     f"(camera_pythonscripts is not written to).")
        self._hide_lab_live()
        self._lab_sized = False
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab, add="+")
        canvas.bind("<Configure>", lambda e: self._cap_layout(), add="+")
        try:   # panels collapsed / expanded -> the minimum size changes
            self.lab.top_section.bind("<Configure>", lambda e: self.root.after(60, self._cap_layout), add="+")
        except AttributeError:
            pass
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.bind_all(seq, self._cap_wheel, add="+")

    MIN_PREVIEW, MIN_LOG = 230, 90

    def _cap_min_size(self):
        """Smallest size at which the camera GUI shows everything: its toolbar, its whole
        panel row (Camera / SLM / Auto Cycle, as currently expanded), previews at least
        MIN_PREVIEW px and a few lines of log."""
        lab = self.lab
        try:
            w = lab.top_section.winfo_reqwidth() + 16
            h = (lab.toolbar.winfo_reqheight() + lab.top_section.winfo_reqheight()
                 + self.MIN_PREVIEW + self.MIN_LOG + 40)
            return w, h
        except (tk.TclError, AttributeError):
            return 0, 0

    def _cap_layout(self):
        if getattr(self, "_cap_in_layout", False):
            return
        self._cap_in_layout = True
        try:
            c = self._cap_canvas
            cw, ch = c.winfo_width(), c.winfo_height()
            if cw < 20 or ch < 20:
                return
            mw, mh = self._cap_min_size()
            need_h, need_v = mw > cw, mh > ch
            (self._cap_vsb.grid(row=0, column=1, sticky="ns") if need_v else self._cap_vsb.grid_remove())
            (self._cap_hsb.grid(row=1, column=0, sticky="ew") if need_h else self._cap_hsb.grid_remove())
            w, h = max(cw, mw), max(ch, mh)
            c.itemconfigure(self._cap_win, width=w, height=h)
            c.configure(scrollregion=(0, 0, w, h))
            if not need_v:
                c.yview_moveto(0)
            if not need_h:
                c.xview_moveto(0)
        except tk.TclError:
            pass
        finally:
            self._cap_in_layout = False
        self._schedule_fit()

    def _cap_wheel(self, e):
        """Mouse wheel / trackpad over the Capture tab scrolls it (when it is scrollable),
        except over widgets that scroll themselves (mask tree, log, previews)."""
        c = getattr(self, "_cap_canvas", None)
        if c is None or not self._cap_vsb.winfo_ismapped():
            return
        w = e.widget
        if not isinstance(w, tk.Misc) or isinstance(w, (tk.Text, tk.Listbox, ttk.Treeview)):
            return
        if not (w is c or str(w).startswith(str(self._cap_holder))):
            return
        if getattr(e, "num", None) in (4, 5):
            step = -1 if e.num == 4 else 1
        else:
            step = -e.delta if sys.platform == "darwin" else -(e.delta // 120)
        c.yview_scroll(int(step), "units")

    def _hide_lab_live(self):
        """Inside the pipeline the live feed lives in its own window (Live window ⧉), so hide
        the camera GUI's live toggles and live panes here and give their room to the mask /
        capture previews. Only the on-screen layout changes; the camera GUI's file doesn't."""
        lab, mod = self.lab, self.labmod
        try:
            for w in (lab._live_ctl_frame, lab.blackfly_live_label.master, lab.basler_live_label.master):
                w.pack_forget()
        except Exception as e:  # noqa  (camera GUI changed shape - leave it as it is)
            self.log(f"(kept the camera GUI's live panes: {e})")
            return
        orig = lab.apply_preview_size
        lo_w, lo_h = getattr(mod, "PREVIEW_MIN_W", 120), getattr(mod, "PREVIEW_MIN_H", 80)
        hi_w, hi_h = getattr(mod, "PREVIEW_MAX_W", 1400), getattr(mod, "PREVIEW_MAX_H", 1000)

        def apply_preview_size(force=False):
            if self.busy:                         # don't redraw over the pipeline's mask/frame previews
                return
            orig(force=force)                     # the camera GUI's own sizing first
            changed = set()
            for key, col in lab._preview_cols.items():
                cw, ch = col.winfo_width(), col.winfo_height()
                if cw < 60 or ch < 60:
                    continue
                avail = ch - 60 if key == "mask" else ch - 90     # no live pane below any more
                w = max(lo_w, min(hi_w, cw - 30))
                h = max(lo_h, min(hi_h, avail))
                if (w, h) != tuple(lab.preview_sizes[key]):
                    lab.preview_sizes[key] = (w, h)
                    changed.add(key)
            if changed:
                lab._render_previews(changed)
        lab.apply_preview_size = apply_preview_size   # this running instance only

    def _on_tab(self, _e=None):
        if self.adj_laser and self.nb.select() != str(self.cal_tab):
            self._adj_stop()
        if self.lab is None or self.nb.select() != str(getattr(self, "capture_tab", "")):
            return
        if not self._lab_sized:
            self._lab_sized = True
            self.root.update_idletasks()
            for fn in ("_set_top_sash", "_set_log_sash"):
                if hasattr(self.lab, fn):
                    getattr(self.lab, fn)()
            self.capture_tab.bind("<Configure>", lambda e: self._schedule_fit(), add="+")
        self._schedule_fit()

    def _schedule_fit(self):
        if getattr(self, "_fit_id", None):
            self.root.after_cancel(self._fit_id)
        self._fit_id = self.root.after(120, self._fit_lab)

    def _fit_lab(self, min_preview=230):
        """Keep the camera GUI's preview row visible inside the tab: if its control
        panels + log would leave less than `min_preview` px, trim the log first, then
        the panels (the same dividers its own collapse buttons move)."""
        self._fit_id = None
        lab = self.lab
        if self.busy:
            return
        try:
            self.root.update_idletasks()
            pw = lab.main_pw
            total = pw.winfo_height()
            if total < 200:
                return
            top = lab.top_section.winfo_reqheight()
            logh = lab.log_section.winfo_reqheight()
            if total - top - logh < min_preview:
                logh = max(90, total - top - min_preview)
                top = max(120, min(top, total - logh - min_preview))
                pw.sashpos(0, top)
                pw.sashpos(1, total - logh)
            if hasattr(lab, "apply_preview_size"):
                lab.apply_preview_size(force=True)
        except tk.TclError:
            pass

    def _build_cube(self):
        f = ttk.Frame(self.nb, padding=6)
        self.nb.add(f, text="3  Cube")
        left = ttk.Frame(f); left.pack(side="left", fill="y")
        ttk.Label(left, text="Scenes").pack(anchor="w")
        self.scene_list = tk.Listbox(left, width=26, height=14, exportselection=False)
        self.scene_list.pack(fill="y")
        self.scene_list.bind("<<ListboxSelect>>", lambda e: self.show_scene())
        self.force_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Ignore failed check (--force)", variable=self.force_var).pack(anchor="w", pady=4)
        ttk.Button(left, text="Make cube", command=self.run_cube).pack(fill="x")
        ttk.Button(left, text="Load cube in viewer", command=self.load_cube).pack(fill="x", pady=4)
        self.scene_info = ttk.Label(left, text="", justify="left", foreground=self.C["text2"], wraplength=200)
        self.scene_info.pack(anchor="w", pady=6)
        right = ttk.Frame(f); right.pack(side="left", fill="both", expand=True, padx=8)
        self.cube_canvas = tk.Canvas(right, bg="black", width=760, height=430, highlightthickness=0)
        self.cube_canvas.pack(anchor="nw")
        self.cube_canvas.bind("<Button-1>", self.on_cube_click)
        ttk.Label(right, text="Click the image to see the spectrum at that point (3×3 average).", foreground=self.C["muted"]).pack(anchor="w")
        self.spec_canvas = tk.Canvas(right, bg="white", width=760, height=200, highlightthickness=0)
        self.spec_canvas.pack(anchor="nw", pady=4)

    # ------------------------------------------------------------------ logging / queue
    def log(self, text, tag=None):
        self.q.put(("log", text, tag))

    def _write_log(self, text, tag=None):
        if tag is None:
            t = text.lower()
            tag = ("bad" if ("error" in t or "fail" in t or "mismatch" in t or "flipped" in t) else
                   "warn" if ("warning" in t or "saturated" in t) else
                   "prompt" if t.lstrip().startswith(">>") else
                   "dim" if (" cell " in t and "done" in t) else
                   "ok" if ("passed" in t or " ok" in t or "done" in t) else None)
        if self.log_dest == "cal":
            widget = self.cal_term
            p = self._cal_log_path()
            if p is not None:
                try:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with open(p, "a") as fh:
                        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {text.rstrip()}\n")
                except OSError:
                    pass
        else:
            widget = self.log_text
            if tag == "dim":
                return                      # notebook cell ticks only belong in the calibration panel
        widget.insert("end", text.rstrip() + "\n", None if (tag == "dim" and widget is self.log_text) else tag)
        widget.see("end")

    def _drain(self):
        try:
            while True:
                m = self.q.get_nowait()
                if m[0] == "log":
                    self._write_log(m[1], m[2])
                elif m[0] == "progress":
                    _, i, n, text = m
                    self.pbar.configure(maximum=n, value=i)
                    self.status.configure(text=f"{self.busy}: {i}/{n}  {text}")
                elif m[0] == "done":
                    self._job_done(m[1], m[2])
                elif m[0] == "live":
                    self._show_live(m[1])
                elif m[0] == "lastcap":
                    self._show_capture(m[1])
                elif m[0] == "cube_loaded":
                    self._cube_loaded()
        except queue.Empty:
            pass
        self.root.after(80, self._drain)

    # ------------------------------------------------------------------ session
    def load_session(self, name):
        try:
            self.session = Session(name)
        except Exception as e:  # noqa
            messagebox.showerror("Session", str(e)); return
        self.sess_var.set(name)
        self.sess_box.configure(values=list_sessions())
        if self.lab is not None and not self.busy:
            self.rig = None                 # re-made from the camera window for this session
        s = self.session.settings
        self.hw_label.configure(text=f"hardware: {s['hardware']}")
        self._build_laser_rows()
        self.cal_term.delete("1.0", "end")           # show the end of this session's calibration log
        p = self._cal_log_path()
        if p is not None and p.is_file():
            try:
                tail = p.read_text(errors="replace").splitlines()[-300:]
                self.cal_term.insert("end", f"--- earlier output (last {len(tail)} lines of {p.name}) ---\n", "head")
                self.cal_term.insert("end", "\n".join(tail) + "\n")
                self.cal_term.insert("end", "--- end of earlier output ---\n\n", "head")
                self.cal_term.see("end")
            except OSError:
                pass
        self.settle_var.set(str(s["settle_ms"]))
        self.flip_var.set(bool(s["calibration"].get("flip_patterns_x", False)))
        self.cap_exp.set(str(s["capture"]["exposure_us"]))
        self.bf_var.set(bool(s["blackfly"]["enabled"]))
        self.live_exp.set(str(s["capture"]["exposure_us"]))
        self.refresh_status()

    def new_session(self):
        name = simpledialog.askstring("New session", "Session name (e.g. the date, 101426):", parent=self.root)
        if not name:
            return
        if (Session(name).settings_path).exists():
            messagebox.showerror("New session", f"{name} already exists."); return
        prev = self.session.settings if self.session else {}
        sim = messagebox.askyesno("New session", "Use the simulator (no hardware)?\n\nChoose No for the real lab setup.",
                                  parent=self.root)
        reuse = ""
        if self.session and (self.session.dir / "calibration" / "reverse_lookup.npz").is_file():
            if messagebox.askyesno("New session", f"Reuse the calibration of session {self.session.name}?\n"
                                   "(Only if nothing in the optics has moved since.)", parent=self.root):
                reuse = self.session.name
        extra = {k: prev[k] for k in ("lab", "slm", "basler", "blackfly", "settle_ms", "calibration", "capture") if k in prev}
        extra["hardware"] = "sim" if sim else "real"
        extra.setdefault("calibration", {})
        extra["calibration"] = dict(extra["calibration"], use_from_session=reuse)
        Session.create(name, **extra)
        self.load_session(name)

    def open_settings(self):
        if not self.session:
            return
        p = str(self.session.settings_path)
        subprocess.Popen(["open", "-t", p] if sys.platform == "darwin" else ["xdg-open", p])
        self.log(f"Edit {p}, then re-select the session to reload it.")

    def save_form(self):
        s = self.session.settings
        try:
            for (folder, nm_var, exp_var, _), laser in zip(self.laser_rows, s["calibration"]["lasers"]):
                laser[1] = float(nm_var.get())
                laser[2] = float(exp_var.get())
            if self.check_var.get():
                s["calibration"]["check_laser"] = self.check_var.get()
            s["settle_ms"] = int(float(self.settle_var.get()))
            s["calibration"]["flip_patterns_x"] = bool(self.flip_var.get())
            s["capture"]["exposure_us"] = float(self.cap_exp.get())
            s["blackfly"]["enabled"] = bool(self.bf_var.get())
        except ValueError:
            messagebox.showerror("Settings", "Wavelengths, exposures and settle time must be numbers."); return False
        self.session.save()
        return True

    def refresh_status(self):
        if not self.session:
            return
        s = self.session
        cal = s.cal_dir
        check_laser = s.settings["calibration"].get("check_laser")
        for folder, nm_var, exp_var, status in self.laser_rows:
            d = cal / "Captures" / folder
            n = len(list(d.glob("cap_*.png"))) if d.is_dir() else 0
            pj = {}
            if (d / "patterns.json").is_file():
                try:
                    pj = json.loads((d / "patterns.json").read_text())
                except Exception:  # noqa
                    pj = {}
            if pj.get("incomplete"):
                text = f"INCOMPLETE - stopped at {n}/46 - press Capture to redo (partial files are cleared)"
                status.configure(text=text, foreground=self.C["bad"])
                continue
            if n == 0:
                text = "not captured"
            else:
                text = f"{n}/46"
                info = {}
                if (d / "patterns.json").is_file():
                    try:
                        info = json.loads((d / "patterns.json").read_text())
                    except Exception:  # noqa
                        info = {}
                if info.get("captured"):
                    text += f"  captured {time.strftime('%b %d %H:%M', time.strptime(info['captured'][:16], '%Y-%m-%dT%H:%M'))}"
                if info.get("exposure_us"):
                    text += f" @ {info['exposure_us'] / 1e6:g} s"
                if info.get("flip_patterns_x"):
                    text += ", flipped"
                if info.get("saturated_frames"):
                    text += f", {len(info['saturated_frames'])} saturated!"
            if folder == check_laser:
                text += "  + check frames" if (cal / "Captures" / "check" / "zero.tiff").is_file() else "  (check frames: none yet)"
            status.configure(text=text, foreground=self.C["good"] if n >= 46 else self.C["warn"] if n else self.C["muted"])
        parts = []
        if s.settings["calibration"].get("use_from_session"):
            parts.append(f"using session {s.settings['calibration']['use_from_session']}'s calibration")
        parts.append("Table: " + ("yes" if s.table.is_file() else "not yet"))
        if (cal / "check.json").is_file():
            r = json.loads((cal / "check.json").read_text())
            txt = "Check: " + ("PASSED" if r["passed"] else "FAILED")
            if "shift_x" in r:
                txt += f" (left-right {r['shift_x']['predicted_px']:+} predicted vs {r['shift_x']['measured_px']:+} px)"
            parts.append(txt)
        self.cal_summary.configure(text="   ·   ".join(parts))
        self.scene_list.delete(0, "end")
        for sc in s.scenes():
            has = (s.scene_dir(sc) / "cubes" / "cube_combined.npz").is_file()
            self.scene_list.insert("end", f"{sc}{'   [cube]' if has else ''}")
        masks = ["black", "white"] + [f"mask_{k:02d}" for k in range(56)] + [f"pat_{k:02d}" for k in range(44)]
        self.mask_box.configure(values=masks)

    # ------------------------------------------------------------------ hardware
    def connect(self):
        if not self.session:
            messagebox.showinfo("Hardware", "Create or select a session first:\n"
                                "click 'New session…' at the top left.")
            return
        if self.rig:
            messagebox.showinfo("Hardware", "Already connected.")
            return
        self._write_log(f"Connecting to {self.session.settings['hardware']} hardware…", "head")
        self.root.update_idletasks()
        try:
            from hardware import make_rig
            self.rig = make_rig(self.session, tk_master=self.root, log=self.log)
            self.rig.open(blackfly=self.session.settings["blackfly"]["enabled"])
            self.rig.set_exposure(float(self.live_exp.get()))
        except Exception as e:  # noqa
            if self.rig is not None:
                try:
                    self.rig.close()
                except Exception:
                    pass
            self.rig = None
            self._write_log(f"Could not connect: {type(e).__name__}: {e}", "bad")
            messagebox.showerror("Hardware", f"Could not connect:\n{type(e).__name__}: {e}"); return
        self.connect_btn.configure(state="disabled"); self.disconnect_btn.configure(state="normal")
        self.log("Hardware connected.", "ok")

    def disconnect(self):
        self.live_on = False
        if self.rig:
            self.rig.close()
            self.rig = None
        self.connect_btn.configure(state="normal"); self.disconnect_btn.configure(state="disabled")
        self.log("Hardware disconnected.")

    def _need_rig(self):
        if self.lab is not None:
            from hardware import LabGuiRig, RigError, make_rig
            if self.session.settings.get("hardware") == "sim":  # simulator ignores the camera GUI
                if self.rig is None:
                    self.rig = make_rig(self.session, log=self.log)
                    self.rig.open()
                return True
            rig = self.rig if isinstance(self.rig, LabGuiRig) else LabGuiRig(self.session, self.lab, self.labmod, log=self.log)
            try:
                rig.open()
            except RigError as e:
                messagebox.showinfo("Camera controls", str(e), parent=self.root)
                return False
            self.rig = rig
            return True
        if not self.rig:
            messagebox.showinfo("Hardware", "Connect the hardware first (top right).")
            return False
        return True

    def apply_exposure(self):
        if self._need_rig():
            try:
                self.rig.set_exposure(float(self.live_exp.get()))
                self.log(f"Exposure set to {float(self.live_exp.get()):.0f} µs")
            except Exception as e:  # noqa
                messagebox.showerror("Exposure", str(e))

    def _mask_array(self, name):
        if name == "black":
            return np.zeros((1080, 1920), np.uint8)
        if name == "white":
            return np.full((1080, 1920), 255, np.uint8)
        folder = self.session.lab_path("scan_masks") if name.startswith("mask_") else self.session.lab_path("patterns")
        m = cv2.imread(str(folder / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(folder / f"{name}.png")
        return m

    def show_selected_mask(self):
        if not self._need_rig() or self.busy:
            return
        try:
            m = self._mask_array(self.mask_choice.get())
        except Exception as e:  # noqa
            messagebox.showerror("Mask", str(e)); return
        self.rig.load_masks([m]); self.rig.show(0)
        ph, _ = to_photo(m, 240, 135); self._photos["mask"] = ph
        self.mask_canvas.delete("all"); self.mask_canvas.create_image(0, 0, image=ph, anchor="nw")

    def toggle_live(self):
        if not self._need_rig():
            return
        self.live_on = not self.live_on
        self.live_btn.configure(text="Stop live view" if self.live_on else "Start live view")
        if self.live_on:
            threading.Thread(target=self._live_loop, daemon=True).start()

    def _live_loop(self):
        while self.live_on and self.rig:
            if not self.busy:
                try:
                    fr = self.rig.live_frame(downsample=6)
                    if fr is not None:
                        self.q.put(("live", fr))
                except Exception as e:  # noqa
                    self.log(f"live view: {e}"); self.live_on = False
            time.sleep(0.25)

    def _show_live(self, fr):
        w, h = self.live_canvas.winfo_width(), self.live_canvas.winfo_height()
        ph, _ = to_photo(stretch(fr), max(w, 50), max(h, 50))
        self._photos["live"] = ph
        self.live_canvas.delete("all"); self.live_canvas.create_image(0, 0, image=ph, anchor="nw")
        sat = float((fr >= 4095).mean())
        self.live_stats.configure(text=f"min {fr.min()}  mean {fr.mean():.0f}  max {fr.max()}\n"
                                       f"saturated: {sat:.2%}" + ("  ← lower exposure" if sat > 0.001 else ""))

    # ------------------------------------------------------------------ jobs
    def _start(self, what, dest="main"):
        if self.busy:
            messagebox.showinfo("Busy", f"Wait for '{self.busy}' to finish (or press Stop)."); return False
        self._adj_stop(restore=False)              # the job takes over the SLM + camera
        self.busy = what
        self.log_dest = dest
        self.ui.stop_flag.clear()
        self.stop_btn.configure(state="normal")
        self.pbar.configure(value=0)
        self.status.configure(text=f"{what}…")
        self._write_log(f"=== {what}  ({time.strftime('%H:%M:%S')}) ===", "head")
        return True

    def _job_done(self, ok, msg):
        what = self.busy
        self.busy = None
        self.proc = None
        self.stop_btn.configure(state="disabled")
        self.status.configure(text=f"{what}: {'done' if ok else 'stopped/failed'}")
        if msg:
            self._write_log(msg, "ok" if ok else "bad")
        self.log_dest = "main"
        self.refresh_status()

    def stop(self):
        self.ui.stop_flag.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.log("Stopping…", "warn")

    def run_cal_capture(self, folder):
        if not self.session or not self.save_form() or not self._need_rig():
            return
        s = self.session.settings["calibration"]
        nm, exp = next((l[1], l[2]) for l in s["lasers"] if l[0] == folder)
        extra = " plus the 4 orientation-check frames" if folder == s.get("check_laser") else ""
        if not messagebox.askokcancel(
                f"Capture {folder}",
                f"Capture the 46 Gray-code patterns under the {folder} laser ({nm:g} nm){extra}, "
                f"at {exp:.0f} µs.\n\n"
                + ("Patterns: FLIPPED left-right.\n\n" if self.flip_var.get() else "")
                + "Other lasers' captures are kept. Start?"):
            return
        if not self._start(f"Capture {folder}", dest="cal"):
            return
        self.live_on = False
        self.rig.settle_s = self.session.settings["settle_ms"] / 1000

        def work():
            import stages
            try:
                stages.calibrate_capture(self.session, self.rig, self.ui, log=self.log, laser=folder)
                missing = [l[0] for l in s["lasers"]
                           if not (self.session.cal_dir / "Captures" / l[0] / "cap_white.png").is_file()]
                nxt = ("Next: B. Process." if not missing else f"Still to capture: {', '.join(missing)}.")
                self.q.put(("done", True, f"{folder} captured. {nxt}"))
            except Exception as e:  # noqa
                self.q.put(("done", False, f"ERROR: {e}"))
        threading.Thread(target=work, daemon=True).start()

    def run_capture(self):
        if not self.session or not self.save_form() or not self._need_rig():
            return
        scene = self.scene_var.get().strip()
        if not scene:
            messagebox.showinfo("Capture", "Enter a scene name."); return
        if not self.session.table.is_file():
            if not messagebox.askyesno("Capture", "This session has no calibration table yet. Capture anyway?"):
                return
        if not self._start(f"Capture {scene}"):
            return
        self.live_on = False
        self.rig.settle_s = self.session.settings["settle_ms"] / 1000

        ui = self.ui

        def work():
            import stages
            orig = ui.progress

            def prog(i, n, text=""):
                orig(i, n, text)
                p = self.session.scene_dir(scene) / "captures" / text.split()[0] if text else None
                if p is not None and p.is_file():
                    self.q.put(("lastcap", p))
            ui.progress = prog
            try:
                m = stages.capture_scene(self.session, self.rig, scene, ui, log=self.log, use_blackfly=self.bf_var.get())
                self.q.put(("done", True, f"Scene '{scene}' captured ({m['seconds']} s). Next: 3 Cube."))
            except Exception as e:  # noqa
                self.q.put(("done", False, f"ERROR: {e}"))
            finally:
                ui.progress = orig
        threading.Thread(target=work, daemon=True).start()

    def _show_capture(self, path):
        if self.cap_canvas is None:          # the camera GUI's previews show the frames instead
            self.cap_info.configure(text=f"last frame: {path.name}")
            return
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            return
        ph, _ = to_photo(stretch(raw[::4, ::4]), 640, 430)
        self._photos["cap"] = ph
        self.cap_canvas.delete("all"); self.cap_canvas.create_image(0, 0, image=ph, anchor="nw")
        self.cap_info.configure(text=f"last frame: {path.name}   mean {raw.mean():.0f}   max {raw.max()}")

    def run_proc(self, args, what=None, dest="main"):
        """Run a processing step (`pipeline.py ...`) as a separate process, streaming its output."""
        if not self.session or not self.save_form():
            return
        what = what or " ".join(args).capitalize()
        if not self._start(what, dest):
            return
        # args use "{S}" where the session name goes, e.g. ["cube", "{S}", "scene1"]
        cmd = [self.session.processing_python, "-u", str(PIPELINE_DIR / "pipeline.py")] + \
              [self.session.name if a == "{S}" else a for a in args]
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")

        def work():
            try:
                self.proc = subprocess.Popen(cmd, cwd=str(PIPELINE_DIR), stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, text=True, env=env)
                for line in self.proc.stdout:
                    if " cell " in line and "done" in line and dest != "cal":
                        continue
                    self.log(line.rstrip())
                    if line.lstrip().startswith("[") and "/" in line.split("]")[0]:
                        try:
                            i, n = line.strip()[1:].split("]")[0].split("/")
                            self.ui.progress(int(i), int(n), "")
                        except ValueError:
                            pass
                    mb = re.search(r"band (\d+)/(\d+)", line)
                    if mb:
                        self.ui.progress(int(mb.group(1)), int(mb.group(2)), "table bands")
                rc = self.proc.wait()
                self.q.put(("done", rc == 0, None if rc == 0 else f"{what} failed or was stopped (exit {rc})."))
            except Exception as e:  # noqa
                self.q.put(("done", False, f"ERROR: {e}"))
        threading.Thread(target=work, daemon=True).start()

    def run_cube(self):
        sc = self._selected_scene()
        if sc:
            self.run_proc(["cube", "{S}", sc] + (["--force"] if self.force_var.get() else []), what=f"Cube {sc}")

    # ------------------------------------------------------------------ cube viewer
    def _selected_scene(self):
        sel = self.scene_list.curselection()
        if not sel:
            messagebox.showinfo("Scene", "Select a scene on the left."); return None
        return self.scene_list.get(sel[0]).split()[0]

    def show_scene(self):
        sel = self.scene_list.curselection()
        if not sel:
            return
        sc = self.scene_list.get(sel[0]).split()[0]
        d = self.session.scene_dir(sc)
        info = []
        if (d / "manifest.json").is_file():
            m = json.loads((d / "manifest.json").read_text())
            exp = f"{m['exposure_us']:.0f} µs" if m.get("exposure_us") else "not recorded"
            info.append(f"captured {m['captured']}\nexposure {exp}" + ("\n(imported)" if m.get("imported_from") else ""))
            if m.get("zero_drift") is not None:
                info.append(f"zero drift {m['zero_drift']:.1%}")
        if (d / "check.json").is_file():
            r = json.loads((d / "check.json").read_text())
            info.append("check: " + ("PASSED" if r["passed"] else "FAILED - " + " ".join(r["problems"])))
        if (d / "cubes" / "cube.json").is_file():
            c = json.loads((d / "cubes" / "cube.json").read_text())
            info.append(f"cube: {c['frames_ok']} frames, {c['filled_fraction']:.0%} filled")
            if "full" in c:
                f = c["full"]
                info.append(f"cube_full: {f['calibrated']:.0%} measured, {f['extended']:.0%} via extended table, "
                            f"{f['filled']:.0%} filled from neighbours")
        self.scene_info.configure(text="\n".join(info))
        pv = d / "cubes" / "preview.png"
        self.cube_canvas.delete("all")
        if pv.is_file():
            img = cv2.imread(str(pv))[..., ::-1]
            ph, sc_ = to_photo(img, 760, 430)
            self._photos["cube"] = ph
            self._cube_scale = sc_
            self.cube_canvas.create_image(0, 0, image=ph, anchor="nw")

    def load_cube(self):
        sc = self._selected_scene()
        if not sc:
            return
        p = self.session.scene_dir(sc) / "cubes" / "cube_combined.npz"
        if not p.is_file():
            messagebox.showinfo("Cube", "No cube for this scene yet - press 'Make cube'."); return
        self.status.configure(text="Loading cube…")

        def work():
            with np.load(p) as d:
                self.cube = (d["cube"], d["wavelengths"])
            self.q.put(("cube_loaded",))
        threading.Thread(target=work, daemon=True).start()

    def _cube_loaded(self):
        self.status.configure(text="Cube loaded - click the image to see spectra.")
        self.show_scene()

    def on_cube_click(self, ev):
        if self.cube is None:
            self.status.configure(text="Press 'Load cube in viewer' first.")
            return
        cube, wl = self.cube
        sc = getattr(self, "_cube_scale", 1.0)
        x, y = int(ev.x / sc), int(ev.y / sc)
        H, W = cube.shape[:2]
        if not (0 <= x < W and 0 <= y < H):
            return
        patch = cube[max(0, y - 1):y + 2, max(0, x - 1):x + 2].reshape(-1, cube.shape[2])
        with np.errstate(invalid="ignore"):
            spec = np.nanmean(patch, axis=0) if np.isfinite(patch).any() else None
        self.cube_canvas.delete("marker")
        self.cube_canvas.create_oval(ev.x - 5, ev.y - 5, ev.x + 5, ev.y + 5, outline="#ff0", width=2, tags="marker")
        self.plot_spectrum(wl, spec, f"SLM pixel ({x}, {y})")

    def plot_spectrum(self, wl, spec, title):
        c = self.spec_canvas
        c.delete("all")
        W, H, L, R, T, B = 760, 200, 50, 15, 20, 30
        c.create_text(L, 8, text=title, anchor="w", fill="#333")
        if spec is None or not np.isfinite(spec).any():
            c.create_text(W / 2, H / 2, text="no data here (pixel not sampled)", fill="#999"); return
        s = np.nan_to_num(spec)
        top = max(float(s.max()), 1e-9)
        c.create_line(L, H - B, W - R, H - B, fill="#999"); c.create_line(L, T, L, H - B, fill="#999")
        for lam in range(400, 701, 50):
            x = L + (lam - wl[0]) / (wl[-1] - wl[0]) * (W - L - R)
            c.create_line(x, H - B, x, H - B + 4, fill="#999"); c.create_text(x, H - B + 14, text=str(lam), fill="#666")
        c.create_text(W - R, H - 6, text="nm", anchor="e", fill="#666")
        c.create_text(L - 6, T, text=f"{top:.3g}", anchor="e", fill="#666")
        pts = []
        for lam, v in zip(wl, s):
            pts += [L + (lam - wl[0]) / (wl[-1] - wl[0]) * (W - L - R), H - B - v / top * (H - B - T)]
        c.create_line(*pts, fill="#2266cc", width=2)

    # ------------------------------------------------------------------ live window
    def open_live_window(self):
        lw = getattr(self, "live_window", None)
        if lw is not None and lw.alive():
            lw.win.deiconify(); lw.win.lift()
            return
        self.live_window = LiveWindow(self)

    def live_camera(self, choice):
        """The open camera object for 'basler' / 'blackfly' (or the simulator), else None."""
        if self.session is not None and self.session.settings.get("hardware") == "sim":
            if self.rig is None and self.lab is not None:
                self._need_rig()
            return self.rig if (self.rig is not None and choice == "basler") else None
        if self.lab is not None:
            return self.lab.cams.get(choice)
        return getattr(self.rig, choice, None) if self.rig is not None else None

    # ------------------------------------------------------------------ close
    def on_close(self):
        if self.busy and not messagebox.askyesno("Quit", f"'{self.busy}' is still running. Quit anyway?"):
            return
        self.stop()
        self.live_on = False
        try:
            if self.rig:
                self.rig.close()
            if self.lab is not None:
                self.lab.on_close()          # closes the SLM + cameras opened in the Capture tab
        except Exception as e:  # noqa
            print(f"(while closing: {e})")
        finally:
            try:
                self.root.destroy()
            except tk.TclError:
                pass


class LiveWindow:
    """Separate, resizable live view.

    True brightness = what a saved capture really looks like (no stretching);
    Auto-contrast   = stretched like the camera GUI's live pane (easier to see, but it
                      hides under-exposure). Saturated pixels can be shown in red. The
    readout says how full the sensor is, which is what to judge exposure by.
    Grabs on the Tk thread with a short timeout (like the camera GUI's live view) and
    pauses while a pipeline capture is running.
    """
    POLL_MS = 60

    def __init__(self, app):
        self.app = app
        self.win = tk.Toplevel(app.root)
        self.win.title("Live camera")
        self.win.geometry("1000x780")
        bar = ttk.Frame(self.win, padding=6)
        bar.pack(fill="x")
        self.cam_var = tk.StringVar(value="basler")
        ttk.Label(bar, text="Camera:").pack(side="left")
        for c, t in (("basler", "Basler (smears)"), ("blackfly", "Blackfly (RGB)")):
            ttk.Radiobutton(bar, text=t, variable=self.cam_var, value=c).pack(side="left", padx=3)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        self.mode_var = tk.StringVar(value="true")
        ttk.Radiobutton(bar, text="True brightness", variable=self.mode_var, value="true").pack(side="left")
        ttk.Radiobutton(bar, text="Auto-contrast", variable=self.mode_var, value="auto").pack(side="left", padx=3)
        self.sat_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Saturated in red", variable=self.sat_var).pack(side="left", padx=8)
        self.pause_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Pause", variable=self.pause_var).pack(side="right")
        self.stats = ttk.Label(self.win, text="", padding=(8, 0), font=("TkFixedFont", 11))
        self.stats.pack(fill="x")
        self.canvas = tk.Canvas(self.win, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.photo = None
        self._after = None
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self._tick()

    def alive(self):
        try:
            return bool(self.win.winfo_exists())
        except tk.TclError:
            return False

    def close(self):
        if self._after:
            self.win.after_cancel(self._after)
        self.win.destroy()

    def _msg(self, text):
        self.canvas.delete("all")
        self.canvas.create_text(self.canvas.winfo_width() // 2 or 400, self.canvas.winfo_height() // 2 or 300,
                                text=text, fill="#aaa", font=("TkDefaultFont", 14))

    def _tick(self):
        self._after = None
        if not self.alive():
            return
        try:
            self._update()
        except Exception as e:  # noqa
            self._msg(f"Live view error: {e}")
        self._after = self.win.after(self.POLL_MS, self._tick)

    def _update(self):
        if self.pause_var.get():
            return
        if self.app.busy:
            self._msg(f"Paused while '{self.app.busy}' runs")
            return
        choice = self.cam_var.get()
        cam = self.app.live_camera(choice)
        if cam is None:
            where = "top bar → Open" if self.app.lab is not None else "Connect hardware"
            self._msg(f"Open the {'Basler' if choice == 'basler' else 'Blackfly'} first ({where})")
            return
        if hasattr(cam, "grab_preview"):
            frame = cam.grab_preview(timeout_ms=15, downsample=(4 if choice == "basler" else 1))
        else:                                   # simulator
            frame = cam.live_frame(downsample=2)
        if frame is None:
            return
        cw, ch = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        if choice == "basler":
            raw = frame.astype(np.uint16)       # grab_preview gives raw 12-bit (0-4095)
            sat = raw >= 4095
            if self.mode_var.get() == "true":
                g = (raw >> 4).astype(np.uint8)
            else:
                lo, hi = np.percentile(raw, (1, 99.5))
                g = np.clip((raw.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)
            rgb = np.stack([g, g, g], -1)
            if self.sat_var.get():
                rgb[sat] = (255, 0, 0)
            med, p99 = np.percentile(raw, (50, 99.9))
            fs = lambda v: f"{100 * v / 4095:5.1f}%"
            hint = ("  → raise exposure" if p99 < 0.25 * 4095 else
                    "  → lower exposure" if sat.mean() > 0.001 else "  → good")
            self.stats.configure(text=f"sensor fill: typical {fs(med)}   bright parts (99.9%) {fs(p99)}   "
                                      f"saturated {100 * sat.mean():.2f}% of pixels{hint}")
            if self.app.adj_laser:
                self.app.adj_readout.configure(
                    text=f"bright parts {100 * p99 / 4095:.0f}% · saturated {100 * sat.mean():.2f}%{hint}",
                    foreground=self.app.C["bad"] if sat.mean() > 0.001 else
                    self.app.C["warn"] if p99 < 0.25 * 4095 else self.app.C["good"])
        else:
            bgr = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR) if frame.ndim == 2 else frame
            if self.mode_var.get() == "auto":
                lo, hi = np.percentile(bgr, (1, 99.5))
                bgr = np.clip((bgr.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)
            rgb = np.ascontiguousarray(bgr[..., ::-1])
            sat = frame >= 255
            if self.sat_var.get() and frame.ndim == 2:
                m = cv2.resize(sat.astype(np.uint8), (rgb.shape[1], rgb.shape[0])) > 0
                rgb[m] = (255, 0, 0)
            self.stats.configure(text=f"Blackfly: saturated {100 * sat.mean():.2f}% of pixels")
        self.photo, _ = to_photo(rgb, cw, ch)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.photo, anchor="center")


class _LabHost(ttk.Frame):
    """A frame the lab GUI can be built into as if it were its window."""
    def title(self, *_a):
        return None


def main(session_name=None, standalone=False, mirrored=False):
    root = tk.Tk()
    App(root, session_name, embed_lab=not standalone, mirrored=mirrored)
    root.mainloop()


def _import_lab_gui(session_name=None):
    """Import camera_pythonscripts/gui_test_camera_slm.py read-only.

    It keeps mask_folders.json and test_captures/ relative to the working folder, so
    the working folder is switched to pipeline/labgui_state/ first; that folder starts
    with a copy of the lab GUI's remembered mask folders plus the calibration patterns
    and scan masks. Nothing in camera_pythonscripts is written.
    """
    import importlib
    from settings import DEFAULTS
    lab_dir = Path(DEFAULTS["lab"]["camera_code"])
    lab_dir = lab_dir if lab_dir.is_absolute() else (PIPELINE_DIR / lab_dir).resolve()
    names = list_sessions()
    if session_name or names:
        s = Session(session_name or names[-1])
        lab_dir = s.lab_path("camera_code")
    state = PIPELINE_DIR / "labgui_state"
    state.mkdir(exist_ok=True)
    cfg = state / "mask_folders.json"
    if not cfg.exists():
        folders = []
        src = lab_dir / "mask_folders.json"
        if src.is_file():
            try:
                folders = json.loads(src.read_text())
            except Exception:  # noqa
                folders = []
        from settings import default_lab_path
        for extra in (default_lab_path("patterns"), default_lab_path("scan_masks")):
            p = str(extra.resolve())
            if extra.is_dir() and p not in folders:
                folders.append(p)
        cfg.write_text(json.dumps(folders, indent=2))
    sys.dont_write_bytecode = True
    if str(lab_dir) not in sys.path:
        sys.path.insert(0, str(lab_dir))
    os.chdir(state)
    return importlib.import_module("gui_test_camera_slm"), state

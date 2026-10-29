"""
Session settings and folder layout.

Everything the pipeline produces lives under  pipeline/sessions/<session>/ :

    session.json                     settings for this session (edit freely)
    calibration/
        Captures/Blue|Green|Red/     cap_00.png ... cap_43.png, cap_white.png, cap_black.png
        Captures/check/              zero / mask_00 / mask_07 / mask_14 frames for the orientation check
        proj_map/                    decoded maps + interp maps (from the lab notebooks)
        reverse_lookup.npz           THE table
        capture.json, check.json     what was captured / check result
        notebooks/                   executed copies of the lab notebooks (outputs + plots)
    scenes/<scene>/
        captures/                    zero_start.tiff, zero_end.tiff, zero.tiff, scan_00..55.tiff, rgb/
        manifest.json                how the scene was captured
        cubes/                       cube_00..55.npz, cube_combined.npz, preview.png

Lab code (calibration notebooks, camera_pythonscripts, scan masks) is only ever READ.
"""

import copy
import json
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = PIPELINE_DIR / "sessions"

DEFAULTS = {
    # Where the lab's code lives (relative to the pipeline folder, or absolute). Read-only.
    "lab": {
        # The pipeline lives in camera_pythonscripts/pipeline/; the calibration repo sits next to
        # camera_pythonscripts:  <lab>/camera_pythonscripts/pipeline   <lab>/calibration
        # The scan masks are kept inside the pipeline (masks/scan_masks, in git).
        "calibration_code": "../../calibration",
        "camera_code": "../camera",       # drivers + camera GUI (camera_pythonscripts/camera/)
        "scan_masks": "masks/scan_masks",
        "patterns": "../../calibration/patterns",
    },
    # "real" = the lab hardware via camera_pythonscripts; "sim" = built-in simulator (no hardware).
    "hardware": "real",
    # Python used for processing (notebooks need opencv-contrib, scipy, nbclient, ipykernel).
    # Blank = the same Python that is running the pipeline.
    "processing_python": "",
    "slm": {"monitor_index": None, "mirrored": False},
    "basler": {"serial": None, "gain_db": 0.0},
    "blackfly": {"enabled": True, "serial": None, "gain_db": 0.0, "exposure_us": 20000},
    # Time to wait after changing the SLM image before grabbing (one stale frame is
    # also thrown away after the wait, like the lab GUI does).
    "settle_ms": 300,
    "calibration": {
        # (folder name, laser wavelength nm, Basler exposure us) -- in capture order.
        "lasers": [["Blue", 405, 50000], ["Green", 532, 50000], ["Red", 635, 50000]],
        # Laser (folder name) used for the orientation-check frames.
        "check_laser": "Green",
        # Use the calibration of another session instead of this one's (blank = this session).
        "use_from_session": "",
        # Flip the Gray-code patterns left-right before showing them on the SLM (calibration
        # only; scan masks are not flipped). All three lasers must be captured the same way.
        "flip_patterns_x": False,
    },
    "capture": {
        "exposure_us": 100000,
        # all-zero (black) captures: one at the start and one at the end, averaged.
        "zero_at_start": True,
        "zero_at_end": True,
    },
    "cube": {
        # Stop before making cubes if the orientation check on the scene fails.
        "require_check": True,
        # Also write cube_full.npz: a value at every pixel and band (see full_cube.py).
        "full": True,
    },
}


def _merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


# Defaults from before the pipeline moved into camera_pythonscripts/ (sessions made then
# stored these in session.json); they are swapped for the current defaults on load.
_OLD_LAB_DEFAULTS = {"calibration_code": ["../calibration"], "camera_code": ["../camera_pythonscripts", ".."],
                     "scan_masks": ["../slm_scan/scan_masks", "../../slm_scan/scan_masks"],
                     "patterns": ["../calibration/patterns"]}


def default_lab_path(key) -> Path:
    """A lab folder from the DEFAULTS (used where no session is involved)."""
    p = Path(DEFAULTS["lab"][key]).expanduser()
    return p if p.is_absolute() else (PIPELINE_DIR / p).resolve()


class Session:
    def __init__(self, name: str, root: Path = SESSIONS_DIR):
        if not name or any(c in name for c in "/\\:"):
            raise ValueError(f"Bad session name: {name!r}")
        self.name = name
        self.dir = Path(root) / name
        self.settings_path = self.dir / "session.json"
        self.settings = _merge(DEFAULTS, self._read())
        for k, old in _OLD_LAB_DEFAULTS.items():
            if self.settings["lab"].get(k) in old:
                self.settings["lab"][k] = DEFAULTS["lab"][k]

    # ---------------------------------------------------------------- settings
    def _read(self):
        if self.settings_path.is_file():
            return json.loads(self.settings_path.read_text())
        return {}

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(json.dumps(self.settings, indent=2) + "\n")

    @classmethod
    def create(cls, name, **overrides):
        s = cls(name)
        s.settings = _merge(s.settings, overrides)
        s.save()
        return s

    def get(self, *keys):
        v = self.settings
        for k in keys:
            v = v[k]
        return v

    def lab_path(self, key) -> Path:
        p = Path(self.settings["lab"][key]).expanduser()
        return p if p.is_absolute() else (PIPELINE_DIR / p).resolve()

    @property
    def processing_python(self) -> str:
        import sys
        return self.settings.get("processing_python") or sys.executable

    # ---------------------------------------------------------------- folders
    @property
    def cal_dir(self) -> Path:
        """Calibration folder actually used (may come from another session)."""
        other = self.settings["calibration"].get("use_from_session")
        return (SESSIONS_DIR / other / "calibration") if other else (self.dir / "calibration")

    @property
    def table(self) -> Path:
        return self.cal_dir / "reverse_lookup.npz"

    def scene_dir(self, scene) -> Path:
        if not scene or any(c in scene for c in "/\\:"):
            raise ValueError(f"Bad scene name: {scene!r}")
        return self.dir / "scenes" / scene

    def scenes(self):
        d = self.dir / "scenes"
        return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []


def list_sessions():
    return sorted(p.name for p in SESSIONS_DIR.iterdir() if (p / "session.json").is_file()) \
        if SESSIONS_DIR.is_dir() else []

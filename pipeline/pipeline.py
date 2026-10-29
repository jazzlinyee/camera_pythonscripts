#!/usr/bin/env python3
"""
Coded-aperture hyperspectral pipeline: calibrate -> capture -> cube.

    python pipeline.py new <session> [--sim]              make a session (edit sessions/<session>/session.json)
    python pipeline.py calibrate capture <session>       Gray codes under each laser (needs the lab hardware)
    python pipeline.py calibrate process <session>       lab notebooks -> table (reverse_lookup.npz)
    python pipeline.py calibrate check   <session>       does the table match the hardware's orientation?
    python pipeline.py capture <session> <scene>         all-zero + 56 scan masks of a scene
    python pipeline.py cube    <session> <scene> [--force]
    python pipeline.py gui                               camera GUI + pipeline window (add --standalone for pipeline only)
    python pipeline.py status  <session>

The lab's files (calibration notebooks, camera_pythonscripts, scan masks) are only read.
"""

import argparse
import sys

sys.dont_write_bytecode = True   # keep every folder free of __pycache__

from settings import Session, list_sessions  # noqa: E402


def _rig(session):
    from hardware import make_rig
    return make_rig(session)


def cmd_new(a):
    s = Session(a.session)
    if s.settings_path.exists():
        sys.exit(f"Session {a.session} already exists: {s.settings_path}")
    extra = {"hardware": "sim"} if a.sim else {}
    if a.use_calibration_from:
        extra["calibration"] = {"use_from_session": a.use_calibration_from}
    s = Session.create(a.session, **extra)
    print(f"Created {s.settings_path}\nCheck the exposures and settings in it before capturing.")


def cmd_status(a):
    s = Session(a.session)
    print(f"Session {s.name}  ({s.dir})  hardware={s.settings['hardware']}")
    cal = s.cal_dir
    for label, p in [("calibration captures", cal / "capture.json"), ("table", cal / "reverse_lookup.npz"),
                     ("calibration check", cal / "check.json")]:
        print(f"  {label:22s} {'yes' if p.is_file() else '-'}")
    if (cal / "check.json").is_file():
        import json
        r = json.loads((cal / "check.json").read_text())
        print(f"  {'check result':22s} {'PASSED' if r['passed'] else 'FAILED: ' + ' '.join(r['problems'])}")
    for sc in s.scenes():
        d = s.scene_dir(sc)
        print(f"  scene {sc:16s} captured={'yes' if (d / 'manifest.json').is_file() else '-'} "
              f"cube={'yes' if (d / 'cubes' / 'cube_combined.npz').is_file() else '-'}")


def cmd_calibrate(a):
    import stages
    s = Session(a.session)
    if a.step == "capture":
        rig = _rig(s)
        rig.open(blackfly=False)
        try:
            stages.calibrate_capture(s, rig, stages.ConsoleUI(), laser=a.laser)
        finally:
            rig.close()
    elif a.step == "process":
        steps = a.only.split(",") if a.only else ("graycode", "interpolation", "reverse", "unflip")
        stages.calibrate_process(s, steps=steps)
        stages.calibrate_check(s)
    else:
        stages.calibrate_check(s)


def cmd_capture(a):
    import stages
    s = Session(a.session)
    rig = _rig(s)
    rig.open(blackfly=s.settings["blackfly"]["enabled"])
    try:
        stages.capture_scene(s, rig, a.scene, stages.ConsoleUI())
    finally:
        rig.close()


def cmd_position(a):
    import stages
    s = Session(a.session)
    rig = _rig(s)
    rig.open(blackfly=s.settings["blackfly"]["enabled"])
    try:
        stages.capture_position(s, rig, a.scene, stages.ConsoleUI(), mask_path=a.mask)
    finally:
        rig.close()


def cmd_sparse_export(a):
    import stages
    stages.export_positions(Session(a.session), a.scene, background="none")


def cmd_cube(a):
    import stages
    stages.make_cube(Session(a.session), a.scene, force=a.force)


def cmd_gui(a):
    import gui
    gui.main(a.session, standalone=a.standalone, mirrored=a.mirrored)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new"); n.add_argument("session"); n.add_argument("--sim", action="store_true")
    n.add_argument("--use-calibration-from", help="reuse another session's calibration"); n.set_defaults(f=cmd_new)
    st = sub.add_parser("status"); st.add_argument("session"); st.set_defaults(f=cmd_status)
    c = sub.add_parser("calibrate"); c.add_argument("step", choices=["capture", "process", "check"])
    c.add_argument("session"); c.add_argument("--only", help="process only some steps: graycode,interpolation,reverse,unflip")
    c.add_argument("--laser", help="capture only this laser (folder name, e.g. Blue); default: all, in list order")
    c.set_defaults(f=cmd_calibrate)
    cp = sub.add_parser("capture"); cp.add_argument("session"); cp.add_argument("scene"); cp.set_defaults(f=cmd_capture)
    po = sub.add_parser("position", help="capture one position of a moving object (sparse mask + black)")
    po.add_argument("session"); po.add_argument("scene"); po.add_argument("--mask", help="default: scan mask_00")
    po.set_defaults(f=cmd_position)
    se = sub.add_parser("sparse-export", help="sparse cube + RGB for every captured position")
    se.add_argument("session"); se.add_argument("scene"); se.set_defaults(f=cmd_sparse_export)
    cu = sub.add_parser("cube"); cu.add_argument("session"); cu.add_argument("scene")
    cu.add_argument("--force", action="store_true", help="make the cube even if the orientation check fails")
    cu.set_defaults(f=cmd_cube)
    g = sub.add_parser("gui"); g.add_argument("session", nargs="?")
    g.add_argument("--standalone", action="store_true", help="pipeline window only, without the camera GUI")
    g.add_argument("--mirrored", action="store_true", help="passed to the camera GUI (SLM mirrors the screen)")
    g.set_defaults(f=cmd_gui)
    l = sub.add_parser("sessions"); l.set_defaults(f=lambda a: print("\n".join(list_sessions()) or "(none)"))
    a = p.parse_args(argv)
    try:
        a.f(a)
    except KeyboardInterrupt:
        sys.exit("\nStopped.")
    except Exception as e:  # noqa
        from labcode import LabCodeError
        from stages import StageError
        if isinstance(e, (StageError, LabCodeError, ValueError)):
            sys.exit(f"\nERROR: {e}")
        raise


if __name__ == "__main__":
    main()

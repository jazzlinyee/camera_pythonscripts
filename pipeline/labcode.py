"""
Run the lab's own code without modifying it.

* Notebooks / scripts are read, their hard-coded file paths are swapped IN MEMORY,
  and the patched copy is executed. The original files are never written.
  Every swap must match an expected number of times; if the lab code changes
  shape, this stops with a clear message instead of guessing.
* camera_pythonscripts is imported with bytecode writing switched off, so no
  __pycache__ / .pyc files inside that (git-tracked) folder are created or changed.
"""

import copy
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
_STRING = re.compile(r"""(?P<prefix>(?<!\w)[fFrRbB]{1,2})?(?P<q>["'])(?P<body>.*?)(?P=q)""")


class LabCodeError(RuntimeError):
    pass


# ----------------------------------------------------------------- path swapping
class Rule:
    """Replace the contents of string literals that fully match `pattern`.

    replacement may use \\1-style groups. `count` = how many literals must match
    (in non-commented lines); anything else is an error.
    """

    def __init__(self, name, pattern, replacement, count):
        self.name, self.rx, self.repl, self.count = name, re.compile(pattern), replacement, count


def patch_source(src: str, rules, found=None):
    found = found if found is not None else {r.name: 0 for r in rules}
    out = []
    for line in src.split("\n"):
        if line.lstrip().startswith("#"):
            out.append(line)
            continue
        code, sep, comment = _split_comment(line)

        def sub(m):
            body = m.group("body")
            for r in rules:
                mm = r.rx.fullmatch(body)
                if mm:
                    found[r.name] += 1
                    # only \1, \2 ... are special; everything else in the replacement is literal
                    new = re.sub(r"\\(\d)", lambda g: mm.group(int(g.group(1))), r.repl)
                    if m.group("q") in new:
                        raise LabCodeError(f"The path {new!r} contains a quote character; rename the folder.")
                    return f"{m.group('prefix') or ''}{m.group('q')}{new}{m.group('q')}"
            return m.group(0)

        out.append(_STRING.sub(sub, code) + sep + comment)
    return "\n".join(out), found


def _split_comment(line):
    """Split off a trailing # comment that is not inside a string."""
    q = None
    for i, ch in enumerate(line):
        if q:
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
        elif ch == "#":
            return line[:i], "#", line[i + 1:]
    return line, "", ""


def _check(found, rules, what):
    bad = {r.name: (found[r.name], r.count) for r in rules if found[r.name] != r.count}
    if bad:
        detail = ", ".join(f"{k}: found {f}, expected {e}" for k, (f, e) in bad.items())
        raise LabCodeError(f"{what} doesn't look the way the pipeline expects ({detail}). "
                           f"The lab code may have changed - the pipeline did not run it.")


# ----------------------------------------------------------------- notebooks
class CodeRule:
    """Replace an exact piece of code text (not a string literal), e.g. a hard-coded
    number list. `count` = how many times it must occur across the notebook's code cells."""

    def __init__(self, name, old, new, count):
        self.name, self.old, self.new, self.count = name, old, new, count


def run_notebook(nb_path, rules, workdir, executed_copy=None, log=print, slots=None, code_rules=()):
    """Execute a patched copy of a notebook. Returns True/False; logs the failing cell."""
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb_path = Path(nb_path)
    nb = nbformat.read(nb_path, as_version=4)
    nb = copy.deepcopy(nb)
    found = {r.name: 0 for r in rules}
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    for c in code_cells:
        c.source, _ = patch_source(c.source, rules, found)
    _check(found, rules, nb_path.name)
    for cr in code_rules:
        n = sum(c.source.count(cr.old) for c in code_cells)
        if n != cr.count:
            raise LabCodeError(f"{nb_path.name} doesn't look the way the pipeline expects "
                               f"({cr.name}: found {n}, expected {cr.count}). The pipeline did not run it.")
        for c in code_cells:
            c.source = c.source.replace(cr.old, cr.new)

    Path(workdir).mkdir(parents=True, exist_ok=True)
    env_before = os.environ.get("PYTHONDONTWRITEBYTECODE")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    def on_cell(cell, cell_index, **kw):
        n = code_cells.index(cell) + 1 if cell in code_cells else "?"
        log(f"    cell {n}/{len(code_cells)} done")
        for o in cell.get("outputs", []):
            if o.get("output_type") == "stream":
                for ln in o["text"].rstrip().splitlines()[-6:]:
                    log("      " + ANSI.sub("", ln))

    client = NotebookClient(nb, timeout=None,
                            kernel_name=nb.metadata.get("kernelspec", {}).get("name", "python3"),
                            resources={"metadata": {"path": str(workdir)}},
                            on_cell_executed=on_cell)
    ok = True
    try:
        client.execute(stderr=subprocess.DEVNULL)
    except CellExecutionError:
        ok = False
        for idx, c in enumerate(code_cells):
            for o in c.get("outputs", []):
                if o.get("output_type") == "error":
                    log(f"  ERROR in {nb_path.name} cell {idx + 1}: {o['ename']}: {ANSI.sub('', o['evalue'])}")
    finally:
        if env_before is None:
            os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        if executed_copy:
            Path(executed_copy).parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(nb, executed_copy)
    return ok


# ----------------------------------------------------------------- scripts
def run_script(py_path, rules, workdir, python=None, log=print):
    """Run a patched copy of a .py script in a subprocess, streaming its output."""
    src = Path(py_path).read_text()
    patched, found = patch_source(src, rules)
    _check(found, rules, Path(py_path).name)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.Popen([python or sys.executable, "-c", patched], cwd=str(workdir),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in proc.stdout:
        log("    " + line.rstrip())
    return proc.wait() == 0


# ----------------------------------------------------------------- camera code
def import_camera_code(folder):
    """Import the lab's camera/SLM modules read-only (no .pyc written into that folder)."""
    folder = str(Path(folder).resolve())
    if not Path(folder).is_dir():
        raise LabCodeError(f"camera_pythonscripts folder not found: {folder}")
    sys.dont_write_bytecode = True
    if folder not in sys.path:
        sys.path.insert(0, folder)
    mods = {}
    for name in ("camera_base", "slm_display", "basler_camera", "blackfly_camera"):
        mods[name] = importlib.import_module(name)
    return mods

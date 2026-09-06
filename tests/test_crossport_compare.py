"""The cross-port harness is the repo's strongest claim; this pins that
the documented invocation actually works (review C2: ``compare.py``
used to join every argument onto its own directory, so the LEARN.md
command ``python tools/crossport/compare.py tools/crossport/java.out
/tmp/py_check.out`` raised FileNotFoundError)."""

import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS = os.path.join(REPO, "tools", "crossport")
COMPARE = os.path.join(HARNESS, "compare.py")


def _run(*args, cwd=REPO):
    return subprocess.run([sys.executable, COMPARE, *args], cwd=cwd,
                          capture_output=True, text=True, check=True).stdout


def test_repo_relative_paths_work_from_the_repo_root():
    out = _run("tools/crossport/java.out", "tools/crossport/py.out")
    assert "156 labels, 0 mismatches" in out


def test_absolute_and_bare_names_work(tmp_path):
    copy = tmp_path / "py_check.out"
    shutil.copy(os.path.join(HARNESS, "py.out"), copy)
    out = _run(os.path.join(HARNESS, "java.out"), str(copy))
    assert "156 labels, 0 mismatches" in out
    # Bare names resolve beside the script from ANY working directory.
    out = _run("java.out", "cpp.out", cwd=str(tmp_path))
    assert "156 labels, 0 mismatches" in out


def test_checked_in_outputs_agree_pairwise():
    assert "0 mismatches" in _run("java.out", "cpp.out")
    assert "0 mismatches" in _run("java.out", "py.out")
    assert "0 mismatches" in _run("cpp.out", "py.out")

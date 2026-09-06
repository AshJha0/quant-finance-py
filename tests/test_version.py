"""Version pin plus the documentation-count guard.

The v1.0.0 review found README/LEARN quoting module, test and
probe-label counts from an earlier phase. These tests re-derive
every such number from the tree and fail when the docs drift, so a
stale count is a red test rather than a review finding.
"""

import os
import re
import subprocess
import sys

import pytest

import quantfinlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src", "quantfinlib")
DOCS = os.path.join(REPO, "docs")


def _read(*parts):
    with open(os.path.join(REPO, *parts), encoding="utf-8") as f:
        return f.read()


def test_version_is_set():
    assert quantfinlib.__version__ == "1.0.0"


def _module_count():
    n = 0
    for _, _, files in os.walk(SRC):
        n += sum(1 for f in files if f.endswith(".py") and f != "__init__.py")
    return n


def _collected_test_count():
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    m = re.search(r"(\d+) tests? collected", out)
    assert m, out[-500:]
    return int(m.group(1))


def _label_count():
    with open(os.path.join(REPO, "tools", "crossport", "java.out")) as f:
        return sum(1 for line in f if "=" in line)


def test_readme_module_and_test_counts_match_the_tree():
    readme = _read("README.md")
    m = re.search(r"(\d+) modules - (\d+) tests", readme)
    assert m, "README must carry the 'N modules - M tests' line"
    assert int(m.group(1)) == _module_count()
    assert int(m.group(2)) == _collected_test_count()


def test_learn_counts_match_the_tree():
    learn = _read("docs", "LEARN.md")
    m = re.search(r"ships (\d+) modules and (\d+) tests", learn)
    assert m, "LEARN.md Section 1 must state 'ships N modules and M tests'"
    assert int(m.group(1)) == _module_count()
    assert int(m.group(2)) == _collected_test_count()


def test_every_quoted_label_count_matches_java_out():
    labels = _label_count()
    for doc in ("README.md", os.path.join("docs", "LEARN.md"), os.path.join("docs", "COOKBOOK.md")):
        text = _read(doc)
        quoted = {int(x) for x in re.findall(r"(\d+)[- ]label", text)}
        quoted |= {int(x) for x in re.findall(r"(\d+) labelled", text)}
        assert quoted == {labels}, f"{doc} quotes label counts {quoted}, java.out has {labels}"
    for name in ("java.out", "cpp.out", "py.out"):
        with open(os.path.join(REPO, "tools", "crossport", name)) as f:
            assert sum(1 for line in f if "=" in line) == labels


def test_learn_probe_line_counts_match():
    learn = _read("docs", "LEARN.md")
    m = re.search(r"probe\.py\s+-- this port's mirror \((\d+) lines", learn)
    assert m
    with open(os.path.join(REPO, "tools", "crossport", "probe.py")) as f:
        assert int(m.group(1)) == sum(1 for _ in f)
    m = re.search(r"java\.out\s+-- checked-in Java output \((\d+) lines\)", learn)
    assert m
    with open(os.path.join(REPO, "tools", "crossport", "java.out")) as f:
        assert int(m.group(1)) == sum(1 for _ in f)


def test_architecture_package_counts_match_the_tree():
    arch = _read("docs", "ARCHITECTURE.md")
    for pkg, count in re.findall(r'\["([a-z]+) \((\d+)\)"\]', arch):
        name = "microstructure" if pkg == "micro" else pkg
        d = os.path.join(SRC, name)
        actual = sum(1 for _, _, fs in os.walk(d)
                     for f in fs if f.endswith(".py") and f != "__init__.py")
        assert int(count) == actual, f"{name}: ARCHITECTURE says {count}, tree has {actual}"
    m = re.search(r"\((\d+) modules\s+under `src/quantfinlib/`, (\d+) packages\)", arch)
    assert m and int(m.group(1)) == _module_count()
    assert int(m.group(2)) == sum(1 for d in os.listdir(SRC)
                                  if os.path.isdir(os.path.join(SRC, d)) and d != "__pycache__")


def test_pyproject_requires_python_matches_the_interpreter_the_suite_runs_on():
    m = re.search(r'requires-python = ">=(\d+)\.(\d+)"', _read("pyproject.toml"))
    assert m
    assert (int(m.group(1)), int(m.group(2))) <= sys.version_info[:2]


def _cookbook_recipes():
    text = _read("docs", "COOKBOOK.md")
    return re.findall(r"### (\d+)\. (.*?)\n.*?```python\n(.*?)```\n\n```\n(.*?)```", text, flags=re.S)


def test_cookbook_has_at_least_fifteen_recipes():
    recipes = _cookbook_recipes()
    assert len(recipes) >= 15
    m = re.search(r"(\d+) recipes\.", _read("docs", "COOKBOOK.md"))
    assert m and int(m.group(1)) == len(recipes)


@pytest.mark.parametrize("number,title,code,expected",
                         _cookbook_recipes(),
                         ids=[f"recipe{n}" for n, *_ in _cookbook_recipes()])
def test_cookbook_recipe_output_is_current(number, title, code, expected, tmp_path):
    script = tmp_path / f"recipe_{number}.py"
    script.write_text(code)
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "src"))
    r = subprocess.run([sys.executable, str(script)], cwd=REPO, capture_output=True,
                       text=True, env=env)
    assert r.returncode == 0, f"recipe {number} ({title}) failed:\n{r.stderr}"
    got = [ln for ln in r.stdout.rstrip("\n").splitlines() if "/tmp" not in ln]
    want = [ln for ln in expected.rstrip("\n").splitlines() if "/tmp" not in ln]
    assert got == want, f"recipe {number} ({title}) output drifted"

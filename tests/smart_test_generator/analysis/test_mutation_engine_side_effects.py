import os
import subprocess
from pathlib import Path

import pytest

from smart_test_generator.analysis.mutation_engine import MutationTestingEngine


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.mark.timeout(15)
def test_mutation_engine_does_not_touch_working_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Regression test: mutation testing must not mutate the working tree.

    Current behavior (bug): `_run_tests_against_mutant` renames the original source
    to `<file>.backup` and swaps in the mutant directly in-place, then restores it
    later. This still performs writes/renames in the working tree.

    This test asserts the desired behavior: NO rename/remove operations should target
    the original working tree file path. It will FAIL on current implementation and
    should pass once sandboxed execution is introduced.
    """

    # Create minimal project
    project_root = tmp_path
    pkg_dir = project_root / "pkg"
    tests_dir = project_root / "tests"
    mod_file = pkg_dir / "mod.py"
    test_file = tests_dir / "test_mod.py"

    _write_file(pkg_dir / "__init__.py", "")
    _write_file(
        mod_file,
        """
def add(a, b):
    return a + b
""".strip(),
    )

    # Ensure pytest can import the package when executed from a different cwd
    _write_file(
        test_file,
        """
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pkg.mod import add

def test_add():
    assert add(1, 2) == 3
""".strip(),
    )

    # Record filesystem write operations
    recorded_ops = []

    real_rename = os.rename
    real_remove = os.remove

    def recording_rename(src: str, dst: str):
        recorded_ops.append(("rename", Path(src), Path(dst)))
        return real_rename(src, dst)

    def recording_remove(path: str):
        recorded_ops.append(("remove", Path(path)))
        return real_remove(path)

    monkeypatch.setattr(os, "rename", recording_rename)
    monkeypatch.setattr(os, "remove", recording_remove)

    # Avoid actually running pytest; return a successful/failed result quickly
    class DummyCompleted:
        returncode = 1  # non-zero -> mutant considered killed
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: DummyCompleted())

    # Execute mutation testing (limit to 1 mutant for speed)
    engine = MutationTestingEngine()
    engine.run_mutation_testing(str(mod_file), [str(test_file)], max_mutants=1)

    # Desired invariant: no direct rename/remove should target the working tree source
    orig = mod_file.resolve()
    backup = Path(str(orig) + ".backup")

    touched_working_tree = any(
        (
            op == "rename" and (src == orig or dst == orig or src == backup or dst == backup)
        )
        or (op == "remove" and target == orig)
        for (op, *paths) in recorded_ops
        for (src, dst) in ([paths] if op == "rename" else [(None, None)])
        for target in ([paths[0]] if op == "remove" else [None])
    )

    assert not touched_working_tree, (
        "Mutation engine must not rename/remove files in the working tree; run in an isolated sandbox instead"
    )


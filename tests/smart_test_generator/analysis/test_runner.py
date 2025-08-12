import sys
from pathlib import Path
from smart_test_generator.analysis.coverage.runner import run_pytest, RunResult
from smart_test_generator.analysis.coverage.command_builder import CommandSpec


def test_run_pytest_captures_and_writes_artifacts(tmp_path: Path):
    # Create a dummy test file
    test_dir = tmp_path
    (test_dir / "test_dummy.py").write_text("def test_ok():\n    assert 1 == 1\n")

    spec = CommandSpec(argv=[sys.executable, "-m", "pytest", "-q"], cwd=test_dir, env={})
    res: RunResult = run_pytest(spec, artifacts_dir=test_dir / ".artifacts" / "coverage" / "test")

    assert isinstance(res.returncode, int)
    # stdout should contain 'passed' or returncode 0
    assert res.returncode == 0
    # Artifacts exist
    assert (test_dir / ".artifacts" / "coverage" / "test" / "stdout.txt").exists()
    assert (test_dir / ".artifacts" / "coverage" / "test" / "stderr.txt").exists()
    assert (test_dir / ".artifacts" / "coverage" / "test" / "cmd.json").exists()


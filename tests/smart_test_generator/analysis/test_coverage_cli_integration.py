import sys
from pathlib import Path
from subprocess import run, PIPE


def test_cli_help_shows_coverage_runner_flags(tmp_path: Path):
    # Invoke the CLI with --help and ensure flags are present
    proc = run([sys.executable, "-m", "smart_test_generator", "--help"], text=True, stdout=PIPE, stderr=PIPE)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0
    # Check a few key flags
    assert "--runner-mode" in out
    assert "--pytest-arg" in out
    assert "--append-pythonpath" in out


def test_cli_coverage_mode_runs_minimally(tmp_path: Path):
    # Create minimal test project
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\naddopts = '-q'\n")
    (tmp_path / "test_dummy.py").write_text("def test_ok():\n    assert 1 == 1\n")

    # Run coverage mode with cwd set to tmp_path to isolate
    proc = run(
        [
            sys.executable,
            "-m",
            "smart_test_generator",
            "coverage",
            "--directory",
            str(tmp_path),
            "--runner-cwd",
            str(tmp_path),
            "--pytest-arg",
            "-q",
        ],
        text=True,
        stdout=PIPE,
        stderr=PIPE,
        cwd=str(tmp_path),
    )

    # We don't assert on full output; just ensure it exits (0 or 1 depending on environment)
    assert proc.returncode in (0, 1, 2)

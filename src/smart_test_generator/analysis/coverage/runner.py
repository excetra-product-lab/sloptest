"""Process runner for executing pytest with structured capture and artifacts."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

from .command_builder import CommandSpec


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    duration: float
    cmd: List[str]
    cwd: str
    junit_xml_path: Optional[str] = None


def run_pytest(
    spec: CommandSpec,
    *,
    timeout: Optional[int] = None,
    stream: bool = False,
    artifacts_dir: Optional[Path] = None,
    junit_xml: bool = False,
) -> RunResult:
    """Run the provided command spec (typically pytest) and capture structured results.

    - Captures stdout/stderr, return code, and duration
    - Writes artifacts (stdout.txt, stderr.txt, cmd.json) to artifacts_dir if provided,
      else to default: spec.cwd/.artifacts/coverage/<epoch_ms>
    - Raises TimeoutError on timeout with partial outputs included in the exception message
    """
    start = time.time()
    try:
        # Prepare artifacts dir and optional junit path
        final_artifacts_dir = artifacts_dir or _default_artifacts_dir(spec.cwd)
        junit_path = None
        argv = list(spec.argv)
        if junit_xml:
            junit_path = str(final_artifacts_dir / "junit.xml")
            argv = [*argv, "--junitxml", junit_path]
        # For now, stream is informational only; we always capture fully
        completed = subprocess.run(
            argv,
            cwd=spec.cwd,
            env=spec.env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        duration = time.time() - start

        result = RunResult(
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            duration=duration,
            cmd=list(argv),
            cwd=str(spec.cwd),
            junit_xml_path=junit_path,
        )

        # Write artifacts
        _write_artifacts(result, final_artifacts_dir)
        return result

    except subprocess.TimeoutExpired as e:
        duration = time.time() - start
        partial_stdout = e.stdout or ""
        partial_stderr = e.stderr or ""
        # Write partial artifacts
        partial_result = RunResult(
            returncode=-1,
            stdout=partial_stdout,
            stderr=partial_stderr,
            duration=duration,
            cmd=list(spec.argv),
            cwd=str(spec.cwd),
        )
        _write_artifacts(partial_result, artifacts_dir or _default_artifacts_dir(spec.cwd))
        raise TimeoutError(
            f"Command timed out after {timeout}s. Partial stdout/stderr saved to artifacts."
        ) from e


def _default_artifacts_dir(cwd: Path) -> Path:
    ts = int(time.time() * 1000)
    return cwd / ".artifacts" / "coverage" / str(ts)


def _write_artifacts(result: RunResult, base_dir: Path) -> None:
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
        (base_dir / "stdout.txt").write_text(result.stdout)
        (base_dir / "stderr.txt").write_text(result.stderr)
        cmd_payload: Dict[str, object] = {
            "argv": result.cmd,
            "cwd": result.cwd,
            "returncode": result.returncode,
            "duration": result.duration,
        }
        (base_dir / "cmd.json").write_text(json.dumps(cmd_payload, indent=2))
    except Exception:
        # Best-effort artifacts; ignore failures
        pass


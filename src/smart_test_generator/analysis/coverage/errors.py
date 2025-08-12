"""Error parsing utilities for pytest runs with actionable guidance."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class ErrorReport:
    kind: str  # 'env' | 'pytest' | 'timeout' | 'unknown'
    reasons: List[str]
    guidance: List[str]
    excerpts: List[str]


def parse_errors(run_result, preflight: Dict) -> ErrorReport:
    """Parse pytest run output into a structured error report with guidance."""
    text = (run_result.stderr or "") + "\n" + (run_result.stdout or "")
    lines = text.splitlines()

    reasons: List[str] = []
    guidance: List[str] = []
    excerpts: List[str] = _tail(lines)

    # 1) Environment/Import issues
    if re.search(r"ModuleNotFoundError: No module named ", text) or re.search(r"ImportError", text):
        reasons.append("Python import failed (missing module)")
        if not preflight.get("venv_active", True):
            guidance.append("Activate your virtualenv (e.g., 'source .venv/bin/activate').")
        guidance.append("Install missing packages in the active environment (pip install <package>).")
        return ErrorReport(kind="env", reasons=reasons, guidance=guidance, excerpts=excerpts)

    if "pytest: command not found" in text:
        reasons.append("pytest is not on PATH")
        guidance.append("Use 'python -m pytest' or set runner.mode='python-module' in config.")
        return ErrorReport(kind="env", reasons=reasons, guidance=guidance, excerpts=excerpts)

    if re.search(r"No module named pip", text) or re.search(r"pip(3)?: command not found", text):
        reasons.append("pip is not available in this environment")
        if not preflight.get("venv_active", True):
            guidance.append("Activate your virtualenv and ensure pip is installed.")
        guidance.append("Install pip or use the interpreter's ensurepip module: 'python -m ensurepip --upgrade'.")
        return ErrorReport(kind="env", reasons=reasons, guidance=guidance, excerpts=excerpts)

    if re.search(r"pyproject\.toml .* not found", text, flags=re.IGNORECASE):
        reasons.append("pyproject.toml not found")
        guidance.append("Run from the project root or set runner.cwd in config.")
        return ErrorReport(kind="env", reasons=reasons, guidance=guidance, excerpts=excerpts)

    # 2) Pytest collection or test failures
    collected_match = re.search(r"collected\s+(\d+)\s+items?", text)
    if "ERROR collecting" in text:
        reasons.append("Pytest collection error")
        guidance.append("Reproduce with the printed command and fix import/collection issues shown above.")
        return ErrorReport(kind="pytest", reasons=reasons, guidance=guidance, excerpts=excerpts)

    if collected_match or run_result.returncode != 0:
        # If tests were collected, treat as test failure
        if collected_match:
            reasons.append("Tests ran and failed")
        else:
            reasons.append("Pytest exited with a non-zero status")
        guidance.append("Reproduce locally using the printed command; inspect test failures.")
        return ErrorReport(kind="pytest", reasons=reasons, guidance=guidance, excerpts=excerpts)

    # 3) Unknown
    return ErrorReport(kind="unknown", reasons=["Unknown error"], guidance=["Check logs"], excerpts=excerpts)


def _tail(lines: List[str], n: int = 50) -> List[str]:
    return lines[-n:]


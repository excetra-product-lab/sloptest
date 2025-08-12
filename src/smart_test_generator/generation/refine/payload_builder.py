"""Build a structured payload and prompt for LLM-driven test refinement.

The payload includes:
- run_id, repo meta (best-effort), environment summary
- config summary (selected keys)
- recent test changes (optional git diff placeholder)
- failures (normalized records with snippets)

Also builds a best-practice refinement prompt following configured guidelines.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from smart_test_generator.config import Config
from smart_test_generator.analysis.coverage.failure_parser import ParsedFailures, FailureRecord


@dataclass
class CodeSnippet:
    path: str
    start_line: int
    end_line: int
    snippet: str


def _read_snippet(file_path: Path, line: Optional[int], radius: int = 20) -> Optional[CodeSnippet]:
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line is None:
            start, end = 1, min(len(text), radius * 2)
        else:
            start = max(1, line - radius)
            end = min(len(text), line + radius)
        selected = "\n".join(text[start - 1 : end])
        return CodeSnippet(path=str(file_path), start_line=start, end_line=end, snippet=selected)
    except Exception:
        return None


def _gather_repo_meta(project_root: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    try:
        # Best-effort git info
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=project_root, text=True, capture_output=True)
        if branch.returncode == 0:
            meta["branch"] = branch.stdout.strip()
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, text=True, capture_output=True)
        if commit.returncode == 0:
            meta["commit"] = commit.stdout.strip()
    except Exception:
        pass
    return meta


def _gather_env_summary() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def _summarize_config(config: Config) -> Dict[str, Any]:
    return {
        "style": config.get("test_generation.style", {}),
        "coverage": {
            "minimum_line_coverage": config.get("test_generation.coverage.minimum_line_coverage", 80),
            "regenerate_if_below": config.get("test_generation.coverage.regenerate_if_below", 60),
        },
        "runner": config.get("test_generation.coverage.runner", {}),
        "env": config.get("test_generation.coverage.env", {}),
        "test_runner": config.get("test_generation.generation.test_runner", {}),
        "prompt_engineering": config.get("prompt_engineering", {}),
    }


def build_payload(
    *,
    failures: ParsedFailures,
    project_root: Path,
    config: Config,
    tests_written: List[str],
    last_run_command: List[str],
    top_n: int = 5,
    snippet_radius: int = 20,
) -> Dict[str, Any]:
    run_id = str(int(time.time() * 1000))

    # Limit to top N failures
    selected: List[FailureRecord] = failures.failures[:top_n]

    # Build failure entries with snippets
    normalized_failures: List[Dict[str, Any]] = []
    for f in selected:
        test_snippet: Optional[CodeSnippet] = None
        # Try to resolve file relative to project root
        file_path = (project_root / f.file).resolve() if not os.path.isabs(f.file) else Path(f.file)
        if file_path.exists():
            test_snippet = _read_snippet(file_path, f.line, radius=snippet_radius)
        normalized_failures.append(
            {
                "nodeid": f.nodeid,
                "file": f.file,
                "line": f.line,
                "message": f.message,
                "assertion_diff": f.assertion_diff,
                "code_context": asdict(test_snippet) if test_snippet else None,
                "captured_stdout": f.captured_stdout,
                "captured_stderr": f.captured_stderr,
                "duration": f.duration,
            }
        )

    payload: Dict[str, Any] = {
        "run_id": run_id,
        "repo_meta": _gather_repo_meta(project_root),
        "environment": _gather_env_summary(),
        "config_summary": _summarize_config(config),
        "tests_written": tests_written,
        "last_run_command": last_run_command,
        "failures": normalized_failures,
        "failures_total": failures.total,
    }

    return payload


def write_payload_json(payload: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "payload.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def build_refine_prompt(payload: Dict[str, Any], config: Config) -> str:
    """Construct a strong, concise prompt for refinement following 2025 guidelines.

    Emphasizes: step-by-step reasoning, examples in ✓/✗, decisive instructions, minimal fluff.
    """
    style = payload.get("config_summary", {}).get("style", {})
    framework = style.get("framework", "pytest")
    failures = payload.get("failures", [])

    guidance_flags = config.get("prompt_engineering", {})
    encourage_steps = guidance_flags.get("encourage_step_by_step", True)
    use_examples = guidance_flags.get("use_positive_negative_examples", True)
    decisive = guidance_flags.get("decisive_recommendations", True)

    parts: List[str] = []
    parts.append("You are a senior Python testing engineer. Refine failing tests to make the suite pass without weakening test quality.")
    if decisive:
        parts.append("Provide a single, decisive plan and concrete updated test files.")
    if encourage_steps:
        parts.append("Think step-by-step: analyze failures, propose minimal fixes, update tests, and re-run mentally.")
    if use_examples:
        parts.append("Follow ✓/✗ patterns: ✓ correct, minimal changes; ✗ suppressing assertions or masking failures.")

    parts.append(f"Testing framework: {framework}")
    parts.append("Constraints:\n- Do not modify production source files.\n- Keep tests readable and consistent with existing style.\n- Preserve coverage where possible; prefer precise assertions.")

    # Include summarized failures
    parts.append("Failures (top):")
    for idx, f in enumerate(failures, 1):
        nodeid = f.get("nodeid")
        message = f.get("message")
        assertion = f.get("assertion_diff")
        snippet = f.get("code_context", {})
        loc = f"{snippet.get('path')}:{snippet.get('start_line')}-{snippet.get('end_line')}" if snippet else ""
        parts.append(f"{idx}. {nodeid} — {message}\n   {('Assertion: ' + assertion) if assertion else ''}\n   Context: {loc}")

    parts.append("Return JSON with updated tests only: { \"path\": \"<test path>\", \"content\": \"<file content>\" }[].")
    parts.append("Do not include prose outside JSON.")

    return "\n\n".join(p for p in parts if p)


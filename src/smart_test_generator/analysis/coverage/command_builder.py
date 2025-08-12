"""Command builder for invoking pytest with coverage with robust env handling."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


@dataclass
class CommandSpec:
    argv: List[str]
    cwd: Path
    env: Dict[str, str]


def build_pytest_command(
    *,
    project_root: Path,
    config,
    preflight_result: Optional[Dict] = None,
) -> CommandSpec:
    """Construct the pytest command, working directory, and environment from config.

    - Supports runner modes: 'python-module' (default), 'pytest-path', 'custom'
    - Merges runner.args with test_generation.coverage.pytest_args (runner.args first)
    - Sets cwd from runner.cwd or project_root
    - Propagates env by default; merges env.extra; appends env.append_pythonpath to PYTHONPATH
    """
    # Defaults and config lookups
    runner_mode = (config.get('test_generation.coverage.runner.mode', 'python-module') or 'python-module').strip()
    runner_python = config.get('test_generation.coverage.runner.python') or sys.executable
    runner_pytest_path = config.get('test_generation.coverage.runner.pytest_path', 'pytest')
    runner_custom_cmd = config.get('test_generation.coverage.runner.custom_cmd', []) or []
    runner_args = list(config.get('test_generation.coverage.runner.args', []) or [])
    extra_pytest_args = list(config.get('test_generation.coverage.pytest_args', []) or [])

    # Core coverage args
    cov_args = [f"--cov={project_root}", "--cov-report=term-missing"]

    # Build argv
    if runner_mode == 'pytest-path':
        argv = [runner_pytest_path, *cov_args, *runner_args, *extra_pytest_args]
    elif runner_mode == 'custom' and runner_custom_cmd:
        argv = [*runner_custom_cmd, *cov_args, *runner_args, *extra_pytest_args]
    else:
        argv = [runner_python, '-m', 'pytest', *cov_args, *runner_args, *extra_pytest_args]

    # Working directory
    cwd_cfg = config.get('test_generation.coverage.runner.cwd')
    cwd_path = Path(cwd_cfg) if cwd_cfg else project_root

    # Environment
    propagate = bool(config.get('test_generation.coverage.env.propagate', True))
    base_env: Dict[str, str] = dict(os.environ) if propagate else {}
    extra_env: Dict[str, str] = config.get('test_generation.coverage.env.extra', {}) or {}
    for k, v in extra_env.items():
        base_env[str(k)] = str(v)

    append_paths = list(config.get('test_generation.coverage.env.append_pythonpath', []) or [])
    if append_paths:
        sep = os.pathsep
        existing = base_env.get('PYTHONPATH', '')
        parts: List[str] = [p for p in (existing.split(sep) if existing else []) if p]
        parts.extend(str(Path(p)) for p in append_paths)
        base_env['PYTHONPATH'] = sep.join(parts)

    return CommandSpec(argv=argv, cwd=cwd_path, env=base_env)


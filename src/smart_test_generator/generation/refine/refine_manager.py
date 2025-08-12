"""Bounded retry loop for LLM-driven test refinement."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

from smart_test_generator.config import Config
from smart_test_generator.generation.llm_clients import LLMClient
from smart_test_generator.generation.refine.payload_builder import build_refine_prompt


@dataclass
class RefineResponse:
    updated_files: List[Dict[str, str]]  # [{path, content}]
    rationale: str
    plan: str


@dataclass
class RefinementOutcome:
    iterations: int
    final_exit_code: int
    updated_any: bool


def _jitter_delay(base: float, cap: float, attempt: int) -> float:
    return min(cap, base * (2 ** (attempt - 1))) * (1 + random.uniform(-0.2, 0.2))


def run_refinement_cycle(
    *,
    payload: Dict[str, Any],
    project_root: Path,
    artifacts_dir: Path,
    llm_client: LLMClient,
    config: Config,
    apply_updates_fn,
    re_run_pytest_fn,
) -> RefinementOutcome:
    refine_cfg = config.get("test_generation.generation.refine", {}) or {}
    if not refine_cfg.get("enable", False):
        return RefinementOutcome(iterations=0, final_exit_code=1, updated_any=False)

    max_retries = int(refine_cfg.get("max_retries", 2))
    base = float(refine_cfg.get("backoff_base_sec", 1.0))
    cap = float(refine_cfg.get("backoff_max_sec", 8.0))
    stop_on_no_change = bool(refine_cfg.get("stop_on_no_change", True))

    prompt = build_refine_prompt(payload, config)
    updated_any = False
    last_exit = 1

    for attempt in range(1, max_retries + 1):
        iter_dir = artifacts_dir / f"iter_{attempt}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        # Call LLM
        response_text = llm_client.refine_tests({"payload": payload, "prompt": prompt})
        (iter_dir / "llm_response.json").write_text(response_text)

        try:
            data = json.loads(response_text)
            updated_files = data.get("updated_files", [])
        except Exception:
            updated_files = []

        if not updated_files and stop_on_no_change:
            break

        if updated_files:
            updated_any = True
            apply_updates_fn(updated_files, project_root)

        # Re-run pytest
        rr = re_run_pytest_fn()
        last_exit = int(rr)
        (iter_dir / "post_run_exit.txt").write_text(str(last_exit))
        if last_exit == 0:
            break

        time.sleep(_jitter_delay(base, cap, attempt))

    return RefinementOutcome(iterations=attempt, final_exit_code=last_exit, updated_any=updated_any)


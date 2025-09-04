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
import logging
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
from smart_test_generator.utils.prompt_loader import get_prompt_loader
from smart_test_generator.services.git_service import GitService, GitContext
from smart_test_generator.analysis.failure_pattern_analyzer import FailurePatternAnalyzer, FailureAnalysis

logger = logging.getLogger(__name__)


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


def _gather_repo_meta(project_root: Path, include_recent_changes: bool = True) -> Dict[str, Any]:
    """Gather enhanced repository metadata including git diff context."""
    meta: Dict[str, Any] = {}
    
    try:
        git_service = GitService(project_root)
        git_context = git_service.get_git_context(
            include_recent_changes=include_recent_changes,
            days_back=7,
            max_commits=5
        )
        
        meta["branch"] = git_context.current_branch
        meta["commit"] = git_context.current_commit
        meta["has_uncommitted_changes"] = git_context.is_dirty
        
        # Add recent changes context if available
        if git_context.recent_changes:
            changes = git_context.recent_changes
            meta["recent_changes"] = {
                "time_range": changes.time_range,
                "total_files_changed": changes.total_changes,
                "test_files_changed": len(changes.test_files_changed),
                "source_files_changed": len(changes.source_files_changed),
                "recent_commit_messages": changes.commit_messages[:3],  # Top 3 most recent
                "changed_test_files": [f.file_path for f in changes.test_files_changed[:5]],
                "changed_source_files": [f.file_path for f in changes.source_files_changed[:5]]
            }
            
            # Add specific details for test file changes
            if changes.test_files_changed:
                meta["test_changes_detail"] = []
                for change in changes.test_files_changed[:3]:  # Top 3 test file changes
                    meta["test_changes_detail"].append({
                        "file": change.file_path,
                        "status": change.status,
                        "lines_added": change.lines_added,
                        "lines_removed": change.lines_removed
                    })
        
    except Exception as e:
        logger.warning(f"GitService failed, attempting fallback git commands: {e}")
        # Fallback to basic git info if GitService fails
        try:
            branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], 
                                  cwd=project_root, text=True, capture_output=True)
            if branch.returncode == 0:
                meta["branch"] = branch.stdout.strip()
            elif branch.returncode == 128:
                logger.warning("Git repository not initialized or accessible")
                meta["branch"] = "no-git"
            else:
                meta["branch"] = "unknown"
                
            commit = subprocess.run(["git", "rev-parse", "HEAD"], 
                                  cwd=project_root, text=True, capture_output=True)
            if commit.returncode == 0:
                meta["commit"] = commit.stdout.strip()
            elif commit.returncode == 128:
                meta["commit"] = "no-git"
            else:
                meta["commit"] = "unknown"
        except Exception as fallback_e:
            logger.warning(f"Fallback git commands also failed: {fallback_e}")
            meta["branch"] = "no-git"
            meta["commit"] = "no-git"
    
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
    include_git_context: bool = True,
    include_pattern_analysis: bool = True,
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

    # Enhanced payload with git context
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "repo_meta": _gather_repo_meta(project_root, include_recent_changes=include_git_context),
        "environment": _gather_env_summary(),
        "config_summary": _summarize_config(config),
        "tests_written": tests_written,
        "last_run_command": last_run_command,
        "failures": normalized_failures,
        "failures_total": failures.total,
    }
    
    # Add git-based failure context if available
    if include_git_context:
        try:
            git_service = GitService(project_root)
            
            # Get failed test file paths
            failed_test_files = list(set(f.file for f in selected if f.file))
            
            # Analyze which changes might be affecting failing tests
            failure_context = git_service.get_commit_context_for_failures(failed_test_files)
            payload["failure_git_context"] = failure_context
            
            # Check if any failing tests were recently modified
            recent_test_changes = git_service.get_recent_test_changes(
                include_diff_content=False  # Skip diff content for now to save tokens
            )
            
            relevant_test_changes = [
                change for change in recent_test_changes
                if any(failed_file.endswith(change.file_path) or change.file_path.endswith(failed_file)
                      for failed_file in failed_test_files)
            ]
            
            if relevant_test_changes:
                payload["relevant_test_changes"] = [
                    {
                        "file": change.file_path,
                        "status": change.status,
                        "lines_added": change.lines_added,
                        "lines_removed": change.lines_removed
                    }
                    for change in relevant_test_changes[:3]  # Limit to top 3
                ]
                
        except Exception as e:
            # Log error but don't fail the whole payload generation
            logger.warning(f"Failed to add git context to refinement payload: {e}")
    
    # Add failure pattern analysis if enabled
    if include_pattern_analysis:
        try:
            pattern_analyzer = FailurePatternAnalyzer(project_root)
            pattern_analysis = pattern_analyzer.analyze_failures(failures)
            
            # Add pattern analysis to payload
            payload["failure_analysis"] = {
                "total_failures": pattern_analysis.total_failures,
                "pattern_frequencies": {
                    category.value: count 
                    for category, count in pattern_analysis.pattern_frequencies.items()
                },
                "trending_patterns": [
                    {"category": category.value, "trend_score": score}
                    for category, score in pattern_analysis.trending_patterns
                ],
                "fix_suggestions": [
                    {
                        "category": suggestion.category.value,
                        "title": suggestion.title,
                        "description": suggestion.description,
                        "priority": suggestion.priority,
                        "automated": suggestion.automated,
                        "code_example": suggestion.code_example
                    }
                    for suggestion in pattern_analysis.fix_suggestions[:5]  # Top 5 suggestions
                ],
                "confidence_scores": pattern_analysis.confidence_scores
            }
            
            # Add historical context if available
            success_rates = pattern_analyzer.get_success_rates()
            if success_rates:
                payload["failure_analysis"]["historical_success_rates"] = {
                    category.value: rate for category, rate in success_rates.items()
                }
                
        except Exception as e:
            # Log error but don't fail the whole payload generation
            logger.warning(f"Failed to add pattern analysis to refinement payload: {e}")

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

    prompt_loader = get_prompt_loader()
    return prompt_loader.get_advanced_refinement_prompt(
        framework=framework,
        encourage_steps=encourage_steps,
        use_examples=use_examples,
        decisive=decisive,
        failures=failures
    )


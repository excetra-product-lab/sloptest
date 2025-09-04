"""Bounded retry loop for LLM-driven test refinement."""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple

from smart_test_generator.config import Config
from smart_test_generator.generation.llm_clients import LLMClient
from smart_test_generator.generation.refine.payload_builder import build_refine_prompt
from smart_test_generator.analysis.failure_pattern_analyzer import FailurePatternAnalyzer, FailureCategory

logger = logging.getLogger(__name__)


@dataclass
class RefineResponse:
    updated_files: List[Dict[str, str]]  # [{path, content}]
    rationale: str
    plan: str
    confidence: str = "medium"  # high, medium, low
    risk_assessment: str = ""


@dataclass
class RefinementOutcome:
    iterations: int
    final_exit_code: int
    updated_any: bool
    pattern_insights: Dict[str, Any] = None  # Pattern analysis insights
    retry_strategy_used: str = "default"
    confidence_improvement: float = 0.0
    final_confidence: str = "medium"  # Final confidence level from last attempt
    risk_assessments: List[str] = None  # Risk assessments from each attempt
    reasoning_quality: str = "unknown"  # Quality assessment of the reasoning provided


def _jitter_delay(base: float, cap: float, attempt: int, failure_category: str = None) -> float:
    """Calculate delay with jitter, adjusted for failure category."""
    base_delay = min(cap, base * (2 ** (attempt - 1)))
    
    # Adjust delay based on failure category
    if failure_category:
        if failure_category in ['import_error', 'dependency_error']:
            # Import errors often need more time to resolve
            base_delay *= 1.5
        elif failure_category in ['assertion_error', 'mock_error']:
            # Logic errors might resolve faster
            base_delay *= 0.8
    
    return base_delay * (1 + random.uniform(-0.2, 0.2))


def _determine_retry_strategy(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Determine optimal retry strategy based on failure patterns."""
    failure_analysis = payload.get("failure_analysis", {})
    pattern_frequencies = failure_analysis.get("pattern_frequencies", {})
    
    if not pattern_frequencies:
        return "default", {}
    
    # Find dominant failure category
    dominant_category = max(pattern_frequencies, key=pattern_frequencies.get)
    dominant_count = pattern_frequencies[dominant_category]
    total_failures = failure_analysis.get("total_failures", 1)
    dominance_ratio = dominant_count / total_failures
    
    strategy_config = {"dominant_category": dominant_category, "dominance_ratio": dominance_ratio}
    
    # Choose strategy based on failure patterns
    if dominance_ratio > 0.7:  # Single category dominates
        if dominant_category in ['import_error', 'dependency_error']:
            return "dependency_focused", strategy_config
        elif dominant_category in ['assertion_error']:
            return "logic_focused", strategy_config
        elif dominant_category in ['fixture_error', 'mock_error']:
            return "setup_focused", strategy_config
    elif len(pattern_frequencies) > 3:  # Many different failure types
        return "comprehensive", strategy_config
    
    return "balanced", strategy_config


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
        return RefinementOutcome(
            iterations=0, 
            final_exit_code=1, 
            updated_any=False,
            retry_strategy_used="disabled"
        )

    max_retries = int(refine_cfg.get("max_retries", 2))
    base = float(refine_cfg.get("backoff_base_sec", 1.0))
    cap = float(refine_cfg.get("backoff_max_sec", 8.0))
    stop_on_no_change = bool(refine_cfg.get("stop_on_no_change", True))

    # Determine optimal retry strategy based on failure patterns
    retry_strategy, strategy_config = _determine_retry_strategy(payload)
    logger.info(f"Using refinement strategy: {retry_strategy} with config: {strategy_config}")

    # Adjust max_retries based on strategy
    if retry_strategy == "dependency_focused":
        max_retries = max(max_retries, 3)  # Dependencies might need more attempts
    elif retry_strategy == "comprehensive":
        max_retries = min(max_retries + 1, 5)  # Complex cases get one more attempt

    prompt = build_refine_prompt(payload, config)
    updated_any = False
    last_exit = 1
    pattern_analyzer = FailurePatternAnalyzer(project_root)
    initial_confidence = _calculate_average_confidence(payload)
    
    # Track refinement quality metrics
    confidence_levels = []
    risk_assessments = []
    reasoning_quality = "unknown"

    for attempt in range(1, max_retries + 1):
        iter_dir = artifacts_dir / f"iter_{attempt}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        # Call LLM
        try:
            response_text = llm_client.refine_tests({"payload": payload, "prompt": prompt})
            if not isinstance(response_text, str):
                logger.error(f"LLM client returned non-string response type: {type(response_text)}")
                logger.error(f"Response content: {repr(response_text)}")
                response_text = str(response_text) if response_text is not None else ""
            (iter_dir / "llm_response.json").write_text(response_text)
        except Exception as e:
            logger.error(f"Error calling LLM refine_tests method: {e}")
            logger.error(f"Request data types - payload: {type(payload)}, prompt: {type(prompt)}")
            response_text = ""

        # Parse response and extract enhanced information
        refine_response = None
        updated_files = []
        
        try:
            # Check if we have content to parse
            if not response_text or not response_text.strip():
                logger.warning(f"Empty response from LLM for refinement iteration {attempt}")
            else:
                data = json.loads(response_text)
                updated_files = data.get("updated_files", [])
                
                # Create enhanced response object
                refine_response = RefineResponse(
                    updated_files=updated_files,
                    rationale=data.get("rationale", ""),
                    plan=data.get("plan", ""),
                    confidence=data.get("confidence", "medium"),
                    risk_assessment=data.get("risk_assessment", "")
                )
                
                # Log enhanced information for monitoring
                logger.info(f"Refinement attempt {attempt}: "
                          f"confidence={refine_response.confidence}, "
                          f"files_updated={len(updated_files)}, "
                          f"risk={'present' if refine_response.risk_assessment else 'none'}")
                
                # Collect quality metrics
                confidence_levels.append(refine_response.confidence)
                if refine_response.risk_assessment:
                    risk_assessments.append(refine_response.risk_assessment)
                
                # Assess reasoning quality based on presence of detailed rationale and plan
                if len(refine_response.rationale) > 50 and len(refine_response.plan) > 30:
                    reasoning_quality = "good"
                elif refine_response.rationale and refine_response.plan:
                    reasoning_quality = "basic"
                else:
                    reasoning_quality = "poor"
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in refinement iteration {attempt}: {e}")
            logger.error(f"Failed content preview: {repr(response_text[:200])}")
            if "Expecting value: line 1 column 1 (char 0)" in str(e):
                logger.error("The AI returned empty or non-JSON content for refinement")
                logger.debug(f"Full response: {repr(response_text)}")
        except Exception as e:
            logger.error(f"Unexpected error parsing refinement response: {e}")

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
            # Mark successful resolution for pattern learning
            if "failure_analysis" in payload:
                for category_str in payload["failure_analysis"].get("pattern_frequencies", {}):
                    try:
                        category = FailureCategory(category_str)
                        pattern_analyzer.mark_resolution_success("refinement_cycle", category, True)
                    except ValueError:
                        continue
            break

        # Apply smart delay based on dominant failure category
        dominant_category = strategy_config.get("dominant_category", "unknown")
        time.sleep(_jitter_delay(base, cap, attempt, dominant_category))

    # Calculate confidence improvement and final state
    final_confidence_level = confidence_levels[-1] if confidence_levels else "medium"
    
    # Convert confidence levels to numeric for improvement calculation
    confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
    numeric_final = confidence_map.get(final_confidence_level, 0.6)
    confidence_improvement = numeric_final - initial_confidence

    # Gather enhanced pattern insights for the outcome
    pattern_insights = {
        "strategy_used": retry_strategy,
        "strategy_config": strategy_config,
        "initial_confidence": initial_confidence,
        "final_numeric_confidence": numeric_final,
        "failure_categories": list(payload.get("failure_analysis", {}).get("pattern_frequencies", {}).keys()),
        "confidence_progression": confidence_levels,
        "total_attempts": attempt
    }

    return RefinementOutcome(
        iterations=attempt, 
        final_exit_code=last_exit, 
        updated_any=updated_any,
        pattern_insights=pattern_insights,
        retry_strategy_used=retry_strategy,
        confidence_improvement=confidence_improvement,
        final_confidence=final_confidence_level,
        risk_assessments=risk_assessments,
        reasoning_quality=reasoning_quality
    )


def _calculate_average_confidence(payload: Dict[str, Any]) -> float:
    """Calculate average confidence score from failure analysis."""
    failure_analysis = payload.get("failure_analysis", {})
    confidence_scores = failure_analysis.get("confidence_scores", {})
    
    if not confidence_scores:
        return 0.5  # Default neutral confidence
    
    return sum(confidence_scores.values()) / len(confidence_scores)


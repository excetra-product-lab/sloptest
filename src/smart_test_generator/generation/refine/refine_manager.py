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
from rich.panel import Panel

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


def _display_refinement_strategy(feedback, strategy: str, strategy_config: Dict[str, Any], max_retries: int):
    """Display refinement strategy information to user."""
    feedback.section_header("🔧 Refinement Strategy")
    
    # Strategy description
    strategy_descriptions = {
        "comprehensive": "Thorough approach for complex issues with multiple failure types",
        "balanced": "General-purpose approach for mixed failure patterns", 
        "dependency_focused": "Specialized approach for import/dependency errors",
        "logic_focused": "Specialized approach for assertion/logic errors",
        "setup_focused": "Specialized approach for fixture/mock errors"
    }
    
    strategy_info = {
        "Strategy": strategy.replace("_", " ").title(),
        "Description": strategy_descriptions.get(strategy, "Custom strategy"),
        "Max Attempts": str(max_retries),
        "Detection": "Manual Override" if strategy_config.get("overridden") else "Auto-Detected"
    }
    
    if "dominant_category" in strategy_config:
        strategy_info["Primary Issue"] = strategy_config["dominant_category"].replace("_", " ").title()
        strategy_info["Dominance"] = f"{strategy_config.get('dominance_ratio', 0) * 100:.1f}%"
    
    feedback.summary_panel("Strategy Configuration", strategy_info, "blue")


def _display_refinement_attempt(feedback, attempt: int, max_retries: int, refine_response: 'RefineResponse'):
    """Display detailed information about a refinement attempt."""
    feedback.subsection_header(f"🔄 Refinement Attempt {attempt}/{max_retries}")
    
    # Confidence level with color coding
    confidence_colors = {"high": "green", "medium": "yellow", "low": "red"}
    confidence_color = confidence_colors.get(refine_response.confidence.lower(), "blue")
    
    attempt_info = {
        "Confidence Level": f"[{confidence_color}]{refine_response.confidence.title()}[/{confidence_color}]",
        "Files to Update": str(len(refine_response.updated_files)),
    }
    
    # Display attempt summary
    feedback.summary_panel("Attempt Summary", attempt_info, confidence_color)
    
    # Show files being updated if any
    if refine_response.updated_files:
        files_content = []
        for file_info in refine_response.updated_files:
            file_path = file_info.get('path', 'Unknown file')
            content_length = len(file_info.get('content', ''))
            files_content.append(f"📝 {file_path} ({content_length:,} chars)")
        
        files_panel = Panel(
            "\n".join(files_content),
            title="[bold green]📁 Files Being Updated[/bold green]",
            border_style="green",
            padding=(1, 2)
        )
        feedback.console.print(files_panel)
    
    # Display rationale in a structured panel
    rationale_text = refine_response.rationale.strip() if refine_response.rationale else ""
    
    if rationale_text and len(rationale_text) > 0:
        # Clean up fallback messages for better user experience
        if not rationale_text.startswith("[Fallback]") and not rationale_text.startswith("AI provided minimal"):
            # Use a proper panel for the rationale
            rationale_panel = Panel(
                rationale_text,
                title="[bold blue]💭 AI Rationale[/bold blue]",
                border_style="blue",
                padding=(1, 2)
            )
            feedback.console.print(rationale_panel)
        else:
            # For fallback content, show a cleaner version
            feedback.warning("💭 AI Rationale", "AI response was incomplete or unclear for this attempt")
    
    # Display plan in a structured panel
    plan_text = refine_response.plan.strip() if refine_response.plan else ""
    
    if plan_text and len(plan_text) > 0:
        # Clean up fallback messages for better user experience
        if not plan_text.startswith("[Fallback]") and not plan_text.startswith("AI provided minimal"):
            # Use a proper panel for the plan
            plan_panel = Panel(
                plan_text,
                title="[bold cyan]📋 Refinement Plan[/bold cyan]",
                border_style="cyan",
                padding=(1, 2)
            )
            feedback.console.print(plan_panel)
        else:
            # For fallback content, show a cleaner version
            feedback.warning("📋 Refinement Plan", "AI was unable to provide a clear refinement plan for this attempt")
    
    # Display risk assessment in a structured panel
    risk_text = refine_response.risk_assessment.strip() if refine_response.risk_assessment else ""
    
    if risk_text and len(risk_text) > 0:
        # Clean up fallback messages for better user experience  
        if not risk_text.startswith("[Fallback]") and not risk_text.startswith("AI provided minimal"):
            # Color code based on risk level
            risk_color = "red" if "high" in risk_text.lower() else "yellow"
            risk_title = "[bold red]⚠️ High Risk Assessment[/bold red]" if "high" in risk_text.lower() else "[bold yellow]⚠️ Risk Assessment[/bold yellow]"
            
            # Use a proper panel for the risk assessment
            risk_panel = Panel(
                risk_text,
                title=risk_title,
                border_style=risk_color,
                padding=(1, 2)
            )
            feedback.console.print(risk_panel)
        else:
            # For fallback content, show a cleaner version
            feedback.warning("⚠️ Risk Assessment", "Unable to assess risk due to incomplete AI response")
    
    # Add some spacing after each attempt
    feedback.console.print()


def _display_refinement_outcome(feedback, outcome: 'RefinementOutcome'):
    """Display final refinement outcome summary."""
    feedback.section_header("📊 Refinement Summary")
    
    # Determine overall result
    if outcome.final_exit_code == 0:
        result_icon = "✅"
        result_text = "Success"
        result_color = "green"
    else:
        result_icon = "❌"
        result_text = "Failed"  
        result_color = "red"
    
    outcome_info = {
        "Result": f"[{result_color}]{result_icon} {result_text}[/{result_color}]",
        "Iterations": str(outcome.iterations),
        "Strategy Used": outcome.retry_strategy_used.replace("_", " ").title(),
        "Files Updated": "Yes" if outcome.updated_any else "No",
        "Final Confidence": outcome.final_confidence.title() if hasattr(outcome, 'final_confidence') else "Unknown"
    }
    
    if hasattr(outcome, 'confidence_improvement') and outcome.confidence_improvement > 0:
        outcome_info["Confidence Improvement"] = f"+{outcome.confidence_improvement:.1f}%"
    
    feedback.final_summary("Refinement Complete", outcome_info, result_color)
    
    # Provide actionable next steps based on outcome
    if outcome.final_exit_code == 0:
        feedback.success("✅ Next Steps", "Your test files have been successfully refined and should now pass. Run them locally to verify the fixes work in your environment.")
    else:
        next_steps = "❌ Refinement was unable to fix all test failures. Consider:\n"
        next_steps += "• Running the tests locally to see current failure patterns\n"
        next_steps += "• Checking if your production code has changed since tests were generated\n"
        next_steps += "• Manually reviewing test assumptions and mock configurations\n"
        next_steps += "• Using a different refinement strategy in your configuration"
        feedback.warning("🔄 Recommended Next Steps", next_steps)
    
    # Display risk assessments summary with clear context
    if hasattr(outcome, 'risk_assessments') and outcome.risk_assessments:
        risks = [risk for risk in outcome.risk_assessments if risk.strip()]
        if risks:
            risk_header = "🔍 Test File Change Risks & Considerations" 
            risk_intro = "These assessments relate to the AI's changes to your test files (not your production code):"
            
            formatted_risks = []
            for i, risk in enumerate(risks):
                formatted_risks.append(f"📝 Refinement Attempt {i+1}:\n   {risk}")
            
            feedback.info(risk_header, f"{risk_intro}\n\n" + "\n\n".join(formatted_risks))


def run_refinement_cycle(
    *,
    payload: Dict[str, Any],
    project_root: Path,
    artifacts_dir: Path,
    llm_client: LLMClient,
    config: Config,
    apply_updates_fn,
    re_run_pytest_fn,
    feedback=None,
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

    # Check for strategy override in configuration
    configured_strategy = refine_cfg.get("strategy", "auto")
    if configured_strategy != "auto":
        # Use the configured strategy instead of auto-detection
        valid_strategies = ["comprehensive", "balanced", "dependency_focused", "logic_focused", "setup_focused"]
        if configured_strategy in valid_strategies:
            retry_strategy = configured_strategy
            strategy_config = {"overridden": True, "configured_strategy": configured_strategy}
            logger.debug(f"Using configured refinement strategy: {retry_strategy} (overriding auto-detection)")
        else:
            logger.warning(f"Invalid configured strategy '{configured_strategy}', falling back to auto-detection")
            retry_strategy, strategy_config = _determine_retry_strategy(payload)
            logger.debug(f"Using auto-detected refinement strategy: {retry_strategy} with config: {strategy_config}")
    else:
        # Use automatic strategy determination based on failure patterns
        retry_strategy, strategy_config = _determine_retry_strategy(payload)
        logger.debug(f"Using auto-detected refinement strategy: {retry_strategy} with config: {strategy_config}")
    
    # Display strategy information to user
    if feedback:
        _display_refinement_strategy(feedback, retry_strategy, strategy_config, max_retries)

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
    reasoning_quality = "not_assessed"  # Will be updated based on AI responses

    for attempt in range(1, max_retries + 1):
        iter_dir = artifacts_dir / f"iter_{attempt}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        # Call LLM with retry for server errors
        response_text = ""
        api_error_occurred = False
        try:
            response_text = llm_client.refine_tests({"payload": payload, "prompt": prompt})
            if not isinstance(response_text, str):
                logger.error(f"LLM client returned non-string response type: {type(response_text)}")
                logger.error(f"Response content: {repr(response_text)}")
                response_text = str(response_text) if response_text is not None else ""
            (iter_dir / "llm_response.json").write_text(response_text)
        except Exception as e:
            api_error_occurred = True
            error_msg = str(e)
            logger.error(f"Error calling LLM refine_tests method: {e}")
            logger.error(f"Request data types - payload: {type(payload)}, prompt: {type(prompt)}")
            
            # Check for specific error types  
            if "529" in error_msg or "Server Error" in error_msg:
                logger.warning(f"Server overload error (529) on attempt {attempt}. This is often temporary.")
                if feedback:
                    feedback.warning("API Server Error", f"Claude API returned server error (529) on attempt {attempt}. This is typically temporary server overload.")
            elif "400" in error_msg:
                logger.error(f"Bad request error (400) on attempt {attempt}. Check request format.")
                if feedback:
                    feedback.error("API Request Error", "Bad request sent to Claude API. This may indicate a parameter or format issue.")
            
            response_text = ""

        # Parse response and extract enhanced information
        refine_response = None
        updated_files = []
        
        # Handle API errors with informative response
        if api_error_occurred:
            refine_response = RefineResponse(
                updated_files=[],
                rationale=f"API Error occurred during refinement attempt {attempt}: {error_msg}",
                plan="Unable to generate refinement plan due to API error. Will retry if attempts remain.",
                confidence="low",
                risk_assessment="High risk - API communication failed. Consider checking network connectivity or API status."
            )
            
            if feedback:
                _display_refinement_attempt(feedback, attempt, max_retries, refine_response)
                
            # Skip to delay/retry logic
            if attempt < max_retries:
                delay_time = _jitter_delay(base, cap, attempt)
                logger.info(f"Waiting {delay_time:.1f}s before retry due to API error...")
                time.sleep(delay_time)
            continue
        
        try:
            # Check if we have content to parse
            if not response_text or not response_text.strip():
                logger.warning(f"Empty response from LLM for refinement iteration {attempt}")
                
                # Create response for empty content case
                refine_response = RefineResponse(
                    updated_files=[],
                    rationale="Received empty response from AI model - no content returned.",
                    plan="Unable to generate refinement plan due to empty API response.",
                    confidence="low",
                    risk_assessment="High risk - Empty API response suggests communication or processing issue."
                )
                
                if feedback:
                    _display_refinement_attempt(feedback, attempt, max_retries, refine_response)
            else:
                # Log raw response for debugging
                logger.debug(f"Raw API response for attempt {attempt}: {response_text[:500]}...")
                
                data = json.loads(response_text)
                updated_files = data.get("updated_files", [])
                
                # Extract response fields with minimal logging
                raw_rationale = data.get('rationale', '')
                raw_plan = data.get('plan', '')
                raw_risk = data.get('risk_assessment', '')
                
                # Only log detailed response info in debug mode
                logger.debug(f"Refinement response fields for attempt {attempt}: files={len(updated_files)}, confidence={data.get('confidence', 'missing')}")
                logger.debug(f"Response lengths - rationale: {len(raw_rationale)}, plan: {len(raw_plan)}, risk: {len(raw_risk)}")
                
                # Create enhanced response object with validation
                rationale = data.get("rationale", "").strip()
                plan = data.get("plan", "").strip()
                risk_assessment = data.get("risk_assessment", "").strip()
                confidence = data.get("confidence", "medium").strip().lower()
                
                # Provide defaults if fields are empty or minimal
                if not rationale or len(rationale) < 20:
                    logger.debug(f"Minimal rationale received (length: {len(rationale)})")
                    rationale = f"AI provided minimal rationale (received: '{rationale or 'empty'}') for refinement attempt {attempt}. This may indicate an API issue or incomplete response."
                
                if not plan or len(plan) < 20:
                    logger.debug(f"Minimal plan received (length: {len(plan)})")
                    plan = f"AI provided minimal plan (received: '{plan or 'empty'}') for refinement attempt {attempt}. This may indicate an API issue or incomplete response."
                
                if not risk_assessment or len(risk_assessment) < 15:
                    logger.debug(f"Minimal risk assessment received (length: {len(risk_assessment)})")
                    risk_assessment = f"AI provided minimal risk assessment (received: '{risk_assessment or 'empty'}') - manual assessment recommended due to incomplete response."
                
                # Validate confidence level
                if confidence not in ["high", "medium", "low"]:
                    logger.debug(f"Invalid confidence level '{confidence}', defaulting to medium")
                    confidence = "medium"
                
                # Final safeguard - ensure we always have meaningful content
                if not rationale or len(rationale.strip()) < 10:
                    rationale = f"[Fallback] AI response was incomplete for attempt {attempt}. Received: '{rationale or 'empty'}'"
                    
                if not plan or len(plan.strip()) < 10:
                    plan = f"[Fallback] AI response was incomplete for attempt {attempt}. Received: '{plan or 'empty'}'"
                    
                if not risk_assessment or len(risk_assessment.strip()) < 10:
                    risk_assessment = f"[Fallback] AI response was incomplete for attempt {attempt}. Received: '{risk_assessment or 'empty'}'"
                
                refine_response = RefineResponse(
                    updated_files=updated_files,
                    rationale=rationale,
                    plan=plan,
                    confidence=confidence,
                    risk_assessment=risk_assessment
                )
                
                # Display detailed refinement information to user
                if feedback:
                    _display_refinement_attempt(feedback, attempt, max_retries, refine_response)
                
                # Log summary information for monitoring (debug level only)
                logger.debug(f"Refinement attempt {attempt}: "
                          f"confidence={refine_response.confidence}, "
                          f"files_updated={len(updated_files)}, "
                          f"risk={'present' if refine_response.risk_assessment else 'none'}")
                
                # Collect quality metrics
                confidence_levels.append(refine_response.confidence)
                if refine_response.risk_assessment:
                    risk_assessments.append(refine_response.risk_assessment)
                
                # Assess reasoning quality based on content richness and completeness
                rationale_len = len(refine_response.rationale)
                plan_len = len(refine_response.plan)
                risk_len = len(refine_response.risk_assessment)
                
                if rationale_len > 100 and plan_len > 50 and risk_len > 30:
                    reasoning_quality = "detailed_analysis_provided"
                elif rationale_len > 50 and plan_len > 30:
                    reasoning_quality = "adequate_reasoning_provided"
                elif rationale_len > 20 or plan_len > 20:
                    reasoning_quality = "minimal_reasoning_provided"
                elif api_error_occurred:
                    reasoning_quality = "api_error_prevented_assessment"
                else:
                    reasoning_quality = "insufficient_reasoning_from_ai"
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in refinement iteration {attempt}: {e}")
            logger.error(f"Failed content preview: {repr(response_text[:200])}")
            if "Expecting value: line 1 column 1 (char 0)" in str(e):
                logger.error("The AI returned empty or non-JSON content for refinement")
                logger.debug(f"Full response: {repr(response_text)}")
                
                # Create a fallback response with error information
                refine_response = RefineResponse(
                    updated_files=[],
                    rationale=f"JSON parsing failed: {str(e)}. Response was empty or malformed.",
                    plan="Unable to parse refinement plan due to API response error.",
                    confidence="low",
                    risk_assessment="High risk - API response parsing failed. Manual review recommended."
                )
                
                if feedback:
                    _display_refinement_attempt(feedback, attempt, max_retries, refine_response)
                
        except Exception as e:
            logger.error(f"Unexpected error parsing refinement response: {e}")
            
            # Create a fallback response for unexpected errors  
            refine_response = RefineResponse(
                updated_files=[],
                rationale=f"Unexpected error during refinement: {str(e)}",
                plan="Unable to generate refinement plan due to processing error.",
                confidence="low", 
                risk_assessment="High risk - Processing error occurred. Manual review required."
            )
            
            if feedback:
                _display_refinement_attempt(feedback, attempt, max_retries, refine_response)

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

    outcome = RefinementOutcome(
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
    
    # Display final refinement outcome to user
    if feedback:
        _display_refinement_outcome(feedback, outcome)
    
    return outcome


def _calculate_average_confidence(payload: Dict[str, Any]) -> float:
    """Calculate average confidence score from failure analysis."""
    failure_analysis = payload.get("failure_analysis", {})
    confidence_scores = failure_analysis.get("confidence_scores", {})
    
    if not confidence_scores:
        return 0.5  # Default neutral confidence
    
    return sum(confidence_scores.values()) / len(confidence_scores)


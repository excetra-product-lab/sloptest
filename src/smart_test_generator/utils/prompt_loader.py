"""Prompt loading utility for managing YAML-based prompts.

This module provides a centralized way to load and manage prompts from YAML files,
with support for template substitution, caching, and different prompt variations.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


class PromptLoader:
    """Loads and manages prompts from YAML files."""
    
    def __init__(self, prompts_dir: Optional[Union[str, Path]] = None):
        """Initialize the prompt loader.
        
        Args:
            prompts_dir: Directory containing prompt YAML files. 
                        Defaults to 'prompts' directory relative to project root.
        """
        if prompts_dir is None:
            # Find project root by looking for pyproject.toml
            current_path = Path(__file__).resolve()
            project_root = None
            for parent in current_path.parents:
                if (parent / "pyproject.toml").exists():
                    project_root = parent
                    break
            
            if project_root is None:
                # Fallback to current directory structure
                project_root = Path(__file__).resolve().parents[3]  # smart-test-generator root
            
            self.prompts_dir = project_root / "prompts"
        else:
            self.prompts_dir = Path(prompts_dir)
        
        self._prompt_cache: Dict[str, Dict[str, Any]] = {}
        
        # Ensure prompts directory exists
        if not self.prompts_dir.exists():
            logger.warning(f"Prompts directory not found: {self.prompts_dir}")
    
    @lru_cache(maxsize=16)
    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """Load and cache a YAML file.
        
        Args:
            filename: Name of the YAML file (e.g., 'system_prompts.yaml')
            
        Returns:
            Dictionary containing the loaded YAML content
        """
        file_path = self.prompts_dir / filename
        
        if not file_path.exists():
            logger.error(f"Prompt file not found: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading prompt file {file_path}: {e}")
            return {}
    
    def get_system_prompt(self, extended_thinking: bool = False, 
                         use_legacy: bool = False, config=None) -> str:
        """Get the system prompt for test generation.
        
        Args:
            extended_thinking: Whether to include extended thinking instructions
            use_legacy: Whether to use legacy (pre-2025) format
            config: Configuration object for dynamic prompt customization
            
        Returns:
            Complete system prompt string
        """
        prompts = self._load_yaml_file("system_prompts.yaml")
        test_gen = prompts.get("test_generation", {})
        
        if use_legacy:
            return test_gen.get("legacy_system_prompt", "")
        
        # Build modern prompt
        base = test_gen.get("modern_base_prompt", "")
        
        if extended_thinking:
            base += test_gen.get("extended_thinking_addon", "")
        
        base += test_gen.get("modern_instructions", "")
        
        # Apply config-driven guidance if config is provided
        if config:
            base = self._apply_config_guidance(base, config)
        
        return base
    
    def get_refinement_prompt(self, extended_thinking: bool = False,
                            **template_vars) -> str:
        """Get the refinement system prompt.
        
        Args:
            extended_thinking: Whether to include extended thinking instructions
            **template_vars: Variables for template substitution
            
        Returns:
            Complete refinement prompt string
        """
        prompts = self._load_yaml_file("refinement_prompts.yaml")
        refinement = prompts.get("refinement", {})
        
        base = refinement.get("system_prompt_base", "")
        
        if extended_thinking:
            base += refinement.get("extended_thinking_addon", "")
        
        return self._substitute_template(base, template_vars)
    
    def get_refinement_user_content(self, **template_vars) -> str:
        """Get enhanced user content template for refinement requests with context.
        
        Args:
            **template_vars: Variables for template substitution including:
                - Basic: prompt, run_id, branch, commit, python_version, platform, etc.
                - Git context: repo_meta with recent_changes, test_changes_detail, etc.
                - Pattern analysis: failure_analysis with categories, suggestions, etc.
            
        Returns:
            Formatted user content string with rich context
        """
        try:
            prompts = self._load_yaml_file("refinement_prompts.yaml")
            refinement = prompts.get("refinement", {})
            
            # Build enhanced template variables with safe defaults
            enhanced_vars = dict(template_vars)
            
            # Provide safe defaults for required variables
            enhanced_vars.setdefault("prompt", "No specific prompt provided")
            enhanced_vars.setdefault("run_id", "unknown")
            enhanced_vars.setdefault("branch", "unknown")
            enhanced_vars.setdefault("commit", "unknown")
            enhanced_vars.setdefault("python_version", "unknown")
            enhanced_vars.setdefault("platform", "unknown")
            enhanced_vars.setdefault("tests_written_count", 0)
            enhanced_vars.setdefault("last_run_command", "unknown")
            enhanced_vars.setdefault("failures_total", 0)
            
            # Process git context if available
            git_context_summary = self._build_git_context_summary(template_vars, prompts)
            enhanced_vars["git_context_summary"] = git_context_summary
            
            # Process pattern analysis if available  
            pattern_analysis_summary = self._build_pattern_analysis_summary(template_vars, prompts)
            enhanced_vars["pattern_analysis_summary"] = pattern_analysis_summary
            
            template = refinement.get("user_content_template", "")
            if not template:
                logger.warning("No refinement user content template found")
                return f"Refinement request: {enhanced_vars.get('prompt', 'No prompt')}"
                
            return self._substitute_template(template, enhanced_vars)
            
        except Exception as e:
            logger.error(f"Error building refinement user content: {e}")
            # Return a minimal fallback template
            prompt = template_vars.get("prompt", "No prompt provided")
            run_id = template_vars.get("run_id", "unknown")
            return f"# Test Refinement Request\n\n{prompt}\n\n## Context\n- Run ID: {run_id}"
    
    def get_advanced_refinement_prompt(self, framework: str = "pytest",
                                     encourage_steps: bool = True,
                                     use_examples: bool = True,
                                     decisive: bool = True,
                                     failures: list = None) -> str:
        """Get advanced refinement prompt with configurability.
        
        Args:
            framework: Testing framework name
            encourage_steps: Whether to include step-by-step guidance
            use_examples: Whether to include positive/negative examples
            decisive: Whether to include decisive recommendation guidance
            failures: List of failure information
            
        Returns:
            Complete advanced refinement prompt
        """
        prompts = self._load_yaml_file("refinement_prompts.yaml")
        advanced = prompts.get("advanced_refinement", {})
        
        parts = [advanced.get("base_instruction", "")]
        
        if decisive:
            parts.append(advanced.get("decisive_addon", ""))
        
        if encourage_steps:
            parts.append(advanced.get("step_by_step_addon", ""))
        
        if use_examples:
            parts.append(advanced.get("examples_addon", ""))
        
        # Add framework and constraints
        framework_template = advanced.get("framework_constraints_template", "")
        parts.append(self._substitute_template(framework_template, {"framework": framework}))
        
        # Add failure summary if provided
        if failures:
            failure_list = self._format_failure_list(failures)
            failure_template = advanced.get("failure_summary_template", "")
            parts.append(self._substitute_template(failure_template, {"failure_list": failure_list}))
        
        # Add output format
        parts.append(advanced.get("output_format", ""))
        
        return "\n\n".join(p for p in parts if p)
    
    def get_contextual_prompt(self, base_prompt: str, untested_elements: list,
                            existing_tests: str, use_2025_format: bool = True,
                            quality_target: Optional[float] = None,
                            mutation_guidance: Optional[dict] = None,
                            config=None) -> str:
        """Get contextual prompt for incremental testing.
        
        Args:
            base_prompt: Base system prompt to extend
            untested_elements: List of untested elements
            existing_tests: Sample of existing test content
            use_2025_format: Whether to use 2025 format
            quality_target: Quality score target (optional)
            mutation_guidance: Mutation testing guidance data (optional)
            
        Returns:
            Extended contextual prompt
        """
        prompts = self._load_yaml_file("contextual_prompts.yaml")
        
        # Build incremental context
        incremental = prompts.get("incremental_testing", {})
        if use_2025_format:
            context_template = incremental.get("context_2025_format", "")
        else:
            context_template = incremental.get("context_legacy_format", "")
        
        context = self._substitute_template(context_template, {
            "untested_elements": ", ".join(str(e) for e in untested_elements),
            "existing_tests": existing_tests[:800] if use_2025_format else existing_tests[:1000]
        })
        
        # Add quality guidance if provided
        if quality_target:
            quality_guidance = self._get_quality_guidance(quality_target)
            context += f"\n\n{quality_guidance}"
        
        # Add mutation guidance if provided
        if mutation_guidance:
            mutation_guide = self._get_mutation_guidance(mutation_guidance)
            context += f"\n\n{mutation_guide}"
            
        # Add generation guidance based on config
        generation_guide = self._get_generation_guidance(config)
        if generation_guide:
            context += f"\n\n{generation_guide}"
        
        return base_prompt + context
    
    def get_user_content_template(self, **template_vars) -> str:
        """Get formatted user content template.
        
        Args:
            **template_vars: Variables for substitution (directory_structure, xml_content, etc.)
            
        Returns:
            Formatted user content string
        """
        prompts = self._load_yaml_file("templates.yaml")
        common = prompts.get("common", {})
        
        template = common.get("user_content_template", "")
        return self._substitute_template(template, template_vars)
    
    def get_validation_context(self, imports: dict, classes: dict) -> str:
        """Get validation context for LLM requests.
        
        Args:
            imports: Available imports dictionary
            classes: Class signatures dictionary
            
        Returns:
            Formatted validation context
        """
        prompts = self._load_yaml_file("templates.yaml")
        common = prompts.get("common", {})
        
        template = common.get("validation_context_template", "")
        return self._substitute_template(template, {
            "imports": self._format_json_dict(imports),
            "classes": self._format_json_dict(classes)
        })
    
    def get_token_allocation(self, file_count: int) -> int:
        """Get recommended token allocation based on file count.
        
        Args:
            file_count: Number of files being processed
            
        Returns:
            Recommended token count
        """
        prompts = self._load_yaml_file("templates.yaml")
        token_mgmt = prompts.get("token_management", {})
        allocation = token_mgmt.get("token_allocation", {})
        
        if file_count == 1:
            return allocation.get("single_file", 6000)
        elif file_count <= 3:
            return allocation.get("small_batch", 5000)
        elif file_count <= 6:
            return allocation.get("medium_batch", 4000)
        else:
            return allocation.get("large_batch", 3000)
    
    def get_refinement_token_allocation(self, failure_count: int) -> int:
        """Get recommended token allocation for refinement based on failure count.
        
        Args:
            failure_count: Number of failures to process
            
        Returns:
            Recommended token count for refinement
        """
        prompts = self._load_yaml_file("templates.yaml")
        token_mgmt = prompts.get("token_management", {})
        refinement = token_mgmt.get("refinement_tokens", {})
        
        if failure_count <= 2:
            return refinement.get("simple", 8000)
        elif failure_count <= 5:
            return refinement.get("moderate", 12000)
        else:
            return refinement.get("complex", 16000)
    
    def _get_quality_guidance(self, target: float) -> str:
        """Get quality guidance based on target score."""
        prompts = self._load_yaml_file("contextual_prompts.yaml")
        quality = prompts.get("quality_guidance", {})
        
        if target >= 90:
            template = quality.get("high_standard", "")
        elif target >= 75:
            template = quality.get("good_standard", "")
        else:
            template = quality.get("basic_standard", "")
        
        return self._substitute_template(template, {"target": int(target)})
    
    def _get_mutation_guidance(self, mutation_data: dict) -> str:
        """Get mutation testing guidance based on weak spots."""
        prompts = self._load_yaml_file("contextual_prompts.yaml")
        mutation = prompts.get("mutation_guidance", {})
        
        parts = [mutation.get("header", "")]
        
        weak_spots = mutation_data.get("weak_mutation_spots", [])[:3]  # Limit to top 3
        patterns = mutation.get("patterns", {})
        
        # Group by mutation type
        weak_spots_by_type = {}
        for spot in weak_spots:
            mut_type = getattr(spot, 'mutation_type', {}).get('value', 'unknown')
            if mut_type not in weak_spots_by_type:
                weak_spots_by_type[mut_type] = []
            weak_spots_by_type[mut_type].append(spot)
        
        for mut_type, spots in weak_spots_by_type.items():
            if mut_type in patterns:
                parts.append(patterns[mut_type])
                
                # Add suggestion if available
                if spots and hasattr(spots[0], 'suggested_tests') and spots[0].suggested_tests:
                    suggestion_template = mutation.get("suggestion_template", "")
                    suggestion = self._substitute_template(
                        suggestion_template, 
                        {"suggestion": spots[0].suggested_tests[0]}
                    )
                    parts.append(f"  {suggestion}")
        
        # Add target score if provided
        target = mutation_data.get("mutation_score_target")
        if target:
            target_template = mutation.get("target_score", "")
            parts.append(self._substitute_template(target_template, {"target": int(target)}))
        
        return "\n".join(parts)
    
    def _substitute_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Substitute template variables in a string.
        
        Args:
            template: Template string with {variable} placeholders
            variables: Dictionary of variable values
            
        Returns:
            String with variables substituted
        """
        try:
            # Convert any non-string variables to strings first
            safe_variables = {}
            for key, value in variables.items():
                if isinstance(value, (dict, list)):
                    # For complex types, try to format them nicely
                    try:
                        safe_variables[key] = str(value) if value else ""
                    except Exception:
                        safe_variables[key] = f"<{type(value).__name__}>"
                else:
                    safe_variables[key] = str(value) if value is not None else ""
                    
            return template.format(**safe_variables)
        except KeyError as e:
            logger.warning(f"Missing template variable {e} in prompt template")
            logger.debug(f"Available variables: {list(variables.keys())}")
            return template
        except ValueError as e:
            logger.error(f"Invalid template format in: {template[:100]}... Error: {e}")
            return template
        except Exception as e:
            logger.error(f"Error substituting template variables: {e}")
            logger.debug(f"Template: {template[:100]}...")
            logger.debug(f"Variables: {list(variables.keys())}")
            return template
    
    def _format_json_dict(self, data: dict) -> str:
        """Format dictionary as pretty JSON string."""
        try:
            import json
            return json.dumps(data, indent=2)
        except Exception:
            return str(data)
    
    def _format_failure_list(self, failures: list) -> str:
        """Format failure list for prompt inclusion."""
        formatted_failures = []
        
        for idx, f in enumerate(failures, 1):
            nodeid = f.get("nodeid", "")
            message = f.get("message", "")
            assertion = f.get("assertion_diff", "")
            snippet = f.get("code_context", {})
            
            loc = ""
            if snippet:
                path = snippet.get("path", "")
                start = snippet.get("start_line", "")
                end = snippet.get("end_line", "")
                if path and start and end:
                    loc = f"{path}:{start}-{end}"
            
            failure_line = f"{idx}. {nodeid} — {message}"
            if assertion:
                failure_line += f"\n   Assertion: {assertion}"
            if loc:
                failure_line += f"\n   Context: {loc}"
            
            formatted_failures.append(failure_line)
        
        return "\n".join(formatted_failures)
    
    def _apply_config_guidance(self, prompt: str, config) -> str:
        """Apply configuration-driven guidance to the prompt.
        
        Args:
            prompt: Base prompt with placeholder templates
            config: Configuration object with style and generation settings
            
        Returns:
            Prompt with placeholders replaced by appropriate guidance
        """
        prompts = self._load_yaml_file("system_prompts.yaml")
        guidance = prompts.get("guidance_templates", {})
        
        # Get docstring guidance
        docstring_setting = config.get('test_generation.generation.include_docstrings', True)
        if docstring_setting is True:
            docstring_guidance = guidance.get("docstrings", {}).get("include_full", "")
        elif docstring_setting == "minimal":
            docstring_guidance = guidance.get("docstrings", {}).get("include_minimal", "")
        else:  # False or any other value
            docstring_guidance = guidance.get("docstrings", {}).get("exclude", "")
        
        # Get assertion style guidance
        assertion_style = config.get('test_generation.style.assertion_style', 'pytest')
        assertion_guidance = guidance.get("assertions", {}).get(assertion_style, 
                                                              guidance.get("assertions", {}).get("pytest", ""))
        
        # Get mock library guidance
        mock_library = config.get('test_generation.style.mock_library', 'unittest.mock')
        mock_key = mock_library.replace('.', '_').replace('-', '_')  # unittest.mock -> unittest_mock
        mock_guidance = guidance.get("mocking", {}).get(mock_key, 
                                                      guidance.get("mocking", {}).get("unittest_mock", ""))
        
        # Replace placeholders in the prompt
        prompt = prompt.replace("{docstring_guidance}", docstring_guidance)
        prompt = prompt.replace("{assertion_guidance}", assertion_guidance)
        prompt = prompt.replace("{mock_guidance}", mock_guidance)
        
        return prompt
    
    def _build_git_context_summary(self, template_vars: dict, prompts: dict) -> str:
        """Build git context summary for refinement prompts."""
        try:
            repo_meta = template_vars.get("repo_meta", {})
            recent_changes = repo_meta.get("recent_changes", {})
            
            if not recent_changes:
                return ""
            
            refinement = prompts.get("refinement", {})
            formats = refinement.get("recent_changes_formats", {})
            
            # Build recent changes summary
            changes_data = {
            "time_range": recent_changes.get("time_range", "recent"),
            "total_files_changed": recent_changes.get("total_files_changed", 0),
            "test_files_changed": recent_changes.get("test_files_changed", 0),
            "source_files_changed": recent_changes.get("source_files_changed", 0),
            "recent_commit_messages": ", ".join(recent_changes.get("recent_commit_messages", [])[:2])
            }
            
            if changes_data["total_files_changed"] > 0:
                recent_changes_summary = self._substitute_template(
                formats.get("detailed", formats.get("basic", "")), 
                changes_data
                )
            else:
                recent_changes_summary = ""
            
            # Build test changes detail
            test_changes_detail = ""
            test_changes = repo_meta.get("test_changes_detail", [])
            if test_changes:
                details = []
                for change in test_changes[:3]:
                    details.append(f"  - `{change['file']}`: {change['status']} (+{change['lines_added']}/-{change['lines_removed']})")
                test_changes_detail = "- **Recent Test File Changes**:\n" + "\n".join(details)
            
            # Build relevant changes context
            relevant_changes = ""
            relevant_test_changes = template_vars.get("relevant_test_changes", [])
            if relevant_test_changes:
                changes = []
                for change in relevant_test_changes:
                    changes.append(f"  - `{change['file']}` ({change['status']})")
                relevant_changes = "- **Files Affecting Current Failures**:\n" + "\n".join(changes)
            
            # Assemble git context
            context_template = refinement.get("git_context_summary_template", "")
            return self._substitute_template(context_template, {
                "recent_changes_summary": recent_changes_summary,
                "test_changes_detail": test_changes_detail,
                "relevant_changes": relevant_changes
            })
        except Exception as e:
            logger.warning(f"Error building git context summary: {e}")
            return ""  # Return empty string on error
    
    def _build_pattern_analysis_summary(self, template_vars: dict, prompts: dict) -> str:
        """Build pattern analysis summary for refinement prompts."""
        try:
            failure_analysis = template_vars.get("failure_analysis", {})
            
            if not failure_analysis:
                return ""
            
            refinement = prompts.get("refinement", {})
            pattern_formats = refinement.get("pattern_formats", {})
            historical_formats = refinement.get("historical_formats", {})
            
            # Build failure categories list
            pattern_frequencies = failure_analysis.get("pattern_frequencies", {})
            categories_with_counts = ", ".join([
                f"{category} ({count})" 
                for category, count in pattern_frequencies.items()
            ])
            
            # Build trending patterns
            trending_patterns = failure_analysis.get("trending_patterns", [])
            trending_categories = ", ".join([
                f"{item['category']} (↑{item['trend_score']:.1f}x)"
                for item in trending_patterns[:3]
            ])
            
            # Build fix suggestions list
            fix_suggestions = failure_analysis.get("fix_suggestions", [])
            fix_suggestions_list = ""
            if fix_suggestions:
                suggestions = []
                for i, suggestion in enumerate(fix_suggestions[:3], 1):
                    suggestion_text = self._substitute_template(
                        pattern_formats.get("fix_suggestions_format", ""),
                        {
                            "priority": i,
                            "title": suggestion.get("title", ""),
                            "category": suggestion.get("category", ""),
                            "description": suggestion.get("description", ""),
                            "code_example": f"\n       ```python\n       {suggestion.get('code_example', '')}\n       ```" if suggestion.get("code_example") else ""
                        }
                    )
                    suggestions.append(suggestion_text)
                fix_suggestions_list = "\n".join(suggestions)
            
            # Build historical success context
            historical_success_context = ""
            success_rates = failure_analysis.get("historical_success_rates", {})
            if success_rates:
                rates_summary = ", ".join([
                    f"{category}: {rate:.0%}"
                    for category, rate in success_rates.items()
                    if rate > 0
                ])
                if rates_summary:
                    historical_success_context = self._substitute_template(
                        historical_formats.get("success_rates", ""),
                        {"success_rates_summary": rates_summary}
                    )
            
            # Add confidence context
            confidence_scores = failure_analysis.get("confidence_scores", {})
            if confidence_scores:
                avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
                confidence_context = self._substitute_template(
                    historical_formats.get("confidence_context", ""),
                    {"confidence": int(avg_confidence * 100)}
                )
                if historical_success_context:
                    historical_success_context += f"\n- {confidence_context}"
                else:
                    historical_success_context = f"- {confidence_context}"
            
            # Assemble pattern analysis
            context_template = refinement.get("pattern_analysis_summary_template", "")
            return self._substitute_template(context_template, {
                "failure_categories": categories_with_counts,
                "trending_patterns": trending_categories if trending_categories else "None detected",
                "fix_suggestions_list": fix_suggestions_list,
                "historical_success_context": historical_success_context
            })
        except Exception as e:
            logger.warning(f"Error building pattern analysis summary: {e}")
            return ""  # Return empty string on error
    
    def _get_generation_guidance(self, config) -> str:
        """Get generation guidance based on configuration settings."""
        if not config:
            return ""
            
        prompts = self._load_yaml_file("contextual_prompts.yaml")
        generation_guidance = prompts.get("generation_guidance", {})
        
        parts = []
        
        # Fixture guidance
        generate_fixtures = config.get('test_generation.generation.generate_fixtures', True)
        fixture_guidance = generation_guidance.get("fixtures", {})
        if generate_fixtures:
            fixture_guide = fixture_guidance.get("generate", "")
        else:
            fixture_guide = fixture_guidance.get("disable", "")
        
        if fixture_guide:
            parts.append(fixture_guide)
        
        # Parametrize guidance
        parametrize_similar = config.get('test_generation.generation.parametrize_similar_tests', True)
        parametrize_guidance = generation_guidance.get("parametrize", {})
        if parametrize_similar:
            param_guide = parametrize_guidance.get("enable", "")
        else:
            param_guide = parametrize_guidance.get("disable", "")
            
        if param_guide:
            parts.append(param_guide)
        
        return "\n\n".join(parts)


# Global prompt loader instance
_prompt_loader = None


def get_prompt_loader() -> PromptLoader:
    """Get the global prompt loader instance."""
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader()
    return _prompt_loader


def reload_prompts():
    """Reload all prompts (useful for development)."""
    global _prompt_loader
    if _prompt_loader:
        _prompt_loader._load_yaml_file.cache_clear()
        _prompt_loader._prompt_cache.clear()

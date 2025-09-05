"""Provider-agnostic LLM client that composes a transport and shared helpers.

This class implements the existing `LLMClient` interface while ensuring the
transport classes only perform network I/O. Prompt construction, context
building, JSON parsing, validation, and cost logging are handled here.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional

from smart_test_generator.config import Config
from smart_test_generator.utils.prompt_loader import get_prompt_loader
from smart_test_generator.generation.llm_helpers import (
    extract_codebase_imports,
    build_user_content,
    extract_json_content,
    try_recover_partial_results,
    validate_and_fix_tests,
    log_verbose_prompt,
)
from smart_test_generator.generation.clients.base import LLMTransport

logger = logging.getLogger(__name__)


class OrchestratedLLMClient:
    """An `LLMClient` implementation that delegates I/O to an `LLMTransport`."""

    def __init__(
        self,
        *,
        transport: LLMTransport,
        model_name: str,
        cost_manager=None,
        feedback=None,
        config: Optional[Config] = None,
        extended_thinking: bool = False,
    ):
        self.transport = transport
        self.model_name = model_name
        self.cost_manager = cost_manager
        self.feedback = feedback
        self.config = config
        self.extended_thinking = extended_thinking

    def generate_unit_tests(
        self,
        system_prompt: str,
        xml_content: str,
        directory_structure: str,
        source_files: List[str] = None,
        project_root: str = None,
    ) -> Dict[str, str]:
        # Extract minimal codebase info for validation and import hints
        codebase_info = extract_codebase_imports(source_files or [], project_root or "")
        user_content = build_user_content(
            xml_content=xml_content,
            directory_structure=directory_structure,
            codebase_info=codebase_info,
        )

        # Token allocation per file count via prompt templates
        file_count = len(re.findall(r'<file\s+filename=', xml_content))
        prompt_loader = get_prompt_loader()
        tokens_per_file = prompt_loader.get_token_allocation(file_count)
        estimated_output_size = max(1, file_count) * tokens_per_file
        # Apply conservative caps compatible with most providers
        max_tokens = min(32000, max(2000, estimated_output_size))

        log_verbose_prompt(
            feedback=self.feedback,
            model_name=self.model_name,
            system_prompt=system_prompt,
            user_content=user_content,
            file_count=file_count,
        )

        raw = self.transport.generate(
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=max_tokens,
            temperature=0.3,
            response_json=True,
        )

        # Log cost if usage available
        usage = self.transport.get_token_usage()
        if usage and self.cost_manager:
            try:
                self.cost_manager.log_token_usage(self.model_name, int(usage.get('input', 0)), int(usage.get('output', 0)))
            except Exception:
                pass

        if not raw or not str(raw).strip():
            logger.error(f"Empty response from transport for model {self.model_name}")
            return {}

        json_content = extract_json_content(str(raw))
        try:
            tests_dict = json.loads(json_content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error from {self.model_name}: {e}")
            tests_dict = try_recover_partial_results(json_content, e)

        return validate_and_fix_tests(tests_dict, codebase_info)

    def refine_tests(self, request: Dict) -> str:
        payload = request.get("payload", {}) or {}
        prompt_text = request.get("prompt", "")
        if not prompt_text or not payload:
            return json.dumps({
                "updated_files": [],
                "rationale": "No refinement data provided",
                "plan": "Cannot proceed without proper refinement context",
            })

        # Build refinement prompts
        prompt_loader = get_prompt_loader()
        system_prompt = prompt_loader.get_refinement_prompt(extended_thinking=self.extended_thinking, config=self.config)
        user_content = prompt_loader.get_refinement_user_content(
            prompt=prompt_text,
            run_id=payload.get('run_id', 'unknown'),
            branch=payload.get('repo_meta', {}).get('branch', 'unknown'),
            commit=payload.get('repo_meta', {}).get('commit', 'unknown'),
            python_version=payload.get('environment', {}).get('python', 'unknown'),
            platform=payload.get('environment', {}).get('platform', 'unknown'),
            tests_written_count=len(payload.get('tests_written', [])),
            last_run_command=' '.join(payload.get('last_run_command', [])),
            failures_total=payload.get('failures_total', 0),
            repo_meta=payload.get('repo_meta', {}),
            failure_analysis=payload.get('failure_analysis', {}),
            relevant_test_changes=payload.get('relevant_test_changes', []),
            codebase_context=payload.get('codebase_context', {}),
        )

        # Token allocation for refinement
        failure_count = len(payload.get("failures", []))
        max_tokens = prompt_loader.get_refinement_token_allocation(failure_count)

        log_verbose_prompt(
            feedback=self.feedback,
            model_name=self.model_name,
            system_prompt=system_prompt,
            user_content=user_content,
            file_count=1,
        )

        raw = self.transport.refine(
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=max_tokens,
            temperature=0.2,
            response_json=True,
        )

        # Log cost if usage available
        usage = self.transport.get_token_usage()
        if usage and self.cost_manager:
            try:
                self.cost_manager.log_token_usage(self.model_name, int(usage.get('input', 0)), int(usage.get('output', 0)))
            except Exception:
                pass

        if not raw or not str(raw).strip():
            return json.dumps({
                "updated_files": [],
                "rationale": "Empty response from AI model",
                "plan": "No refinement suggestions provided",
            })

        json_content = extract_json_content(str(raw))
        try:
            refinement_result = json.loads(json_content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in refinement response from {self.model_name}: {e}")
            recovered = try_recover_partial_results(json_content, e)
            if not isinstance(recovered, dict) or "updated_files" not in recovered:
                recovered = {
                    "updated_files": [],
                    "rationale": "Failed to recover refinement results from malformed JSON",
                    "plan": "JSON parsing error occurred during recovery",
                }
            return json.dumps(recovered)

        if "updated_files" not in refinement_result:
            refinement_result["updated_files"] = []
        return json.dumps(refinement_result)



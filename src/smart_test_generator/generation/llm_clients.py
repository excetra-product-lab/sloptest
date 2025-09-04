"""LLM client implementations for test generation."""

import json
import logging
import re
import requests
import ast
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Set
from pathlib import Path

from smart_test_generator.exceptions import LLMClientError, AuthenticationError
from smart_test_generator.utils.prompt_loader import get_prompt_loader

try:
    from langchain_aws import ChatBedrock
except Exception:  # pragma: no cover - optional dependency handled at runtime
    ChatBedrock = None

try:
    import boto3
    from botocore.exceptions import BotoCoreError, NoCredentialsError
except Exception:  # pragma: no cover - optional dependency handled at runtime
    boto3 = None
    BotoCoreError = Exception
    NoCredentialsError = Exception

logger = logging.getLogger(__name__)


def validate_generated_test_safe(test_content: str, filepath: str, available_imports: Dict[str, str], 
                                config: 'Config' = None) -> tuple[bool, List[str]]:
    """Validate generated test content with enhanced security measures."""
    errors = []
    
    # Get security config or use defaults
    if config:
        security_config = config.get('security', {})
        block_dangerous = security_config.get('block_dangerous_patterns', True)
        max_file_size = security_config.get('max_generated_file_size', 50000)
    else:
        block_dangerous = True
        max_file_size = 50000
    
    # Security check 1: Reject obviously malicious patterns (if enabled)
    if block_dangerous:
        dangerous_patterns = [
            r'__import__\s*\(',
            r'eval\s*\(',
            r'exec\s*\(',
            r'compile\s*\(',
            r'open\s*\(["\'][^"\']*["\'],\s*["\'][wa]',  # Only block file writing/appending
            r'subprocess',
            r'os\.system',
            r'os\.popen',
            r'__builtins__',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, test_content, re.IGNORECASE):
                errors.append(f"Security: Potentially dangerous pattern '{pattern}' found in {filepath}")
                return False, errors
    
    # Security check 2: Limit content size
    if len(test_content) > max_file_size:
        errors.append(f"Security: Generated test file too large ({len(test_content)} chars) in {filepath}")
        return False, errors
    
    # Security check 3: Basic syntax validation without full AST parsing
    try:
        # Use compile() in a safer way - just check syntax, don't execute
        compile(test_content, filepath, 'exec', dont_inherit=True, optimize=0)
    except SyntaxError as e:
        errors.append(f"Syntax error in {filepath}: {e}")
        return False, errors
    except Exception as e:
        errors.append(f"Validation error in {filepath}: {e}")
        return False, errors
    
    # Skip import validation - let tests import anything and fail at runtime if needed
    # This is more flexible and less restrictive than trying to validate all possible imports
    
    return len(errors) == 0, errors





def validate_generated_test(test_content: str, filepath: str, available_imports: Dict[str, str], 
                           config: 'Config' = None) -> tuple[bool, List[str]]:
    """Validate generated test content for common issues."""
    # Use the safer validation method
    return validate_generated_test_safe(test_content, filepath, available_imports, config)


def extract_codebase_imports(source_files: List[str], project_root: str) -> Dict[str, any]:
    """Extract available imports and class signatures from the codebase."""
    available_imports = {}
    class_signatures = {}
    
    for file_path in source_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to extract classes and their constructors (safe - only for analysis)
            tree = ast.parse(content)
            
            # Get relative module path for imports
            rel_path = os.path.relpath(file_path, project_root)
            module_path = rel_path.replace(os.sep, '.').replace('.py', '')
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    full_class_name = f"{module_path}.{node.name}"
                    
                    # Extract __init__ signature for dataclasses/classes
                    init_method = None
                    required_fields = []
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                            init_method = item
                            break
                    
                    # Check if it's a dataclass by looking for @dataclass decorator
                    is_dataclass = any(
                        (isinstance(d, ast.Name) and d.id == 'dataclass') or
                        (isinstance(d, ast.Attribute) and d.attr == 'dataclass')
                        for d in node.decorator_list
                    )
                    
                    if is_dataclass:
                        # For dataclasses, extract field annotations (more reliable than __init__)
                        for item in node.body:
                            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                                field_name = item.target.id
                                # Check if field has a default value or is marked with field(default_factory=...)
                                has_default = (item.value is not None) or (
                                    isinstance(item.value, ast.Call) and 
                                    isinstance(item.value.func, ast.Name) and 
                                    item.value.func.id == 'field'
                                )
                                if not has_default:
                                    required_fields.append(field_name)
                    elif init_method:
                        # For regular classes, extract required parameters from __init__
                        for arg in init_method.args.args[1:]:  # Skip 'self'
                            # Check if argument has a default value
                            defaults_offset = len(init_method.args.args) - len(init_method.args.defaults or [])
                            arg_index = init_method.args.args.index(arg)
                            has_default = arg_index >= defaults_offset
                            if not has_default:
                                required_fields.append(arg.arg)
                    
                    class_signatures[node.name] = {
                        'full_name': full_class_name,
                        'is_dataclass': is_dataclass,
                        'required_fields': required_fields,
                        'module': module_path
                    }
                    
                    available_imports[node.name] = module_path
                
                elif isinstance(node, ast.FunctionDef):
                    # Track top-level functions too
                    available_imports[node.name] = module_path
                    
        except Exception as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            continue
    
    return {
        'imports': available_imports,
        'classes': class_signatures
    }


def get_legacy_system_prompt() -> str:
    """Get the legacy system prompt for test generation (pre-2025 guidelines)."""
    prompt_loader = get_prompt_loader()
    return prompt_loader.get_system_prompt(extended_thinking=False, use_legacy=True)


def get_system_prompt(config: 'Config' = None, extended_thinking: bool = False) -> str:
    """Get the system prompt for test generation."""
    prompt_loader = get_prompt_loader()
    
    # Check if we should use 2025 guidelines (default: True)
    use_legacy = config and not config.get('prompt_engineering.use_2025_guidelines', True)
    
    return prompt_loader.get_system_prompt(
        extended_thinking=extended_thinking, 
        use_legacy=use_legacy,
        config=config
    )


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate_unit_tests(self, system_prompt: str, xml_content: str, directory_structure: str, 
                           source_files: List[str] = None, project_root: str = None) -> Dict[str, str]:
        """Generate unit tests for the provided code."""
        pass
    
    def refine_tests(self, request: Dict) -> str:
        """Refine existing tests based on failure information.

        Args:
            request: Dictionary containing payload and prompt for refinement

        Returns:
            JSON string with updated_files, rationale, and plan
        """
        # Default implementation - subclasses should override for specific behavior
        payload = request.get("payload", {})
        prompt = request.get("prompt", "")

        # For now, return empty response - indicates no refinements
        return json.dumps({
            "updated_files": [],
            "rationale": "No refinements needed",
            "plan": "Tests appear to be working as expected"
        })

    def _call_refine_api(self, system_prompt: str, user_content: str, model_name: str) -> str:
        """Helper method to call LLM API for refinement."""
        # This is a placeholder that concrete implementations should override
        return self.refine_tests({"payload": {}, "prompt": ""})

    def _extract_codebase_info(self, source_files: List[str], project_root: str) -> Dict:
        """Extract codebase information for validation."""
        codebase_info = {}
        if source_files and project_root:
            codebase_info = extract_codebase_imports(source_files, project_root)
        return codebase_info

    def _build_validation_context(self, codebase_info: Dict) -> str:
        """Build validation context string from codebase info."""
        if not codebase_info:
            return ""
        
        prompt_loader = get_prompt_loader()
        return prompt_loader.get_validation_context(
            imports=codebase_info.get('imports', {}),
            classes=codebase_info.get('classes', {})
        )

    def _build_user_content(self, xml_content: str, directory_structure: str, codebase_info: Dict) -> str:
        """Build user content for LLM request."""
        validation_context = self._build_validation_context(codebase_info)
        
        prompt_loader = get_prompt_loader()
        return prompt_loader.get_user_content_template(
            directory_structure=directory_structure,
            xml_content=xml_content,
            validation_context=validation_context
        )

    def _log_verbose_prompt(self, model_name: str, system_prompt: str, user_content: str, file_count: int):
        """Log verbose prompt information if feedback is available."""
        if hasattr(self, 'feedback') and self.feedback:
            content_size = len(system_prompt + user_content)
            self.feedback.verbose_prompt_display(
                model_name=model_name,
                system_prompt=system_prompt,
                user_content=user_content,
                content_size=content_size,
                file_count=file_count
            )

    def _validate_and_fix_tests(self, tests_dict: Dict[str, str], codebase_info: Dict) -> Dict[str, str]:
        """Validate generated tests and attempt to fix basic issues."""
        if not codebase_info or not tests_dict:
            return tests_dict
            
        available_imports = codebase_info.get('imports', {})
        validated_tests = {}
        
        for filepath, test_content in tests_dict.items():
            is_valid, errors = validate_generated_test(test_content, filepath, available_imports, None)
            if is_valid:
                validated_tests[filepath] = test_content
            else:
                logger.warning(f"Generated test for {filepath} has validation errors: {errors}")
                # Try to fix basic issues automatically
                fixed_content = self._attempt_basic_fixes(test_content, errors)
                if fixed_content:
                    is_valid_fixed, _ = validate_generated_test(fixed_content, filepath, available_imports, None)
                    if is_valid_fixed:
                        validated_tests[filepath] = fixed_content
                        logger.info(f"Auto-fixed test for {filepath}")
                    else:
                        logger.error(f"Could not auto-fix test for {filepath}")
                else:
                    logger.error(f"Skipping invalid test for {filepath}")
        
        return validated_tests

    def _extract_json_content(self, content: str) -> str:
        """Extract JSON content from the response."""
        json_content = content.strip()

        # Remove any markdown formatting
        if json_content.startswith("```"):
            # Find the actual JSON content
            lines = json_content.split('\n')
            start_idx = 0
            end_idx = len(lines)

            for i, line in enumerate(lines):
                if line.strip().startswith("```"):
                    if start_idx == 0:
                        start_idx = i + 1
                    else:
                        end_idx = i
                        break

            json_content = '\n'.join(lines[start_idx:end_idx])

        # Ensure we have valid JSON boundaries
        json_content = json_content.strip()
        if not json_content.startswith("{"):
            start = json_content.find("{")
            if start >= 0:
                json_content = json_content[start:]

        if not json_content.endswith("}"):
            # Try to find the last complete JSON object
            # Count braces to find where JSON should end
            brace_count = 0
            last_valid_pos = -1

            for i, char in enumerate(json_content):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_valid_pos = i + 1

            if last_valid_pos > 0:
                json_content = json_content[:last_valid_pos]
                logger.warning(f"⚠️  JSON appears truncated - found complete JSON up to position {last_valid_pos}")
            else:
                logger.warning("⚠️  JSON appears severely truncated - no complete JSON object found")

        return json_content

    def _try_recover_partial_results(self, json_content: str, error: json.JSONDecodeError) -> Dict[str, str]:
        """Try to recover partial results from invalid JSON."""
        # Log where the error occurred
        error_line = json_content.count('\n', 0, error.pos) + 1 if hasattr(error, 'pos') else 0
        logger.error(f"JSON parse error at line {error_line}, position {error.pos if hasattr(error, 'pos') else 'unknown'}")
        
        # Check if this looks like truncation
        if json_content.rstrip().endswith('",') or json_content.rstrip().endswith('"'):
            logger.warning("Response appears to be truncated (ends abruptly)")
        elif json_content.count('{') > json_content.count('}'):
            logger.warning("Response appears to be truncated (unmatched opening braces)")

        # Try to salvage partial results
        logger.info("Attempting to recover partial JSON results...")
        try:
            # Method 1: Try to fix truncated JSON by adding closing braces
            fixed_json = json_content.rstrip()
            
            # Handle common truncation patterns
            if fixed_json.endswith('",'):
                # Likely truncated in the middle of a string value
                fixed_json = fixed_json[:-1]  # Remove trailing comma
            elif fixed_json.endswith('"'):
                # Truncated right after a string value
                pass  # Keep as is
            elif fixed_json.endswith(','):
                # Truncated after a comma
                fixed_json = fixed_json[:-1]  # Remove trailing comma
            
            # Count open braces and try to close them
            open_braces = fixed_json.count('{') - fixed_json.count('}')
            if open_braces > 0:
                # Add missing closing braces
                fixed_json += '}' * open_braces
                
                try:
                    result = json.loads(fixed_json)
                    logger.warning(f"✓ Recovered truncated JSON by adding {open_braces} closing braces")
                    logger.warning(f"✓ Recovered {len(result)} test files from truncated response")
                    return result
                except json.JSONDecodeError:
                    pass
            
            # Method 2: Try to extract individual file entries
            partial_results = {}

            # Use regex to find complete file entries
            # Match patterns like "filepath": "content"
            pattern = r'"([^"]+\.py)"\s*:\s*"((?:[^"\\]|\\.)*)"'
            matches = re.findall(pattern, json_content, re.DOTALL)

            for filepath, content in matches:
                # Unescape the content
                try:
                    # Use json.loads to properly unescape
                    unescaped_content = json.loads(f'"{content}"')
                    partial_results[filepath] = unescaped_content
                    logger.info(f"Recovered tests for: {filepath}")
                except Exception as e:
                    logger.warning(f"Failed to parse content for: {filepath} - {e}")

            if partial_results:
                logger.warning(f"Partially recovered tests for {len(partial_results)} files")
                return partial_results
                
            # Method 3: Try to find and extract the largest valid JSON substring
            logger.info("Attempting to find largest valid JSON substring...")
            best_result = {}
            best_length = 0
            
            # Try different ending positions
            for end_pos in range(len(json_content), 0, -100):
                try:
                    test_json = json_content[:end_pos]
                    # Try to balance braces
                    open_braces = test_json.count('{') - test_json.count('}')
                    if open_braces > 0:
                        test_json += '}' * open_braces
                    
                    result = json.loads(test_json)
                    if len(str(result)) > best_length:
                        best_result = result
                        best_length = len(str(result))
                        
                except json.JSONDecodeError:
                    continue
                    
            if best_result:
                logger.info(f"Recovered largest valid JSON subset with {len(best_result)} entries")
                return best_result

        except Exception as recovery_error:
            logger.error(f"Failed to recover partial results: {recovery_error}")

        return {}

    def _attempt_basic_fixes(self, content: str, errors: List[str]) -> str:
        """Attempt to fix basic syntax and mock configuration issues."""
        try:
            # Fix common mock configuration issues that cause runtime errors
            if "mock.get.return_value = Mock" in content:
                content = content.replace("mock.get.return_value = Mock", "mock.get.return_value = []")
            
            if "mock_config.get(" in content and "return_value" not in content:
                # Add proper mock configuration for config objects
                import_section = content.split("def test_")[0] if "def test_" in content else content[:500]
                if "mock_config = Mock()" in import_section and "mock_config.get.return_value" not in import_section:
                    content = content.replace(
                        "mock_config = Mock()",
                        "mock_config = Mock()\\nmock_config.get.return_value = []"
                    )
            
            # Add missing ANY import if it's being used but not imported
            if "ANY" in content and "from unittest.mock import" in content and "ANY" not in content.split("from unittest.mock import")[1].split("\n")[0]:
                content = content.replace(
                    "from unittest.mock import",
                    "from unittest.mock import ANY,"
                )
            
            # Try parsing again to see if fixes worked
            compile(content, "test_fix_validation", 'exec', dont_inherit=True, optimize=0)
            return content
        except:
            return None


class BedrockClient(LLMClient):
    """Client for interacting with AWS Bedrock using langchain-aws ChatBedrock with STS assume-role."""

    def __init__(self, *, role_arn: str, inference_profile: str, region: str = "us-east-1", 
                 extended_thinking: bool = False, cost_manager=None, feedback=None, config=None):
        if boto3 is None:
            raise LLMClientError(
                "boto3 is not installed",
                suggestion="Install with: pip install boto3"
            )
        if ChatBedrock is None:
            raise LLMClientError(
                "langchain-aws is not installed",
                suggestion="Install with: pip install langchain-aws"
            )

        self.region = region
        self.cost_manager = cost_manager
        self.inference_profile = inference_profile
        self.extended_thinking = extended_thinking
        self.feedback = feedback
        self.config = config
        # Assume role to obtain temporary credentials
        try:
            base_session = boto3.Session(region_name=region)
            sts = base_session.client("sts", region_name=region)
            assumed = sts.assume_role(RoleArn=role_arn, RoleSessionName="bedrock")
            creds = assumed["Credentials"]
        except Exception as e:
            raise AuthenticationError(
                f"Failed to assume role for Bedrock: {e}",
                suggestion="Verify RoleArn and trust policy, and that your caller has sts:AssumeRole permission."
            )

        # Create bedrock-runtime client with 5-minute read timeout using the assumed credentials
        try:
            import botocore.config as botocore_config
            config = botocore_config.Config(read_timeout=300, retries={"max_attempts": 3, "mode": "standard"})
            session = boto3.Session(
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
                region_name=region,
            )
            bedrock_runtime = session.client("bedrock-runtime", config=config)
        except Exception as e:
            raise AuthenticationError(
                f"Failed to create Bedrock runtime client: {e}",
                suggestion="Check region and permissions for Bedrock runtime."
            )

        # Initialize ChatBedrock using the inference profile (as model_id) and provider anthropic
        try:
            # ChatBedrock uses 'inference_profile' and can receive a runtime_client
            # Pass the inference profile as model id
            self.chat = ChatBedrock(
                model_id=inference_profile,
                region_name=region,
                provider="anthropic",
                config=config,
                client=bedrock_runtime,
            )
        except Exception as e:
            raise LLMClientError(
                f"Failed to initialize ChatBedrock: {e}",
                suggestion="Ensure the inference profile exists and the provider is correct."
            )

    def generate_unit_tests(self, system_prompt: str, xml_content: str, directory_structure: str,
                            source_files: List[str] = None, project_root: str = None) -> Dict[str, str]:
        """Generate unit tests using Bedrock Converse/Invoke API, assuming Claude on Bedrock."""
        # Extract codebase information for validation
        codebase_info = self._extract_codebase_info(source_files, project_root)
        user_content = self._build_user_content(xml_content, directory_structure, codebase_info)

        # Calculate optimal parameters for Bedrock
        file_count = len(re.findall(r'<file\s+filename=', xml_content))
        
        # Token allocation for Bedrock - same logic as Claude for consistency
        if file_count == 1:
            tokens_per_file = 6000  # Single files get detailed tests
        elif file_count <= 3:
            tokens_per_file = 5000  # Small batches get good detail
        elif file_count <= 6:
            tokens_per_file = 4000  # Medium batches get moderate detail
        else:
            tokens_per_file = 3000  # Large batches get focused tests
        
        estimated_output_size = file_count * tokens_per_file
        max_tokens = min(32000, max(8000, estimated_output_size))

        logger.info(f"Using Bedrock with max_tokens: {max_tokens:,} for {file_count} files ({tokens_per_file:,} tokens per file, estimated total: {estimated_output_size:,})")

        # Log verbose prompt information if feedback is available
        self._log_verbose_prompt(f"AWS Bedrock ({self.inference_profile})", system_prompt, user_content, file_count)

        try:
            # Use langchain ChatBedrock with a system + user message structure and token limits
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ]
            
            # Set generation parameters including max_tokens
            generation_kwargs = {
                "max_tokens": max_tokens,
                "temperature": 0.3,
            }
            
            # Invoke with generation parameters
            result = self.chat.invoke(messages, **generation_kwargs)
            content = getattr(result, "content", "") or ""
            # Normalize possible structured content into text
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    else:
                        try:
                            # Support objects with attribute access
                            maybe_text = getattr(part, "text", None)
                            if maybe_text:
                                text_parts.append(maybe_text)
                        except Exception:
                            continue
                content = "\n".join(text_parts)

            # Check if we have content to parse
            if not content or not content.strip():
                logger.error("Empty response from Bedrock API")
                raise LLMClientError(
                    "Empty response from Bedrock API",
                    suggestion="The AI model returned an empty response. Try reducing context size or check your API limits."
                )
            
            # Extract JSON content with truncation handling
            json_content = self._extract_json_content(content)
            
            try:
                tests_dict = json.loads(json_content)
                logger.info(f"Successfully parsed tests for {len(tests_dict)} files from Bedrock")
                
                # Check if we got tests for all expected files
                if len(tests_dict) < file_count:
                    logger.warning(f"Only received tests for {len(tests_dict)}/{file_count} files from Bedrock")
                    logger.error("Response may have been truncated due to token limits.")
                    logger.error(f"Try reducing batch size (current: {file_count} files) or simplify the source code.")
                    
                    if file_count > 1:
                        logger.error(f"SUGGESTION: Try running with --batch-size {max(1, file_count // 2)} to reduce context size.")
                        
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error from Bedrock: {e}")
                logger.error(f"Failed content preview: {repr(json_content[:200])}")
                if "Expecting value: line 1 column 1 (char 0)" in str(e):
                    logger.error("Bedrock returned empty or non-JSON content")
                    logger.debug(f"Full response: {repr(content)}")
                
                # Try to recover partial results using the same logic as Claude
                tests_dict = self._try_recover_partial_results(json_content, e)

            return self._validate_and_fix_tests(tests_dict, codebase_info)
        except (BotoCoreError, NoCredentialsError) as e:
            raise AuthenticationError(
                f"AWS Bedrock authentication failed: {e}",
                suggestion="Ensure AWS credentials are configured (env, config/credentials files, or SSO)."
            )
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error from Bedrock: {e}")
            logger.error(f"Failed content preview: {repr(content[:200] if 'content' in locals() else 'No content')}")
            if "Expecting value: line 1 column 1 (char 0)" in str(e):
                logger.error("Bedrock returned empty or non-JSON content")
                if 'content' in locals():
                    logger.debug(f"Full response: {repr(content)}")
            return {}
        except Exception as e:
            raise LLMClientError(
                f"Bedrock request failed: {e}",
                suggestion="Verify model ID, region, and permissions for Bedrock use."
            )

    def refine_tests(self, request: Dict) -> str:
        """Refine existing tests based on failure information using AWS Bedrock."""
        payload = request.get("payload", {})
        prompt = request.get("prompt", "")

        if not prompt or not payload:
            logger.warning("Empty refinement prompt or payload")
            return json.dumps({
                "updated_files": [],
                "rationale": "No refinement data provided",
                "plan": "Cannot proceed without proper refinement context"
            })

        # Create system prompt for refinement with extended thinking if enabled
        prompt_loader = get_prompt_loader()
        system_prompt = prompt_loader.get_refinement_prompt(extended_thinking=self.extended_thinking, config=self.config)

        # Prepare the user content with refinement context
        user_content = prompt_loader.get_refinement_user_content(
            prompt=prompt,
            run_id=payload.get('run_id', 'unknown'),
            branch=payload.get('repo_meta', {}).get('branch', 'unknown branch'),
            commit=payload.get('repo_meta', {}).get('commit', 'unknown')[:8] if len(payload.get('repo_meta', {}).get('commit', 'unknown')) > 8 else payload.get('repo_meta', {}).get('commit', 'unknown'),  # Short commit hash
            python_version=payload.get('environment', {}).get('python', 'unknown'),
            platform=payload.get('environment', {}).get('platform', 'unknown'),
            tests_written_count=len(payload.get('tests_written', [])),
            last_run_command=' '.join(payload.get('last_run_command', [])),
            failures_total=payload.get('failures_total', 0),
            repo_meta=payload.get('repo_meta', {}),
            failure_analysis=payload.get('failure_analysis', {})
        )

        # Calculate token allocation for refinement
        failure_count = len(payload.get("failures", []))
        if failure_count <= 2:
            max_tokens = 8000  # Simple refinements
        elif failure_count <= 5:
            max_tokens = 12000  # Moderate complexity
        else:
            max_tokens = 16000  # Complex refinements

        logger.info(f"Using Bedrock for refinement with max_tokens: {max_tokens:,}")

        # Log verbose prompt information if feedback is available
        self._log_verbose_prompt(f"AWS Bedrock ({self.inference_profile})", system_prompt, user_content, 1)

        try:
            # Use langchain ChatBedrock with a system + user message structure
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ]

            # Set generation parameters including max_tokens
            generation_kwargs = {
                "max_tokens": max_tokens,
                "temperature": 0.2,  # Lower temperature for more consistent refinements
            }

            # Invoke with generation parameters
            result = self.chat.invoke(messages, **generation_kwargs)
            content = getattr(result, "content", "") or ""

            # Normalize possible structured content into text
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    else:
                        try:
                            # Support objects with attribute access
                            maybe_text = getattr(part, "text", None)
                            if maybe_text:
                                text_parts.append(maybe_text)
                        except Exception:
                            continue
                content = "\n".join(text_parts)

            # Check if we have content to parse
            if not content or not content.strip():
                logger.error("Empty response from Bedrock API for refinement")
                return json.dumps({
                    "updated_files": [],
                    "rationale": "Empty response from AI model",
                    "plan": "No refinement suggestions provided"
                })

            # Extract JSON content with truncation handling
            json_content = self._extract_json_content(content)

            try:
                refinement_result = json.loads(json_content)
                logger.info("Successfully parsed refinement response from Bedrock")

                # Validate the response structure
                if "updated_files" not in refinement_result:
                    logger.warning("Refinement response missing 'updated_files' key")
                    refinement_result["updated_files"] = []

                return json.dumps(refinement_result)

            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error in refinement response from Bedrock: {e}")
                logger.error(f"Failed content preview: {repr(json_content[:200])}")
                if "Expecting value: line 1 column 1 (char 0)" in str(e):
                    logger.error("Bedrock returned empty or non-JSON content for refinement")
                    logger.debug(f"Full response: {repr(content)}")

                # Try to recover partial results for refinement
                recovered_results = self._try_recover_partial_results(json_content, e)
                # Convert to proper refinement format if needed
                if not isinstance(recovered_results, dict) or "updated_files" not in recovered_results:
                    recovered_results = {
                        "updated_files": [],
                        "rationale": "Failed to recover refinement results from malformed JSON",
                        "plan": "JSON parsing error occurred during recovery"
                    }
                return json.dumps(recovered_results)

        except (BotoCoreError, NoCredentialsError) as e:
            raise AuthenticationError(
                f"AWS Bedrock authentication failed for refinement: {e}",
                suggestion="Ensure AWS credentials are configured (env, config/credentials files, or SSO)."
            )
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in refinement response from Bedrock: {e}")
            logger.error(f"Failed content preview: {repr(content[:200] if 'content' in locals() else 'No content')}")
            if "Expecting value: line 1 column 1 (char 0)" in str(e):
                logger.error("Bedrock returned empty or non-JSON content for refinement")
                if 'content' in locals():
                    logger.debug(f"Full response: {repr(content)}")
            return json.dumps({
                "updated_files": [],
                "rationale": "Failed to parse AI response",
                "plan": "JSON parsing error occurred"
            })
        except Exception as e:
            raise LLMClientError(
                f"Bedrock refinement request failed: {e}",
                suggestion="Verify model ID, region, and permissions for Bedrock use."
            )


class AzureOpenAIClient(LLMClient):
    """Client for interacting with Azure OpenAI."""

    def __init__(self, endpoint: str, api_key: str, deployment_name: str, extended_thinking: bool = False, 
                 cost_manager=None, feedback=None, config=None):
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_version = "2024-10-21"
        self.extended_thinking = extended_thinking
        self.cost_manager = cost_manager
        self.feedback = feedback
        self.config = config

    def generate_unit_tests(self, system_prompt: str, xml_content: str, directory_structure: str, 
                           source_files: List[str] = None, project_root: str = None) -> Dict[str, str]:
        """Generate unit tests for the provided code."""
        url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        # Extract codebase information for validation
        codebase_info = self._extract_codebase_info(source_files, project_root)
        user_content = self._build_user_content(xml_content, directory_structure, codebase_info)

        # Log verbose prompt information if feedback is available
        file_count = len(re.findall(r'<file\s+filename=', xml_content))
        self._log_verbose_prompt(f"Azure OpenAI ({self.deployment_name})", system_prompt, user_content, file_count)

        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.3,
            "max_tokens": 16000,
            "response_format": {"type": "json_object"}
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()
            content = result['choices'][0]['message']['content']

            # Parse JSON response
            tests_dict = json.loads(content)
            
            # Validate generated tests if we have codebase info
            return self._validate_and_fix_tests(tests_dict, codebase_info)

        except Exception as e:
            logger.error(f"Failed to generate tests: {e}")
            return {}

    def refine_tests(self, request: Dict) -> str:
        """Refine existing tests based on failure information using Azure OpenAI."""
        payload = request.get("payload", {})
        prompt = request.get("prompt", "")

        if not prompt or not payload:
            logger.warning("Empty refinement prompt or payload")
            return json.dumps({
                "updated_files": [],
                "rationale": "No refinement data provided",
                "plan": "Cannot proceed without proper refinement context"
            })

        url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        # Create system prompt for refinement with extended thinking if enabled
        prompt_loader = get_prompt_loader()
        system_prompt = prompt_loader.get_refinement_prompt(extended_thinking=self.extended_thinking)

        # Prepare the user content with refinement context
        user_content = prompt_loader.get_refinement_user_content(
            prompt=prompt,
            run_id=payload.get('run_id', 'unknown'),
            branch=payload.get('repo_meta', {}).get('branch', 'unknown branch'),
            commit=payload.get('repo_meta', {}).get('commit', 'unknown')[:8] if len(payload.get('repo_meta', {}).get('commit', 'unknown')) > 8 else payload.get('repo_meta', {}).get('commit', 'unknown'),  # Short commit hash
            python_version=payload.get('environment', {}).get('python', 'unknown'),
            platform=payload.get('environment', {}).get('platform', 'unknown'),
            tests_written_count=len(payload.get('tests_written', [])),
            last_run_command=' '.join(payload.get('last_run_command', [])),
            failures_total=payload.get('failures_total', 0),
            repo_meta=payload.get('repo_meta', {}),
            failure_analysis=payload.get('failure_analysis', {})
        )

        # Calculate token allocation for refinement
        failure_count = len(payload.get("failures", []))
        if failure_count <= 2:
            max_tokens = 8000  # Simple refinements
        elif failure_count <= 5:
            max_tokens = 12000  # Moderate complexity
        else:
            max_tokens = 16000  # Complex refinements

        payload_data = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.2,  # Lower temperature for more consistent refinements
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"}
        }

        # Log refinement request
        logger.info(f"Sending refinement request to Azure OpenAI using deployment: {self.deployment_name}")
        logger.debug(f"Refinement payload size: {len(json.dumps(payload_data)):,} characters")

        try:
            response = requests.post(url, headers=headers, json=payload_data)
            response.raise_for_status()

            result = response.json()
            content = result['choices'][0]['message']['content']

            # Parse JSON response
            try:
                refinement_result = json.loads(content)
                logger.info("Successfully parsed refinement response from Azure OpenAI")

                # Validate the response structure
                if "updated_files" not in refinement_result:
                    logger.warning("Refinement response missing 'updated_files' key")
                    refinement_result["updated_files"] = []

                return json.dumps(refinement_result)

            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error in refinement response from Azure OpenAI: {e}")
                logger.error(f"Failed content preview: {repr(content[:200])}")
                return json.dumps({
                    "updated_files": [],
                    "rationale": "Failed to parse AI response",
                    "plan": "JSON parsing error occurred"
                })

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None

            if status_code == 401:
                raise AuthenticationError(
                    "Invalid API key or authentication failed for Azure OpenAI refinement",
                    suggestion="Check your Azure OpenAI API key and ensure it's valid."
                )
            elif status_code == 429:
                raise LLMClientError(
                    "API rate limit exceeded for Azure OpenAI refinement",
                    status_code=status_code,
                    suggestion="Wait a moment and try again."
                )
            else:
                raise LLMClientError(
                    f"HTTP error {status_code} for Azure OpenAI refinement: {e}",
                    status_code=status_code,
                    suggestion="Check your network connection and try again."
                )

        except requests.exceptions.Timeout:
            raise LLMClientError(
                "Azure OpenAI refinement request timed out",
                suggestion="Try again or reduce the complexity of the refinement request."
            )
        except requests.exceptions.ConnectionError:
            raise LLMClientError(
                "Failed to connect to Azure OpenAI for refinement",
                suggestion="Check your internet connection and try again."
            )
        except Exception as e:
            logger.error(f"Unexpected error during Azure OpenAI refinement: {type(e).__name__}: {e}")
            raise LLMClientError(
                f"Unexpected error during Azure OpenAI refinement: {e}",
                suggestion="This appears to be a bug. Please try again or report the issue."
            )

class ClaudeAPIClient(LLMClient):
    """Client for interacting with Claude API directly."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", extended_thinking: bool = False,
                 thinking_budget: int = 4096, cost_manager=None, feedback=None, config=None):
        self.api_key = api_key
        self.model = model
        self.extended_thinking = extended_thinking
        self.thinking_budget = thinking_budget
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.cost_manager = cost_manager
        self.feedback = feedback
        self.config = config

    def generate_unit_tests(self, system_prompt: str, xml_content: str, directory_structure: str, 
                           source_files: List[str] = None, project_root: str = None) -> Dict[str, str]:
        """Generate unit tests for the provided code."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        # Extract codebase information for validation
        codebase_info = self._extract_codebase_info(source_files, project_root)
        user_content = self._build_user_content(xml_content, directory_structure, codebase_info)

        # Get system prompt with extended thinking instructions if enabled
        system_prompt = get_system_prompt(config=self.config, extended_thinking=self.extended_thinking)

        # Calculate content size
        total_content = system_prompt + user_content
        content_size = len(total_content)
        logger.info(f"Total content size: {content_size:,} characters")

        # Count files to process
        file_count = len(re.findall(r'<file\s+filename=', xml_content))
        logger.info(f"Processing {file_count} files")

        # Check if content might be too large
        # More logical token allocation: base tokens per file, scaled appropriately
        base_tokens_per_file = 4000  # Reasonable comprehensive tests per file
        
        # Calculate target output tokens based on file count
        if file_count == 1:
            # Single files can get more detailed tests
            tokens_per_file = 6000
        elif file_count <= 3:
            # Small batches get good detail
            tokens_per_file = 5000
        elif file_count <= 6:
            # Medium batches get moderate detail
            tokens_per_file = 4000
        else:
            # Large batches get focused tests
            tokens_per_file = 3000
        
        # Calculate total tokens needed, respecting model limits
        estimated_output_size = file_count * tokens_per_file
        max_tokens = min(32000, max(8000, estimated_output_size))  # Minimum 8k for any request

        logger.info(f"Setting max_tokens to {max_tokens:,} for {file_count} files ({tokens_per_file:,} tokens per file, estimated total: {estimated_output_size:,})")

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        }

        # Add extended thinking configuration if enabled
        if self.extended_thinking:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget
            }
            # Adjust max_tokens to account for thinking tokens
            payload["max_tokens"] = max(max_tokens, self.thinking_budget + 4096)
            logger.info(f"Extended thinking enabled with budget: {self.thinking_budget} tokens")

        # Log verbose prompt information if feedback is available
        self._log_verbose_prompt(self.model, system_prompt, user_content, file_count)
        
        logger.info(f"Sending request to Claude API using model: {self.model}")
        logger.debug(f"Request payload size: {len(json.dumps(payload)):,} characters")

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()

            # Check if response was truncated
            stop_reason = result.get('stop_reason', 'unknown')
            if stop_reason == 'max_tokens':
                logger.warning(f"⚠️  Response truncated at {max_tokens:,} tokens - this will likely cause JSON parsing errors")

            # Log token usage
            if 'usage' in result:
                input_tokens = result['usage'].get('input_tokens', 'N/A')
                output_tokens = result['usage'].get('output_tokens', 'N/A')
                logger.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}")
                
                # Log to cost manager if available
                if self.cost_manager and input_tokens != 'N/A' and output_tokens != 'N/A':
                    self.cost_manager.log_token_usage(self.model, input_tokens, output_tokens)

                # Warn if we're close to token limit
                if output_tokens != 'N/A' and output_tokens >= max_tokens - 100:
                    logger.warning(f"Output tokens ({output_tokens}) near limit ({max_tokens})")

            content = result['content'][0]['text']
            logger.debug(f"Response content length: {len(content):,} characters")

            # Try to extract JSON from the response
            json_content = self._extract_json_content(content)
            
            # Check if we have any content to parse
            if not json_content or not json_content.strip():
                logger.error("Empty response from Claude API")
                logger.debug(f"Original response content: {repr(content[:500])}")
                raise LLMClientError(
                    "Empty response from Claude API",
                    suggestion="The AI model returned an empty response. Try reducing context size or check your API limits."
                )

            # Parse JSON response
            try:
                tests_dict = json.loads(json_content)
                logger.info(f"Successfully parsed tests for {len(tests_dict)} files")

                # Validate we got tests for all files
                if len(tests_dict) < file_count:
                    logger.warning(f"Only received tests for {len(tests_dict)}/{file_count} files")
                    if stop_reason == 'max_tokens':
                        logger.error("Response was truncated due to token limit.")
                        logger.error(f"Try reducing batch size (current: {file_count} files) or simplify the source code.")
                        
                        # If this is a batch, suggest specific recovery
                        if file_count > 1:
                            logger.error(f"SUGGESTION: Try running with --batch-size {max(1, file_count // 2)} to reduce context size.")

                for filepath in tests_dict.keys():
                    test_size = len(tests_dict[filepath])
                    logger.debug(f"  - Generated tests for: {filepath} ({test_size:,} chars)")

                # Validate generated tests if we have codebase info
                return self._validate_and_fix_tests(tests_dict, codebase_info)

            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                logger.error(f"Failed content preview: {repr(json_content[:200])}")
                if "Expecting value: line 1 column 1 (char 0)" in str(e):
                    logger.error("The AI returned empty or non-JSON content")
                    logger.debug(f"Full original response: {repr(content)}")
                return self._try_recover_partial_results(json_content, e)

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None
            
            if status_code == 401:
                raise AuthenticationError(
                    "Invalid API key or authentication failed",
                    suggestion="Check your Claude API key and ensure it's valid."
                )
            elif status_code == 429:
                raise LLMClientError(
                    "API rate limit exceeded",
                    status_code=status_code,
                    suggestion="Wait a moment and try again, or reduce the batch size."
                )
            elif status_code == 400:
                error_details = e.response.text
                try:
                    error_json = json.loads(error_details)
                    error_msg = error_json.get('error', {}).get('message', 'Bad request')
                except:
                    error_msg = error_details[:200] + "..." if len(error_details) > 200 else error_details
                
                raise LLMClientError(
                    f"API request error: {error_msg}",
                    status_code=status_code,
                    suggestion="Check your request parameters and try again with fewer files."
                )
            else:
                raise LLMClientError(
                    f"HTTP error {status_code}: {e}",
                    status_code=status_code,
                    suggestion="Check your network connection and try again."
                )
                
        except requests.exceptions.Timeout:
            raise LLMClientError(
                "Request timed out after 120 seconds",
                suggestion="Try again with fewer files or check your network connection."
            )
        except requests.exceptions.ConnectionError:
            raise LLMClientError(
                "Failed to connect to Claude API",
                suggestion="Check your internet connection and try again."
            )
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.error(f"Failed content preview: {repr(json_content[:200])}")
            if "Expecting value: line 1 column 1 (char 0)" in str(e):
                logger.error("The AI returned empty or non-JSON content")
                # Try to provide more context about what went wrong
                if not content.strip():
                    logger.error("Response was completely empty")
                elif not json_content.strip():
                    logger.error("No JSON content found after extraction")
                    logger.debug(f"Original response: {repr(content[:500])}")
            return self._try_recover_partial_results(json_content, e)
        except Exception as e:
            logger.error(f"Unexpected error: {type(e).__name__}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise LLMClientError(
                f"Unexpected error: {e}",
                suggestion="This appears to be a bug. Please try again or report the issue."
            )

    def refine_tests(self, request: Dict) -> str:
        """Refine existing tests based on failure information using Claude API."""
        payload = request.get("payload", {})
        prompt = request.get("prompt", "")

        if not prompt or not payload:
            logger.warning("Empty refinement prompt or payload")
            return json.dumps({
                "updated_files": [],
                "rationale": "No refinement data provided",
                "plan": "Cannot proceed without proper refinement context"
            })

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        # Create system prompt for refinement with extended thinking if enabled
        prompt_loader = get_prompt_loader()
        system_prompt = prompt_loader.get_refinement_prompt(extended_thinking=self.extended_thinking)

        # Prepare the user content with refinement context
        user_content = prompt_loader.get_refinement_user_content(
            prompt=prompt,
            run_id=payload.get('run_id', 'unknown'),
            branch=payload.get('repo_meta', {}).get('branch', 'unknown branch'),
            commit=payload.get('repo_meta', {}).get('commit', 'unknown')[:8] if len(payload.get('repo_meta', {}).get('commit', 'unknown')) > 8 else payload.get('repo_meta', {}).get('commit', 'unknown'),  # Short commit hash
            python_version=payload.get('environment', {}).get('python', 'unknown'),
            platform=payload.get('environment', {}).get('platform', 'unknown'),
            tests_written_count=len(payload.get('tests_written', [])),
            last_run_command=' '.join(payload.get('last_run_command', [])),
            failures_total=payload.get('failures_total', 0),
            repo_meta=payload.get('repo_meta', {}),
            failure_analysis=payload.get('failure_analysis', {})
        )

        # Calculate token allocation for refinement
        failure_count = len(payload.get("failures", []))
        if failure_count <= 2:
            max_tokens = 8000  # Simple refinements
        elif failure_count <= 5:
            max_tokens = 12000  # Moderate complexity
        else:
            max_tokens = 16000  # Complex refinements

        payload_data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0.2,  # Lower temperature for more consistent refinements
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        }

        # Add extended thinking configuration for refinement if enabled
        if self.extended_thinking:
            payload_data["thinking"] = {
                "type": "enabled",
                "budget_tokens": min(self.thinking_budget, 8192)  # Use smaller budget for refinements
            }
            # Adjust max_tokens to account for thinking tokens
            payload_data["max_tokens"] = max(max_tokens, self.thinking_budget + 2048)

        # Log refinement request
        logger.info(f"Sending refinement request to Claude API using model: {self.model}")
        logger.debug(f"Refinement payload size: {len(json.dumps(payload_data)):,} characters")

        try:
            response = requests.post(self.api_url, headers=headers, json=payload_data, timeout=120)
            response.raise_for_status()

            result = response.json()

            # Check if response was truncated
            stop_reason = result.get('stop_reason', 'unknown')
            if stop_reason == 'max_tokens':
                logger.warning(f"⚠️  Refinement response truncated at {max_tokens:,} tokens")

            # Log token usage
            if 'usage' in result:
                input_tokens = result['usage'].get('input_tokens', 'N/A')
                output_tokens = result['usage'].get('output_tokens', 'N/A')
                logger.info(f"Refinement token usage - Input: {input_tokens}, Output: {output_tokens}")

                # Log to cost manager if available
                if self.cost_manager and input_tokens != 'N/A' and output_tokens != 'N/A':
                    self.cost_manager.log_token_usage(self.model, input_tokens, output_tokens)

            content = result['content'][0]['text']
            logger.debug(f"Refinement response content length: {len(content):,} characters")

            # Try to extract JSON from the response
            json_content = self._extract_json_content(content)

            # Check if we have any content to parse
            if not json_content or not json_content.strip():
                logger.error("Empty response from Claude API for refinement")
                logger.debug(f"Original response content: {repr(content[:500])}")
                return json.dumps({
                    "updated_files": [],
                    "rationale": "Empty response from AI model",
                    "plan": "No refinement suggestions provided"
                })

            # Parse JSON response
            try:
                refinement_result = json.loads(json_content)
                logger.info("Successfully parsed refinement response from Claude API")

                # Validate the response structure
                if "updated_files" not in refinement_result:
                    logger.warning("Refinement response missing 'updated_files' key")
                    refinement_result["updated_files"] = []

                return json.dumps(refinement_result)

            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error in refinement response: {e}")
                logger.error(f"Failed content preview: {repr(json_content[:200])}")
                if "Expecting value: line 1 column 1 (char 0)" in str(e):
                    logger.error("The AI returned empty or non-JSON content for refinement")
                    logger.debug(f"Full original response: {repr(content)}")

                # Try to recover partial results for refinement
                recovered_results = self._try_recover_partial_results(json_content, e)
                # Convert to proper refinement format if needed
                if not isinstance(recovered_results, dict) or "updated_files" not in recovered_results:
                    recovered_results = {
                        "updated_files": [],
                        "rationale": "Failed to recover refinement results from malformed JSON",
                        "plan": "JSON parsing error occurred during recovery"
                    }
                return json.dumps(recovered_results)

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None

            if status_code == 401:
                raise AuthenticationError(
                    "Invalid API key or authentication failed for refinement",
                    suggestion="Check your Claude API key and ensure it's valid."
                )
            elif status_code == 429:
                raise LLMClientError(
                    "API rate limit exceeded for refinement",
                    status_code=status_code,
                    suggestion="Wait a moment and try again."
                )
            elif status_code == 400:
                error_details = e.response.text
                try:
                    error_json = json.loads(error_details)
                    error_msg = error_json.get('error', {}).get('message', 'Bad request')
                except:
                    error_msg = error_details[:200] + "..." if len(error_details) > 200 else error_details

                raise LLMClientError(
                    f"API request error for refinement: {error_msg}",
                    status_code=status_code,
                    suggestion="Check your refinement request parameters."
                )
            else:
                raise LLMClientError(
                    f"HTTP error {status_code} for refinement: {e}",
                    status_code=status_code,
                    suggestion="Check your network connection and try again."
                )

        except requests.exceptions.Timeout:
            raise LLMClientError(
                "Refinement request timed out after 120 seconds",
                suggestion="Try again or reduce the complexity of the refinement request."
            )
        except requests.exceptions.ConnectionError:
            raise LLMClientError(
                "Failed to connect to Claude API for refinement",
                suggestion="Check your internet connection and try again."
            )
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in refinement: {e}")
            return json.dumps({
                "updated_files": [],
                "rationale": "Failed to parse AI response",
                "plan": "JSON parsing error occurred"
            })
        except Exception as e:
            logger.error(f"Unexpected error during refinement: {type(e).__name__}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise LLMClientError(
                f"Unexpected error during refinement: {e}",
                suggestion="This appears to be a bug. Please try again or report the issue."
            )

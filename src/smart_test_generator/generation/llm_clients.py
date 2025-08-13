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
    return """You are an expert Python test generation assistant. You excel at creating comprehensive, runnable unit tests that follow best practices.

<task>
Analyze Python code files provided in XML format and generate complete unit test files for each one.
</task>

<requirements>
<testing_framework>pytest</testing_framework>
<coverage_target>aim for 80%+ code coverage</coverage_target>
<test_independence>tests must run in any order without dependencies</test_independence>
</requirements>

<test_structure>
- Use meaningful test names that describe the scenario being tested
- Add clear docstrings explaining what each test verifies
- Group related tests in test classes when appropriate
- Include proper imports and fixtures
- Test happy paths, edge cases, and error conditions
- Use parametrize for testing multiple scenarios with different inputs
- Mock external dependencies appropriately
- Handle async functions with pytest-asyncio when needed
</test_structure>

<code_quality>
- Write clean, readable test code
- Use specific assertions with clear error messages
- Follow the same code style as the source project
- Structure test files to mirror source code organization
</code_quality>

<data_handling>
When creating instances of classes from the codebase:
- Check the AVAILABLE_IMPORTS section for valid class/function names
- Review CLASS_SIGNATURES for required constructor parameters
- Provide ALL required fields when instantiating dataclasses or classes
</data_handling>

<mock_configuration>
Configure mocks properly to avoid runtime errors:
- Set mock.return_value = [] for methods that should return lists
- Use mock.get.return_value = [] for config objects that return lists
- Import unittest.mock.ANY when using ANY in assertions
</mock_configuration>

<output_format>
Return a valid JSON object where:
- Keys are relative file paths matching the filepath attribute in the XML
- Values are complete test file contents as properly escaped strings
- Use \\n for newlines, \\" for quotes, \\\\ for backslashes, \\t for tabs
- Include all necessary imports and dependencies
- Return ONLY the JSON object with no additional text or markdown
</output_format>

<example_structure>
{
  "src/models/user.py": "import pytest\\nfrom unittest.mock import Mock\\n\\nclass TestUser:\\n    def test_creation(self):\\n        \\\"\\\"\\\"Test user creation with valid data.\\\"\\\"\\\"\\n        # Test implementation here\\n        pass"
}
</example_structure>"""


def get_system_prompt(config: 'Config' = None) -> str:
    """Get the system prompt for test generation."""
    # Check if we should use 2025 guidelines (default: True)
    if config and not config.get('prompt_engineering.use_2025_guidelines', True):
        return get_legacy_system_prompt()
    
    return """You are an expert Python test generator. Create comprehensive, runnable pytest tests that follow best practices.

APPROACH:
Think step-by-step before writing tests:
1. Analyze the code structure and identify testable elements
2. Determine test scenarios (happy path, edge cases, errors)
3. Plan assertions that would fail if code behavior changes
4. Write the complete test file

REQUIREMENTS:
- Framework: pytest
- Target: 80%+ coverage
- Independence: tests run in any order
- Style: match existing project patterns

GOOD TEST CHARACTERISTICS:
✓ Descriptive names: test_calculates_compound_interest_for_monthly_compounding()
✓ Specific assertions: assert result == 1050.0, not assert result > 0
✓ Edge cases: empty inputs, None values, boundary conditions
✓ Error scenarios: invalid inputs raise expected exceptions
✓ Clear arrange-act-assert structure
✓ Proper mocking of external dependencies

POOR TEST CHARACTERISTICS:
✗ Generic names: test_function()
✗ Weak assertions: assert result is not None
✗ Missing edge cases
✗ No error condition testing
✗ Tests that depend on execution order
✗ Unmocked external calls

DATA HANDLING:
- Check AVAILABLE_IMPORTS for valid imports
- Review CLASS_SIGNATURES for constructor parameters
- Provide ALL required fields for dataclasses/classes
- Mock external dependencies properly

MOCK SETUP:
Configure mocks to prevent runtime errors:
- Set mock.return_value = [] for list-returning methods
- Use mock.get.return_value = [] for config objects
- Import unittest.mock.ANY when using ANY in assertions

OUTPUT FORMAT:
Return only a valid JSON object:
{
  "relative/path/to/source.py": "import pytest\\nfrom unittest.mock import Mock\\n\\nclass TestExample:\\n    def test_specific_behavior(self):\\n        \\\"\\\"\\\"Test that X does Y when Z.\\\"\\\"\\\"\\n        # Arrange\\n        # Act\\n        # Assert\\n        pass"
}

Use proper JSON escaping: \\n for newlines, \\" for quotes, \\\\ for backslashes."""


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate_unit_tests(self, system_prompt: str, xml_content: str, directory_structure: str, 
                           source_files: List[str] = None, project_root: str = None) -> Dict[str, str]:
        """Generate unit tests for the provided code."""
        pass


class BedrockClient(LLMClient):
    """Client for interacting with AWS Bedrock using langchain-aws ChatBedrock with STS assume-role."""

    def __init__(self, *, role_arn: str, inference_profile: str, region: str = "us-east-1", cost_manager=None):
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
        codebase_info = {}
        if source_files and project_root:
            codebase_info = extract_codebase_imports(source_files, project_root)

        validation_context = ""
        if codebase_info:
            validation_context = f"""

AVAILABLE_IMPORTS (Only use these - never import non-existent classes):
{json.dumps(codebase_info.get('imports', {}), indent=2)}

CLASS_SIGNATURES (Required fields for dataclasses):
{json.dumps(codebase_info.get('classes', {}), indent=2)}"""

        user_content = f"""<directory_structure>
{directory_structure}
</directory_structure>

<code_files>
{xml_content}
</code_files>{validation_context}

Generate comprehensive unit tests for each file in the code_files section above."""

        try:
            # Use langchain ChatBedrock with a system + user message structure
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ]
            result = self.chat.invoke(messages)
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

            tests_dict = json.loads(content) if content else {}

            if codebase_info and tests_dict:
                available_imports = codebase_info.get('imports', {})
                validated_tests = {}
                for filepath, test_content in tests_dict.items():
                    is_valid, errors = validate_generated_test(test_content, filepath, available_imports, None)
                    if is_valid:
                        validated_tests[filepath] = test_content
                    else:
                        logger.warning(f"Generated test for {filepath} has validation errors: {errors}")
                        fixed_content = self._attempt_basic_fixes(test_content, errors)
                        if fixed_content:
                            is_valid_fixed, _ = validate_generated_test(fixed_content, filepath, available_imports, None)
                            if is_valid_fixed:
                                validated_tests[filepath] = fixed_content
                            else:
                                logger.error(f"Could not auto-fix test for {filepath}")
                        else:
                            logger.error(f"Skipping invalid test for {filepath}")
                return validated_tests

            return tests_dict
        except (BotoCoreError, NoCredentialsError) as e:
            raise AuthenticationError(
                f"AWS Bedrock authentication failed: {e}",
                suggestion="Ensure AWS credentials are configured (env, config/credentials files, or SSO)."
            )
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return {}
        except Exception as e:
            raise LLMClientError(
                f"Bedrock request failed: {e}",
                suggestion="Verify model ID, region, and permissions for Bedrock use."
            )

    def _attempt_basic_fixes(self, content: str, errors: List[str]) -> str:
        try:
            if "mock.get.return_value = Mock" in content:
                content = content.replace("mock.get.return_value = Mock", "mock.get.return_value = []")
            if "mock_config.get(" in content and "return_value" not in content:
                import_section = content.split("def test_")[0] if "def test_" in content else content[:500]
                if "mock_config = Mock()" in import_section and "mock_config.get.return_value" not in import_section:
                    content = content.replace(
                        "mock_config = Mock()",
                        "mock_config = Mock()\nmock_config.get.return_value = []"
                    )
            if "ANY" in content and "from unittest.mock import" in content and "ANY" not in content.split("from unittest.mock import")[1].split("\n")[0]:
                content = content.replace(
                    "from unittest.mock import",
                    "from unittest.mock import ANY,"
                )
            compile(content, "test_fix_validation", 'exec', dont_inherit=True, optimize=0)
            return content
        except:
            return None


class AzureOpenAIClient(LLMClient):
    """Client for interacting with Azure OpenAI."""

    def __init__(self, endpoint: str, api_key: str, deployment_name: str, cost_manager=None):
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_version = "2024-10-21"
        self.cost_manager = cost_manager

    def generate_unit_tests(self, system_prompt: str, xml_content: str, directory_structure: str, 
                           source_files: List[str] = None, project_root: str = None) -> Dict[str, str]:
        """Generate unit tests for the provided code."""
        url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        # Extract codebase information for validation
        codebase_info = {}
        if source_files and project_root:
            codebase_info = extract_codebase_imports(source_files, project_root)

        # Build validation context
        validation_context = ""
        if codebase_info:
            validation_context = f"""

AVAILABLE_IMPORTS (Only use these - never import non-existent classes):
{json.dumps(codebase_info.get('imports', {}), indent=2)}

CLASS_SIGNATURES (Required fields for dataclasses):
{json.dumps(codebase_info.get('classes', {}), indent=2)}"""

        user_content = f"""<directory_structure>
{directory_structure}
</directory_structure>

<code_files>
{xml_content}
</code_files>{validation_context}

Generate comprehensive unit tests for each file in the code_files section above."""

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
            if codebase_info and tests_dict:
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
            
            return tests_dict

        except Exception as e:
            logger.error(f"Failed to generate tests: {e}")
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


class ClaudeAPIClient(LLMClient):
    """Client for interacting with Claude API directly."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", cost_manager=None):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.cost_manager = cost_manager

    def generate_unit_tests(self, system_prompt: str, xml_content: str, directory_structure: str, 
                           source_files: List[str] = None, project_root: str = None) -> Dict[str, str]:
        """Generate unit tests for the provided code."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        # Extract codebase information for validation
        codebase_info = {}
        if source_files and project_root:
            codebase_info = extract_codebase_imports(source_files, project_root)

        # Calculate content size
        total_content = system_prompt + xml_content + directory_structure
        content_size = len(total_content)
        logger.info(f"Total content size: {content_size:,} characters")

        # Count files to process
        file_count = len(re.findall(r'<file\s+filename=', xml_content))
        logger.info(f"Processing {file_count} files")

        # Build validation context
        validation_context = ""
        if codebase_info:
            validation_context = f"""

AVAILABLE_IMPORTS (Only use these - never import non-existent classes):
{json.dumps(codebase_info.get('imports', {}), indent=2)}

CLASS_SIGNATURES (Required fields for dataclasses):
{json.dumps(codebase_info.get('classes', {}), indent=2)}"""

        user_content = f"""<directory_structure>
{directory_structure}
</directory_structure>

<code_files>
{xml_content}
</code_files>{validation_context}

Generate comprehensive unit tests for each file in the code_files section above."""

        # Check if content might be too large
        estimated_output_size = file_count * 3000  # Rough estimate of output size per file
        
        # Use more generous token limits to avoid truncation
        if file_count == 1:
            # For single files, use a higher limit to ensure complete generation
            max_tokens = min(32000, max(16384, estimated_output_size // 2))
        else:
            # For multiple files, be more conservative
            max_tokens = min(32000, max(8192, estimated_output_size // 4))

        logger.info(f"Setting max_tokens to {max_tokens:,} based on {file_count} files")

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

        logger.info(f"Sending request to Claude API using model: {self.model}")
        logger.debug(f"Request payload size: {len(json.dumps(payload)):,} characters")

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()

            # Check if response was truncated
            stop_reason = result.get('stop_reason', 'unknown')
            if stop_reason == 'max_tokens':
                logger.warning("Response was truncated due to max_tokens limit")

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

            # Parse JSON response
            try:
                tests_dict = json.loads(json_content)
                logger.info(f"Successfully parsed tests for {len(tests_dict)} files")

                # Validate we got tests for all files
                if len(tests_dict) < file_count:
                    logger.warning(f"Only received tests for {len(tests_dict)}/{file_count} files")
                    if stop_reason == 'max_tokens':
                        logger.error("Response was truncated. Consider processing fewer files at once.")

                for filepath in tests_dict.keys():
                    test_size = len(tests_dict[filepath])
                    logger.debug(f"  - Generated tests for: {filepath} ({test_size:,} chars)")

                # Validate generated tests if we have codebase info
                if codebase_info and tests_dict:
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

                return tests_dict

            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
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
            return self._try_recover_partial_results(json_content, e)
        except Exception as e:
            logger.error(f"Unexpected error: {type(e).__name__}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            raise LLMClientError(
                f"Unexpected error: {e}",
                suggestion="This appears to be a bug. Please try again or report the issue."
            )

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
                logger.warning(f"Truncated JSON to last valid position at character {last_valid_pos}")

        return json_content

    def _try_recover_partial_results(self, json_content: str, error: json.JSONDecodeError) -> Dict[str, str]:
        """Try to recover partial results from invalid JSON."""
        # Log where the error occurred
        error_line = json_content.count('\n', 0, error.pos) + 1 if hasattr(error, 'pos') else 0
        logger.error(f"Error at line {error_line}, position {error.pos if hasattr(error, 'pos') else 'unknown'}")

        # Try to salvage partial results
        logger.info("Attempting to parse partial JSON results...")
        try:
            # Method 1: Try to fix truncated JSON by adding closing braces
            fixed_json = json_content.rstrip()
            
            # Count open braces and try to close them
            open_braces = fixed_json.count('{') - fixed_json.count('}')
            if open_braces > 0:
                # Add missing closing braces
                fixed_json += '}' * open_braces
                
                try:
                    result = json.loads(fixed_json)
                    logger.info(f"Successfully recovered JSON by adding {open_braces} closing braces")
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

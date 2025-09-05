"""Shared helpers for LLM request/response handling.

These functions centralize prompt building, context extraction, JSON parsing,
validation, and logging so that concrete client classes only handle API I/O.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from typing import Dict, List, Optional

from smart_test_generator.utils.prompt_loader import get_prompt_loader

logger = logging.getLogger(__name__)


def get_legacy_system_prompt() -> str:
    """Get the legacy system prompt for test generation (pre-2025 guidelines)."""
    prompt_loader = get_prompt_loader()
    return prompt_loader.get_system_prompt(extended_thinking=False, use_legacy=True)


def get_system_prompt(config: 'Config' = None, extended_thinking: bool = False) -> str:
    """Get the system prompt for test generation according to config flags."""
    prompt_loader = get_prompt_loader()
    # Check if we should use 2025 guidelines (default: True)
    use_legacy = config and not config.get('prompt_engineering.use_2025_guidelines', True)
    return prompt_loader.get_system_prompt(
        extended_thinking=extended_thinking,
        use_legacy=use_legacy,
        config=config,
    )


def extract_codebase_imports(source_files: List[str], project_root: str) -> Dict[str, any]:
    """Extract available imports and class signatures from the codebase.

    Returns a mapping of importable names and class constructor signatures that can be
    used to build validation context for the LLM.
    """
    available_imports: Dict[str, str] = {}
    class_signatures: Dict[str, Dict[str, any]] = {}

    for file_path in source_files or []:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            # Get relative module path for imports
            rel_path = os.path.relpath(file_path, project_root)
            module_path = rel_path.replace(os.sep, '.').replace('.py', '')

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    full_class_name = f"{module_path}.{node.name}"

                    init_method = None
                    required_fields: List[str] = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                            init_method = item
                            break

                    # Detect dataclasses by decorator
                    is_dataclass = any(
                        (isinstance(d, ast.Name) and d.id == 'dataclass') or
                        (isinstance(d, ast.Attribute) and d.attr == 'dataclass')
                        for d in node.decorator_list
                    )

                    if is_dataclass:
                        for item in node.body:
                            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                                field_name = item.target.id
                                has_default = (item.value is not None) or (
                                    isinstance(item.value, ast.Call)
                                    and isinstance(item.value.func, ast.Name)
                                    and item.value.func.id == 'field'
                                )
                                if not has_default:
                                    required_fields.append(field_name)
                    elif init_method:
                        for arg in init_method.args.args[1:]:  # Skip 'self'
                            defaults_offset = len(init_method.args.args) - len(init_method.args.defaults or [])
                            arg_index = init_method.args.args.index(arg)
                            has_default = arg_index >= defaults_offset
                            if not has_default:
                                required_fields.append(arg.arg)

                    class_signatures[node.name] = {
                        'full_name': full_class_name,
                        'is_dataclass': is_dataclass,
                        'required_fields': required_fields,
                        'module': module_path,
                    }
                    available_imports[node.name] = module_path

                elif isinstance(node, ast.FunctionDef):
                    available_imports[node.name] = module_path

        except Exception as e:
            logger.debug(f"Failed to parse {file_path}: {e}")
            continue

    return {
        'imports': available_imports,
        'classes': class_signatures,
    }


def build_validation_context(codebase_info: Dict) -> str:
    """Build validation context string from codebase info."""
    if not codebase_info:
        return ""
    prompt_loader = get_prompt_loader()
    return prompt_loader.get_validation_context(
        imports=codebase_info.get('imports', {}),
        classes=codebase_info.get('classes', {}),
    )


def build_user_content(
    *, xml_content: str, directory_structure: str, codebase_info: Dict
) -> str:
    """Build user content for LLM request combining directory, code XML and validation."""
    validation_context = build_validation_context(codebase_info)
    prompt_loader = get_prompt_loader()
    return prompt_loader.get_user_content_template(
        directory_structure=directory_structure,
        xml_content=xml_content,
        validation_context=validation_context,
    )


def log_verbose_prompt(
    *, feedback, model_name: str, system_prompt: str, user_content: str, file_count: int
) -> None:
    """Log verbose prompt information via user feedback, if available."""
    if feedback:
        content_size = len((system_prompt or "") + (user_content or ""))
        try:
            feedback.verbose_prompt_display(
                model_name=model_name,
                system_prompt=system_prompt,
                user_content=user_content,
                content_size=content_size,
                file_count=file_count,
            )
        except Exception:
            # Avoid failing if feedback interface changes
            logger.debug("verbose_prompt_display not available on feedback")


def validate_generated_test_safe(
    test_content: str, filepath: str, available_imports: Dict[str, str], config: 'Config' = None
) -> tuple[bool, List[str]]:
    """Validate generated test content with enhanced security measures."""
    errors: List[str] = []

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
            r'open\s*\(["\']^["\']*["\'],\s*["\'][wa]',
            r'subprocess',
            r'os\.system',
            r'os\.popen',
            r'__builtins__',
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, test_content, re.IGNORECASE):
                errors.append(
                    f"Security: Potentially dangerous pattern '{pattern}' found in {filepath}"
                )
                return False, errors

    # Security check 2: Limit content size
    if len(test_content) > max_file_size:
        errors.append(
            f"Security: Generated test file too large ({len(test_content)} chars) in {filepath}"
        )
        return False, errors

    # Security check 3: Basic syntax validation without full AST parsing
    try:
        compile(test_content, filepath, 'exec', dont_inherit=True, optimize=0)
    except SyntaxError as e:
        errors.append(f"Syntax error in {filepath}: {e}")
        return False, errors
    except Exception as e:
        errors.append(f"Validation error in {filepath}: {e}")
        return False, errors

    return len(errors) == 0, errors


def validate_generated_test(
    test_content: str, filepath: str, available_imports: Dict[str, str], config: 'Config' = None
) -> tuple[bool, List[str]]:
    """Validate generated test content for common issues (safe version)."""
    return validate_generated_test_safe(test_content, filepath, available_imports, config)


def normalize_generated_test_content(content: str) -> str:
    """Best-effort normalization of model-generated test code strings."""
    try:
        if not content:
            return content
        text = content.strip()
        # Strip code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            start_idx = None
            end_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith("```"):
                    if start_idx is None:
                        start_idx = i + 1
                    else:
                        end_idx = i
                        break
            if start_idx is not None and end_idx is not None and end_idx > start_idx:
                text = "\n".join(lines[start_idx:end_idx])

        looks_double_escaped = ("\\n" in text and "\n" not in text) or ("\\t" in text) or ("\\\"" in text)
        if looks_double_escaped:
            try:
                text = json.loads(f'"{text}"')
            except Exception:
                try:
                    text = bytes(text, "utf-8").decode("unicode_escape")
                except Exception:
                    pass
        return text
    except Exception:
        return content


def attempt_basic_fixes(content: str, errors: List[str]) -> Optional[str]:
    """Attempt to fix basic syntax and mock configuration issues."""
    try:
        if "mock.get.return_value = Mock" in content:
            content = content.replace("mock.get.return_value = Mock", "mock.get.return_value = []")
        if "mock_config.get(" in content and "return_value" not in content:
            import_section = content.split("def test_")[0] if "def test_" in content else content[:500]
            if "mock_config = Mock()" in import_section and "mock_config.get.return_value" not in import_section:
                content = content.replace(
                    "mock_config = Mock()",
                    "mock_config = Mock()\nmock_config.get.return_value = []",
                )
        if "ANY" in content and "from unittest.mock import" in content and "ANY" not in content.split("from unittest.mock import")[1].split("\n")[0]:
            content = content.replace(
                "from unittest.mock import",
                "from unittest.mock import ANY,",
            )
        compile(content, "test_fix_validation", 'exec', dont_inherit=True, optimize=0)
        return content
    except Exception:
        return None


def validate_and_fix_tests(tests_dict: Dict[str, str], codebase_info: Dict) -> Dict[str, str]:
    """Validate generated tests and attempt to fix basic issues."""
    if not codebase_info or not tests_dict:
        return tests_dict
    available_imports = codebase_info.get('imports', {})
    validated_tests: Dict[str, str] = {}
    for filepath, test_content in tests_dict.items():
        normalized_content = normalize_generated_test_content(test_content)
        is_valid, errs = validate_generated_test(normalized_content, filepath, available_imports, None)
        if is_valid:
            validated_tests[filepath] = normalized_content
        else:
            logger.warning(f"Generated test for {filepath} has validation errors: {errs}")
            fixed_content = attempt_basic_fixes(normalized_content, errs)
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


def extract_json_content(content: str) -> str:
    """Extract JSON content from a possibly-wrapped response string."""
    json_content = (content or "").strip()
    if json_content.startswith("```"):
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
    json_content = json_content.strip()
    if not json_content.startswith("{"):
        start = json_content.find("{")
        if start >= 0:
            json_content = json_content[start:]
    if not json_content.endswith("}"):
        brace_count = 0
        last_valid_pos = -1
        for i, ch in enumerate(json_content):
            if ch == '{':
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0:
                    last_valid_pos = i + 1
        if last_valid_pos > 0:
            json_content = json_content[:last_valid_pos]
            logger.warning(
                f"JSON appears truncated - found complete JSON up to position {last_valid_pos}"
            )
        else:
            logger.warning("JSON appears severely truncated - no complete JSON object found")
    return json_content


def try_recover_partial_results(json_content: str, error: json.JSONDecodeError) -> Dict[str, str]:
    """Try to recover partial results from invalid JSON content."""
    error_line = json_content.count('\n', 0, getattr(error, 'pos', 0)) + 1 if hasattr(error, 'pos') else 0
    logger.error(
        f"JSON parse error at line {error_line}, position {getattr(error, 'pos', 'unknown')}"
    )
    if json_content.rstrip().endswith('",') or json_content.rstrip().endswith('"'):
        logger.warning("Response appears to be truncated (ends abruptly)")
    elif json_content.count('{') > json_content.count('}'):
        logger.warning("Response appears to be truncated (unmatched opening braces)")

    # Strategy 1: try to close braces
    try:
        fixed_json = json_content.rstrip()
        if fixed_json.endswith('",'):
            fixed_json = fixed_json[:-1]
        elif fixed_json.endswith(','):
            fixed_json = fixed_json[:-1]
        open_braces = fixed_json.count('{') - fixed_json.count('}')
        if open_braces > 0:
            fixed_json += '}' * open_braces
            try:
                result = json.loads(fixed_json)
                logger.warning(
                    f"Recovered truncated JSON by adding {open_braces} closing braces"
                )
                return result
            except json.JSONDecodeError:
                pass
    except Exception:
        pass

    # Strategy 2: extract individual entries
    partial_results: Dict[str, str] = {}
    pattern = r'"([^"]+\.py)"\s*:\s*"((?:[^"\\]|\\.)*)"'
    try:
        matches = re.findall(pattern, json_content, re.DOTALL)
        for filepath, content in matches:
            try:
                unescaped_content = json.loads(f'"{content}"')
                partial_results[filepath] = unescaped_content
            except Exception:
                continue
        if partial_results:
            logger.warning(f"Partially recovered tests for {len(partial_results)} files")
            return partial_results
    except Exception:
        pass

    # Strategy 3: best-effort largest valid substring
    best_result: Dict[str, str] = {}
    best_length = 0
    for end_pos in range(len(json_content), 0, -100):
        try:
            test_json = json_content[:end_pos]
            open_braces = test_json.count('{') - test_json.count('}')
            if open_braces > 0:
                test_json += '}' * open_braces
            result = json.loads(test_json)
            rlen = len(str(result))
            if rlen > best_length:
                best_result = result
                best_length = rlen
        except json.JSONDecodeError:
            continue
    if best_result:
        logger.info(f"Recovered largest valid JSON subset with {len(best_result)} entries")
        return best_result
    return {}



"""Test coverage analysis."""

import os
import sys
import json
import subprocess
import ast
import logging
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Union

from smart_test_generator.models.data_models import TestCoverage
from smart_test_generator.config import Config

logger = logging.getLogger(__name__)


class ASTCoverageAnalyzer:
    """AST-based coverage analysis for fallback when pytest fails."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def analyze_file_coverage(self, filepath: str, test_files: List[str] = None) -> TestCoverage:
        """Analyze coverage of a single file using AST analysis."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content, filename=filepath)
            
            # Get all executable lines
            executable_lines = self._get_executable_lines(tree)
            
            # Get all functions and classes
            all_functions = self._get_all_functions_and_methods(tree)
            
            # Estimate coverage based on existing tests
            covered_lines, covered_functions = self._estimate_coverage(
                filepath, executable_lines, all_functions, test_files or []
            )
            
            # Calculate line coverage percentage
            total_executable = len(executable_lines)
            covered_count = len(covered_lines)
            line_coverage = (covered_count / total_executable * 100) if total_executable > 0 else 0
            
            # Determine uncovered functions
            uncovered_functions = all_functions - covered_functions
            
            # Get missing lines
            missing_lines = list(executable_lines - covered_lines)
            missing_lines.sort()
            
            return TestCoverage(
                filepath=filepath,
                line_coverage=line_coverage,
                branch_coverage=self._estimate_branch_coverage(tree, covered_lines),
                missing_lines=missing_lines,
                covered_functions=covered_functions,
                uncovered_functions=uncovered_functions
            )
            
        except Exception as e:
            logger.error(f"AST coverage analysis failed for {filepath}: {e}")
            return self._create_zero_coverage(filepath)
    
    def _get_executable_lines(self, tree: ast.AST) -> Set[int]:
        """Extract all executable line numbers from AST."""
        executable_lines = set()
        
        for node in ast.walk(tree):
            # Skip certain non-executable nodes
            if isinstance(node, (ast.Module, ast.Import, ast.ImportFrom)):
                continue
            
            # Include statements, expressions, and definitions
            if hasattr(node, 'lineno'):
                executable_lines.add(node.lineno)
                
                # For compound statements, include all lines
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    for line in range(node.lineno, node.end_lineno + 1):
                        executable_lines.add(line)
        
        return executable_lines
    
    def _get_all_functions_and_methods(self, tree: ast.AST) -> Set[str]:
        """Extract all function and method names from AST."""
        functions = set()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if it's a method (inside a class)
                parent = getattr(node, 'parent_class', None)
                if parent:
                    functions.add(f"{parent}.{node.name}")
                else:
                    functions.add(node.name)
        
        # Add parent class information to methods
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        child.parent_class = node.name
                        functions.add(f"{node.name}.{child.name}")
        
        return functions
    
    def _estimate_coverage(self, filepath: str, executable_lines: Set[int], 
                          all_functions: Set[str], test_files: List[str]) -> Tuple[Set[int], Set[str]]:
        """Estimate coverage based on existing test files."""
        covered_lines = set()
        covered_functions = set()
        
        if not test_files:
            # No tests exist - assume nothing is covered
            return covered_lines, covered_functions
        
        # Look for test files that might test this source file
        source_name = Path(filepath).stem
        relevant_test_files = []
        
        for test_file in test_files:
            test_name = Path(test_file).stem
            if (source_name in test_name or 
                test_name.replace('test_', '') == source_name or
                test_name.replace('_test', '') == source_name):
                relevant_test_files.append(test_file)
        
        if not relevant_test_files:
            # No relevant test files found, but tests exist for other modules
            # Be conservative - assume minimal coverage to avoid regeneration
            if test_files:
                # If any test files exist in the project, assume some basic coverage
                # to avoid unnecessary regeneration
                if all_functions:
                    # Assume 20% of functions might be covered by integration tests
                    estimated_covered = list(all_functions)[:max(1, len(all_functions) // 5)]
                    covered_functions.update(estimated_covered)
                    covered_lines = self._estimate_lines_for_functions(filepath, covered_functions)
            return covered_lines, covered_functions
        
        # Analyze test files to see what functions they might test
        tested_functions = self._analyze_test_files(relevant_test_files, all_functions)
        
        if tested_functions:
            # If functions are tested, estimate line coverage
            covered_functions = tested_functions
            covered_lines = self._estimate_lines_for_functions(filepath, tested_functions)
        else:
            # Relevant test files exist but we can't detect specific tested functions
            # Be more conservative to avoid regeneration - assume moderate coverage
            if all_functions:
                # Assume 40% coverage when we have relevant test files but can't parse them well
                estimated_count = max(1, len(all_functions) * 2 // 5)  # 40% of functions
                estimated_covered = list(all_functions)[:estimated_count]
                covered_functions.update(estimated_covered)
                covered_lines = self._estimate_lines_for_functions(filepath, covered_functions)
        
        return covered_lines, covered_functions
    
    def _analyze_test_files(self, test_files: List[str], source_functions: Set[str]) -> Set[str]:
        """Analyze test files to identify which source functions are tested."""
        tested_functions = set()
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content, filename=test_file)
                
                # Look for function calls and imports that might indicate testing
                for node in ast.walk(tree):
                    # Check function calls in test methods
                    if isinstance(node, ast.Call):
                        func_name = self._get_function_name_from_call(node)
                        
                        # Handle direct function calls
                        if func_name in source_functions:
                            tested_functions.add(func_name)
                        
                        # Handle module.function calls (e.g., source.tested_function)
                        elif func_name and '.' in func_name:
                            _, function_part = func_name.rsplit('.', 1)
                            if function_part in source_functions:
                                tested_functions.add(function_part)
                    
                    # Check test method names that might indicate what they test
                    elif isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                        # Extract potential function name from test name
                        test_target = node.name.replace('test_', '')
                        for func_name in source_functions:
                            if test_target in func_name.lower() or func_name.lower() in test_target:
                                tested_functions.add(func_name)
            
            except Exception as e:
                logger.warning(f"Could not analyze test file {test_file}: {e}")
        
        return tested_functions
    
    def _get_function_name_from_call(self, call_node: ast.Call) -> Optional[str]:
        """Extract function name from a call node."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            if isinstance(call_node.func.value, ast.Name):
                return f"{call_node.func.value.id}.{call_node.func.attr}"
        return None
    
    def _estimate_lines_for_functions(self, filepath: str, tested_functions: Set[str]) -> Set[int]:
        """Estimate which lines are covered based on tested functions."""
        covered_lines = set()
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content, filename=filepath)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_name = node.name
                    
                    # Check if this function is tested
                    is_tested = any(func_name in tested_func for tested_func in tested_functions)
                    
                    if is_tested:
                        # Add all lines in this function
                        start_line = node.lineno
                        end_line = getattr(node, 'end_lineno', start_line)
                        for line in range(start_line, end_line + 1):
                            covered_lines.add(line)
                        
                        # Estimate partial coverage for complex functions
                        complexity = self._calculate_function_complexity(node)
                        if complexity > 3:
                            # For complex functions, assume only 70% coverage
                            total_lines = list(range(start_line, end_line + 1))
                            coverage_count = int(len(total_lines) * 0.7)
                            covered_lines.update(total_lines[:coverage_count])
                        
        except Exception as e:
            logger.warning(f"Could not estimate line coverage for {filepath}: {e}")
        
        return covered_lines
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler, 
                                 ast.With, ast.AsyncWith, ast.Try)):
                complexity += 1
        return complexity
    
    def _estimate_branch_coverage(self, tree: ast.AST, covered_lines: Set[int]) -> float:
        """Estimate branch coverage based on covered lines."""
        total_branches = 0
        covered_branches = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                total_branches += 2  # Each conditional has at least 2 branches
                if hasattr(node, 'lineno') and node.lineno in covered_lines:
                    covered_branches += 1  # Assume at least one branch is covered
        
        return (covered_branches / total_branches * 100) if total_branches > 0 else 0
    
    def _create_zero_coverage(self, filepath: str) -> TestCoverage:
        """Create zero coverage data for a file."""
        try:
            all_functions = set()
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=filepath)
                all_functions = self._get_all_functions_and_methods(tree)
        except Exception:
            pass
        
        return TestCoverage(
            filepath=filepath,
            line_coverage=0.0,
            branch_coverage=0.0,
            missing_lines=[],
            covered_functions=set(),
            uncovered_functions=all_functions
        )


class CoverageAnalyzer:
    """Analyze existing test coverage to identify gaps."""

    def __init__(self, project_root: Path, config: Config):
        self.project_root = project_root
        self.config = config
        self.coverage_data = {}
        self.ast_analyzer = ASTCoverageAnalyzer(project_root)

    def run_coverage_analysis(self, source_files: List[str], test_files: List[str] = None) -> Dict[str, TestCoverage]:
        """Run pytest with coverage to get detailed metrics, fallback to AST analysis."""
        try:
            # Preflight checks for environment robustness
            preflight = self.preflight_check()
            if preflight.get("status") != "ok":
                # Log guidance but continue (we will likely fall back to AST)
                for msg in preflight.get("messages", []):
                    logger.warning(msg)
                # If pytest is not importable, raise to trigger AST fallback immediately
                if not preflight.get("pytest_importable", False):
                    raise Exception("pytest not importable in current interpreter")

            # Try to run pytest with coverage first
            logger.info("Attempting pytest coverage analysis...")
            coverage_map = self._run_pytest_coverage(source_files)
            
            # Check if we got meaningful results
            has_real_coverage = any(
                cov.line_coverage > 0 or cov.covered_functions 
                for cov in coverage_map.values()
            )
            
            if has_real_coverage:
                logger.info("Pytest coverage analysis successful")
                return coverage_map
            else:
                logger.info("Pytest coverage analysis returned empty results, falling back to AST analysis")
                return self._run_ast_coverage_fallback(source_files, test_files)
                
        except Exception as e:
            logger.warning(f"Pytest coverage analysis failed: {e}")
            logger.info("Falling back to AST-based coverage analysis")
            return self._run_ast_coverage_fallback(source_files, test_files)

    def _run_pytest_coverage(self, source_files: List[str]) -> Dict[str, TestCoverage]:
        """Run pytest with coverage analysis."""
        # Build command and environment from configuration
        from smart_test_generator.analysis.coverage.command_builder import build_pytest_command
        from smart_test_generator.analysis.coverage.runner import run_pytest
        from smart_test_generator.analysis.coverage.errors import parse_errors

        spec = build_pytest_command(project_root=self.project_root, config=self.config, preflight_result=None)

        run_result = run_pytest(spec)

        # Check if .coverage file was created
        coverage_file = self.project_root / ".coverage"
        if coverage_file.exists():
            return self._parse_coverage_file(coverage_file, source_files)
        else:
            logger.warning("No .coverage file created - tests may not exist yet or pytest failed")
            # Provide actionable guidance using error parser
            report = parse_errors(run_result, self.preflight_check())
            for line in report.excerpts[-20:]:  # log tail for context
                logger.debug(line)
            guidance_text = "; ".join(report.guidance) if report.guidance else "See pytest output for details"
            from smart_test_generator.exceptions import CoverageAnalysisError
            exit_code = 2 if report.kind == "env" else (124 if report.kind == "timeout" else 1)
            raise CoverageAnalysisError(
                f"No coverage file generated ({report.kind}). {guidance_text}",
                kind=report.kind,
                exit_code=exit_code,
            )

    

    def preflight_check(self) -> Dict[str, Union[str, bool, List[str]]]:
        """Check environment readiness to run pytest coverage.

        Returns a dict containing:
        - status: 'ok' | 'warn'
        - venv_active: bool
        - pytest_importable: bool
        - coverage_importable: bool
        - messages: List[str] guidance messages
        """
        messages: List[str] = []

        # Detect virtualenv activation
        venv_active = bool(os.environ.get("VIRTUAL_ENV")) or (getattr(sys, 'base_prefix', sys.prefix) != sys.prefix)

        # Try importing pytest/coverage in current interpreter
        pytest_importable = True
        coverage_importable = True
        try:
            import pytest  # type: ignore  # noqa: F401
        except Exception:
            pytest_importable = False
            messages.append(
                f"pytest is not importable in interpreter '{sys.executable}'. Install with 'pip install pytest pytest-cov' in the active environment."
            )

        try:
            import coverage as _cov  # type: ignore  # noqa: F401
        except Exception:
            coverage_importable = False
            messages.append(
                f"coverage.py is not importable in interpreter '{sys.executable}'. Install with 'pip install coverage pytest-cov'."
            )

        # Suggest activation commands if a local venv exists but not active
        if not venv_active:
            candidate = None
            for name in ('.venv', 'venv', '.env', 'env'):
                path = (self.project_root / name)
                if path.exists():
                    candidate = path
                    break
            if candidate:
                if platform.system().lower().startswith('win'):
                    activate_cmd = f"{candidate}\\Scripts\\activate"
                else:
                    activate_cmd = f"source {candidate}/bin/activate"
                messages.append(
                    f"A virtual environment was detected at '{candidate}', but it is not active. Activate it with: {activate_cmd}"
                )
            else:
                messages.append(
                    "No active virtual environment detected. Create one with 'python -m venv .venv' and activate it, then install dependencies."
                )

        status = "ok" if (pytest_importable and coverage_importable) else "warn"
        return {
            "status": status,
            "venv_active": venv_active,
            "pytest_importable": pytest_importable,
            "coverage_importable": coverage_importable,
            "messages": messages,
        }

    def _run_ast_coverage_fallback(self, source_files: List[str], test_files: List[str] = None) -> Dict[str, TestCoverage]:
        """Run AST-based coverage analysis as fallback."""
        logger.info("Running AST-based coverage analysis")
        coverage_map = {}
        
        # Find test files if not provided
        if test_files is None:
            test_files = self._find_test_files()
        
        for filepath in source_files:
            try:
                coverage_map[filepath] = self.ast_analyzer.analyze_file_coverage(filepath, test_files)
                logger.debug(f"AST analysis completed for {filepath}")
            except Exception as e:
                logger.error(f"AST coverage analysis failed for {filepath}: {e}")
                coverage_map[filepath] = self._create_empty_coverage_for_file(filepath)
        
        return coverage_map

    def _find_test_files(self) -> List[str]:
        """Find all test files in the project."""
        from smart_test_generator.utils.file_utils import FileUtils
        
        test_patterns = ['test_*.py', '*_test.py']
        test_files = FileUtils.find_files_by_pattern(self.project_root, test_patterns)
        
        return [str(f) for f in test_files if f.is_file()]

    def _parse_coverage_file(self, coverage_file: Path, source_files: List[str]) -> Dict[str, TestCoverage]:
        """Parse .coverage file to extract detailed metrics."""
        try:
            import coverage
            
            # Load coverage data
            cov = coverage.Coverage(data_file=str(coverage_file))
            cov.load()
            
            coverage_map = {}
            
            for filepath in source_files:
                try:
                    # Get analysis for this file
                    analysis = cov.analysis(filepath)
                    if analysis:
                        filename, executed_lines, missing_lines, missing_lines_formatted = analysis
                        
                        # Calculate coverage percentage
                        total_lines = len(executed_lines) + len(missing_lines)
                        line_coverage = (len(executed_lines) / total_lines * 100) if total_lines > 0 else 0
                        
                        # Extract covered and uncovered functions
                        covered_funcs, uncovered_funcs = self._analyze_function_coverage(
                            filepath, executed_lines, missing_lines
                        )
                        
                        coverage_map[filepath] = TestCoverage(
                            filepath=filepath,
                            line_coverage=line_coverage,
                            branch_coverage=0,  # Branch coverage requires special setup
                            missing_lines=list(missing_lines),
                            covered_functions=covered_funcs,
                            uncovered_functions=uncovered_funcs
                        )
                    else:
                        # File not in coverage
                        coverage_map[filepath] = self._create_empty_coverage_for_file(filepath)
                        
                except Exception:
                    # File not in coverage or error analyzing
                    coverage_map[filepath] = self._create_empty_coverage_for_file(filepath)
            
            return coverage_map
            
        except ImportError:
            logger.warning("Coverage library not available")
            raise Exception("Coverage library not available")
        except Exception as e:
            logger.error(f"Failed to parse coverage file: {e}")
            raise

    def _create_empty_coverage(self, source_files: List[str]) -> Dict[str, TestCoverage]:
        """Create empty coverage data for files with no tests."""
        coverage_map = {}
        for filepath in source_files:
            coverage_map[filepath] = self._create_empty_coverage_for_file(filepath)
        return coverage_map

    def _create_empty_coverage_for_file(self, filepath: str) -> TestCoverage:
        """Create empty coverage data for a single file."""
        all_funcs = self._get_all_functions(filepath)
        return TestCoverage(
            filepath=filepath,
            line_coverage=0,
            branch_coverage=0,
            missing_lines=[],
            covered_functions=set(),
            uncovered_functions=all_funcs
        )

    def _analyze_function_coverage(self, filepath: str, executed_lines: List[int],
                                   missing_lines: List[int]) -> Tuple[Set[str], Set[str]]:
        """Determine which functions are covered based on line coverage."""
        covered = set()
        uncovered = set()

        try:
            with open(filepath, 'r') as f:
                tree = ast.parse(f.read(), filename=filepath)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Get function line range - handle cases where end_lineno is not available
                    start_line = node.lineno
                    end_line = getattr(node, 'end_lineno', None)
                    
                    if end_line is None:
                        # Calculate end line by examining the body
                        end_line = start_line
                        for child in ast.walk(node):
                            if hasattr(child, 'lineno') and child.lineno > end_line:
                                end_line = child.lineno
                    
                    func_lines = set(range(start_line, end_line + 1))

                    if func_lines.intersection(executed_lines):
                        covered.add(node.name)
                    elif func_lines.intersection(missing_lines):
                        uncovered.add(node.name)
                    else:
                        # If no intersection with either, check if function line is executed
                        if start_line in executed_lines:
                            covered.add(node.name)
                        elif start_line in missing_lines:
                            uncovered.add(node.name)

        except Exception as e:
            logger.error(f"Failed to analyze function coverage for {filepath}: {e}")

        return covered, uncovered

    def _get_all_functions(self, filepath: str) -> Set[str]:
        """Get all function names from a file."""
        functions = set()
        try:
            with open(filepath, 'r') as f:
                tree = ast.parse(f.read(), filename=filepath)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions.add(node.name)

        except Exception as e:
            logger.error(f"Failed to get functions from {filepath}: {e}")

        return functions

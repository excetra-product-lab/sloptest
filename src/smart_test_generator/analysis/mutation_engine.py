"""Mutation testing engine for evaluating test effectiveness."""

import ast
import re
import os
import sys
import subprocess
import shutil
import tempfile
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from dataclasses import replace

from smart_test_generator.models.data_models import (
    MutationType, Mutant, MutationResult, MutationScore, WeakSpot
)

logger = logging.getLogger(__name__)


class SafeFS:
    """Guarded file operations scoped to a sandbox root.

    Denies any write/rename/remove operations that attempt to touch paths outside
    of the provided sandbox root directory.
    """

    def __init__(self, sandbox_root: Path):
        self.sandbox_root = sandbox_root.resolve()

    def _ensure_within(self, path: Path) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(self.sandbox_root)
        except ValueError:
            raise ValueError(f"Path escapes sandbox: {resolved} not under {self.sandbox_root}")
        return resolved

    def write_text(self, path: Path, content: str, encoding: str = 'utf-8') -> None:
        target = self._ensure_within(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding=encoding)

    def remove(self, path: Path) -> None:
        target = self._ensure_within(path)
        if target.exists():
            target.unlink()

    def rename(self, src: Path, dst: Path) -> None:
        s = self._ensure_within(src)
        d = self._ensure_within(dst)
        os.rename(str(s), str(d))

    def replace(self, src: Path, dst: Path) -> None:
        """Atomic replace of dst with src (POSIX semantics)."""
        s = self._ensure_within(src)
        d = self._ensure_within(dst)
        os.replace(str(s), str(d))


class MutationOperator(ABC):
    """Abstract base class for mutation operators.

    Only `get_mutation_type` is abstract to keep the class abstract while allowing
    subclasses that omit other methods to be instantiated for tests that expect
    NotImplementedError at call-time rather than at instantiation-time.
    """
    
    @abstractmethod
    def get_mutation_type(self) -> MutationType:
        """Get the type of mutation this operator performs."""
        raise NotImplementedError
    
    def generate_mutants(self, source_code: str, filepath: str) -> List[Mutant]:
        """Generate mutants for the given source code."""
        raise NotImplementedError
    
    def is_applicable(self, node: Any) -> bool:
        """Check if this operator can be applied to the given AST node."""
        raise NotImplementedError


class ArithmeticOperatorMutator(MutationOperator):
    """Mutate arithmetic operators (+, -, *, /, //, %, **)."""
    
    def __init__(self):
        self.operator_mappings = {
            '+': ['-', '*', '/', '//', '%'],
            '-': ['+', '*', '/', '//', '%'],
            '*': ['+', '-', '/', '//', '%', '**'],
            '/': ['+', '-', '*', '//', '%'],
            '//': ['+', '-', '*', '/', '%'],
            '%': ['+', '-', '*', '/', '//'],
            '**': ['*', '+', '-']
        }
        
        # AST node type mappings for Python
        self.ast_operators = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.FloorDiv: '//',
            ast.Mod: '%',
            ast.Pow: '**'
        }
    
    def get_mutation_type(self) -> MutationType:
        return MutationType.ARITHMETIC_OPERATOR
    
    def is_applicable(self, node: Any) -> bool:
        return isinstance(node, ast.BinOp) and type(node.op) in self.ast_operators
    
    def generate_mutants(self, source_code: str, filepath: str) -> List[Mutant]:
        """Generate arithmetic operator mutants."""
        mutants = []
        
        try:
            tree = ast.parse(source_code)
            lines = source_code.split('\n')
            
            for node in ast.walk(tree):
                if self.is_applicable(node):
                    original_op = self.ast_operators[type(node.op)]
                    
                    for replacement_op in self.operator_mappings.get(original_op, []):
                        mutant_id = f"AOR_{node.lineno}_{node.col_offset}_{replacement_op}"
                        
                        # Create mutant
                        mutant = Mutant(
                            id=mutant_id,
                            mutation_type=self.get_mutation_type(),
                            original_code=original_op,
                            mutated_code=replacement_op,
                            line_number=node.lineno,
                            column_start=node.col_offset,
                            column_end=node.col_offset + len(original_op),
                            description=f"Replace '{original_op}' with '{replacement_op}' at line {node.lineno}",
                            language="python",
                            ast_node_type=type(node.op).__name__,
                            context={
                                'function_context': self._get_function_context(node, tree),
                                'surrounding_code': lines[max(0, node.lineno-2):node.lineno+1] if node.lineno <= len(lines) else []
                            }
                        )
                        mutants.append(mutant)
                        
        except SyntaxError as e:
            logger.warning(f"Syntax error in {filepath}: {e}")
        except Exception as e:
            logger.error(f"Error generating arithmetic mutants for {filepath}: {e}")
        
        return mutants
    
    def _get_function_context(self, target_node: ast.AST, tree: ast.AST) -> Optional[str]:
        """Get the function name containing the target node."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if (hasattr(target_node, 'lineno') and hasattr(node, 'lineno') and
                    node.lineno <= target_node.lineno <= 
                    (node.end_lineno if hasattr(node, 'end_lineno') else node.lineno + 10)):
                    return node.name
        return None


class ComparisonOperatorMutator(MutationOperator):
    """Mutate comparison operators (==, !=, <, >, <=, >=)."""
    
    def __init__(self):
        self.operator_mappings = {
            '==': ['!=', '<', '>', '<=', '>='],
            '!=': ['==', '<', '>', '<=', '>='],
            '<': ['<=', '>', '>=', '==', '!='],
            '>': ['>=', '<', '<=', '==', '!='],
            '<=': ['<', '>=', '>', '==', '!='],
            '>=': ['>', '<=', '<', '==', '!=']
        }
        
        self.ast_operators = {
            ast.Eq: '==',
            ast.NotEq: '!=',
            ast.Lt: '<',
            ast.Gt: '>',
            ast.LtE: '<=',
            ast.GtE: '>='
        }
    
    def get_mutation_type(self) -> MutationType:
        return MutationType.COMPARISON_OPERATOR
    
    def is_applicable(self, node: Any) -> bool:
        return isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in self.ast_operators
    
    def generate_mutants(self, source_code: str, filepath: str) -> List[Mutant]:
        """Generate comparison operator mutants."""
        mutants = []
        
        try:
            tree = ast.parse(source_code)
            lines = source_code.split('\n')
            
            for node in ast.walk(tree):
                if self.is_applicable(node):
                    original_op = self.ast_operators[type(node.ops[0])]
                    
                    for replacement_op in self.operator_mappings.get(original_op, []):
                        mutant_id = f"COR_{node.lineno}_{node.col_offset}_{replacement_op.replace('<', 'LT').replace('>', 'GT').replace('=', 'EQ')}"
                        
                        mutant = Mutant(
                            id=mutant_id,
                            mutation_type=self.get_mutation_type(),
                            original_code=original_op,
                            mutated_code=replacement_op,
                            line_number=node.lineno,
                            column_start=node.col_offset,
                            column_end=node.col_offset + len(original_op),
                            description=f"Replace '{original_op}' with '{replacement_op}' at line {node.lineno}",
                            severity="high" if original_op in ['==', '!='] else "medium",
                            language="python",
                            ast_node_type=type(node.ops[0]).__name__
                        )
                        mutants.append(mutant)
                        
        except Exception as e:
            logger.error(f"Error generating comparison mutants for {filepath}: {e}")
        
        return mutants


class LogicalOperatorMutator(MutationOperator):
    """Mutate logical operators (and, or, not)."""
    
    def __init__(self):
        self.operator_mappings = {
            'and': ['or'],
            'or': ['and']
        }
        
        self.ast_operators = {
            ast.And: 'and',
            ast.Or: 'or'
        }
    
    def get_mutation_type(self) -> MutationType:
        return MutationType.LOGICAL_OPERATOR
    
    def is_applicable(self, node: Any) -> bool:
        return isinstance(node, ast.BoolOp) and type(node.op) in self.ast_operators
    
    def generate_mutants(self, source_code: str, filepath: str) -> List[Mutant]:
        """Generate logical operator mutants."""
        mutants = []
        
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if self.is_applicable(node):
                    original_op = self.ast_operators[type(node.op)]
                    
                    for replacement_op in self.operator_mappings.get(original_op, []):
                        mutant_id = f"LOR_{node.lineno}_{node.col_offset}_{replacement_op}"
                        
                        mutant = Mutant(
                            id=mutant_id,
                            mutation_type=self.get_mutation_type(),
                            original_code=original_op,
                            mutated_code=replacement_op,
                            line_number=node.lineno,
                            column_start=node.col_offset,
                            column_end=node.col_offset + len(original_op),
                            description=f"Replace '{original_op}' with '{replacement_op}' at line {node.lineno}",
                            severity="high",
                            language="python",
                            ast_node_type=type(node.op).__name__
                        )
                        mutants.append(mutant)
                        
        except Exception as e:
            logger.error(f"Error generating logical mutants for {filepath}: {e}")
        
        return mutants


class ConstantValueMutator(MutationOperator):
    """Mutate constant values (numbers, strings, booleans)."""
    
    def get_mutation_type(self) -> MutationType:
        return MutationType.CONSTANT_VALUE
    
    def is_applicable(self, node: Any) -> bool:
        return isinstance(node, (ast.Constant, ast.Num, ast.Str, ast.NameConstant))
    
    def generate_mutants(self, source_code: str, filepath: str) -> List[Mutant]:
        """Generate constant value mutants."""
        mutants = []
        
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if self.is_applicable(node):
                    mutations = self._get_constant_mutations(node)
                    
                    for mutated_value in mutations:
                        mutant_id = f"CVR_{node.lineno}_{node.col_offset}_{str(mutated_value).replace(' ', '_')}"
                        
                        original_value = self._get_node_value(node)
                        
                        mutant = Mutant(
                            id=mutant_id,
                            mutation_type=self.get_mutation_type(),
                            original_code=str(original_value),
                            mutated_code=str(mutated_value),
                            line_number=node.lineno,
                            column_start=node.col_offset,
                            column_end=getattr(node, 'end_col_offset', node.col_offset + len(str(original_value))),
                            description=f"Replace constant '{original_value}' with '{mutated_value}' at line {node.lineno}",
                            severity=self._get_constant_severity(original_value, mutated_value),
                            language="python"
                        )
                        mutants.append(mutant)
                        
        except Exception as e:
            logger.error(f"Error generating constant mutants for {filepath}: {e}")
        
        return mutants
    
    def _get_node_value(self, node: ast.AST) -> Any:
        """Extract value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.NameConstant):
            return node.value
        return None
    
    def _get_constant_mutations(self, node: ast.AST) -> List[Any]:
        """Get possible mutations for a constant value."""
        value = self._get_node_value(node)
        mutations = []
        
        if isinstance(value, int):
            mutations.extend([
                value + 1, value - 1,
                0 if value != 0 else 1,
                1 if value != 1 else 2,
                -1 if value != -1 else -2
            ])
        elif isinstance(value, float):
            mutations.extend([
                value + 1.0, value - 1.0,
                0.0 if value != 0.0 else 1.0,
                1.0 if value != 1.0 else 2.0
            ])
        elif isinstance(value, str):
            if value:
                mutations.extend([
                    "", value + "X", value[:-1] if len(value) > 1 else "X"
                ])
            else:
                mutations.append("X")
        elif isinstance(value, bool):
            mutations.append(not value)
        elif value is None:
            mutations.extend([0, "", False])
        
        return [m for m in mutations if m != value]  # Remove same value
    
    def _get_constant_severity(self, original: Any, mutated: Any) -> str:
        """Determine severity of constant mutation."""
        if isinstance(original, bool) or original is None:
            return "critical"
        elif isinstance(original, (int, float)) and mutated == 0:
            return "high"
        elif isinstance(original, str) and mutated == "":
            return "high"
        else:
            return "medium"


class BoundaryValueMutator(MutationOperator):
    """Mutate boundary values to create off-by-one errors."""
    
    def get_mutation_type(self) -> MutationType:
        return MutationType.BOUNDARY_VALUE
    
    def is_applicable(self, node: Any) -> bool:
        # Apply to comparisons and numeric constants in specific contexts
        return (isinstance(node, ast.Compare) or
                (isinstance(node, (ast.Constant, ast.Num)) and isinstance(self._get_node_value(node), int)))
    
    def generate_mutants(self, source_code: str, filepath: str) -> List[Mutant]:
        """Generate boundary value mutants."""
        mutants = []
        
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Compare):
                    # Mutate comparison boundaries
                    mutants.extend(self._mutate_comparison_boundary(node, source_code))
                elif isinstance(node, (ast.Constant, ast.Num)):
                    # Mutate numeric constants that might be boundaries
                    mutants.extend(self._mutate_numeric_boundary(node, source_code))
                    
        except Exception as e:
            logger.error(f"Error generating boundary mutants for {filepath}: {e}")
        
        return mutants
    
    def _get_node_value(self, node: ast.AST) -> Any:
        """Extract value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        return None
    
    def _mutate_comparison_boundary(self, node: ast.Compare, source_code: str) -> List[Mutant]:
        """Mutate boundary conditions in comparisons."""
        mutants = []
        
        for comparator in node.comparators:
            if isinstance(comparator, (ast.Constant, ast.Num)):
                value = self._get_node_value(comparator)
                if isinstance(value, int):
                    # Create off-by-one mutations
                    for delta in [-1, 1]:
                        new_value = value + delta
                        mutant_id = f"BVR_{node.lineno}_{comparator.col_offset}_{new_value}"
                        
                        mutant = Mutant(
                            id=mutant_id,
                            mutation_type=self.get_mutation_type(),
                            original_code=str(value),
                            mutated_code=str(new_value),
                            line_number=node.lineno,
                            column_start=comparator.col_offset,
                            column_end=getattr(comparator, 'end_col_offset', comparator.col_offset + len(str(value))),
                            description=f"Off-by-one: Replace boundary '{value}' with '{new_value}' at line {node.lineno}",
                            severity="critical",
                            language="python"
                        )
                        mutants.append(mutant)
        
        return mutants
    
    def _mutate_numeric_boundary(self, node: ast.AST, source_code: str) -> List[Mutant]:
        """Mutate numeric constants that might be boundaries."""
        mutants = []
        value = self._get_node_value(node)
        
        if isinstance(value, int) and value in [0, 1, -1, 2, -2]:  # Common boundary values
            for delta in [-1, 1]:
                new_value = value + delta
                mutant_id = f"BVN_{node.lineno}_{node.col_offset}_{new_value}"
                
                mutant = Mutant(
                    id=mutant_id,
                    mutation_type=self.get_mutation_type(),
                    original_code=str(value),
                    mutated_code=str(new_value),
                    line_number=node.lineno,
                    column_start=node.col_offset,
                    column_end=getattr(node, 'end_col_offset', node.col_offset + len(str(value))),
                    description=f"Boundary numeric: Replace '{value}' with '{new_value}' at line {node.lineno}",
                    severity="high",
                    language="python"
                )
                mutants.append(mutant)
        
        return mutants


class MutationTestingEngine:
    """Main engine for mutation testing."""
    
    def __init__(self, operators: Optional[List[MutationOperator]] = None, timeout: int = 30):
        """Initialize mutation testing engine."""
        self.operators = operators or [
            ArithmeticOperatorMutator(),
            ComparisonOperatorMutator(),
            LogicalOperatorMutator(),
            ConstantValueMutator(),
            BoundaryValueMutator()
        ]
        self.timeout = timeout
    
    def generate_mutants(self, source_file: str) -> List[Mutant]:
        """Generate all possible mutants for a source file."""
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            all_mutants = []
            for operator in self.operators:
                mutants = operator.generate_mutants(source_code, source_file)
                all_mutants.extend(mutants)
            
            # Remove duplicates and sort by line number
            unique_mutants = self._deduplicate_mutants(all_mutants)
            return sorted(unique_mutants, key=lambda m: (m.line_number, m.column_start))
            
        except Exception as e:
            logger.error(f"Failed to generate mutants for {source_file}: {e}")
            return []
    
    def run_mutation_testing(self, source_file: str, test_files: List[str], 
                           max_mutants: Optional[int] = None) -> MutationScore:
        """Run complete mutation testing analysis."""
        start_time = time.time()
        
        # Generate mutants
        all_mutants = self.generate_mutants(source_file)
        if max_mutants:
            all_mutants = all_mutants[:max_mutants]
        
        if not all_mutants:
            return self._create_empty_mutation_score(source_file, test_files)
        
        # Test each mutant
        results = []
        for i, mutant in enumerate(all_mutants):
            logger.debug(f"Testing mutant {i+1}/{len(all_mutants)}: {mutant.id}")
            result = self._test_mutant(mutant, source_file, test_files)
            results.append(result)
        
        # Calculate scores and analyze results
        mutation_score = self._calculate_mutation_score(source_file, test_files, results, start_time)
        mutation_score.weak_spots = self._identify_weak_spots(results)
        
        return mutation_score
    
    def _test_mutant(self, mutant: Mutant, source_file: str, test_files: List[str]) -> MutationResult:
        """Test a single mutant against the test suite."""
        start_time = time.time()
        
        try:
            # Create mutated version of source file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                mutated_code = self._apply_mutant(mutant, source_file)
                temp_file.write(mutated_code)
                temp_file.flush()
                
                # Run tests against mutated code
                killed, failing_tests, error_message = self._run_tests_against_mutant(
                    temp_file.name, source_file, test_files
                )
                
                execution_time = time.time() - start_time
                
                return MutationResult(
                    mutant=mutant,
                    killed=killed,
                    execution_time=execution_time,
                    failing_tests=failing_tests,
                    error_message=error_message
                )
                
        except Exception as e:
            return MutationResult(
                mutant=mutant,
                killed=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
        finally:
            # Cleanup temp file
            try:
                if 'temp_file' in locals():
                    os.unlink(temp_file.name)
            except:
                pass
    
    def _apply_mutant(self, mutant: Mutant, source_file: str) -> str:
        """Apply a mutant to source code."""
        with open(source_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if mutant.line_number <= len(lines):
            line = lines[mutant.line_number - 1]  # Convert to 0-based index
            
            # Simple string replacement (could be more sophisticated)
            if mutant.original_code in line:
                new_line = line.replace(mutant.original_code, mutant.mutated_code, 1)
                lines[mutant.line_number - 1] = new_line
        
        return ''.join(lines)
    
    def _run_tests_against_mutant(self, mutant_file: str, original_file: str, 
                                 test_files: List[str]) -> Tuple[bool, List[str], Optional[str]]:
        """Run tests against a mutant and determine if it was killed.

        This implementation runs in an isolated sandbox to ensure the working tree
        is never modified. It:
        - Creates a TemporaryDirectory sandbox
        - Copies the project subtree that contains the source and tests
        - Writes the mutant contents into the sandboxed source path
        - Executes pytest within the sandbox cwd
        - Cleans up the sandbox on exit
        """
        try:
            # Resolve absolute paths
            original_path = Path(original_file).resolve()
            test_paths = [Path(tf).resolve() for tf in test_files]

            # Determine a common project root that contains both source and tests
            candidate_dirs = [original_path.parent] + [p.parent for p in test_paths]
            common_root = Path(os.path.commonpath([str(p) for p in candidate_dirs]))

            # Create sandbox and copy project subtree
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                sandbox_project_root = tmpdir_path / common_root.name

                # Copy the subtree, ignoring heavy/unnecessary folders
                def _ignore(dirpath, names):
                    ignore_names = set()
                    patterns = {
                        '.git', '.hg', '.svn', '.venv', 'venv', '__pycache__', '.pytest_cache', '.mypy_cache',
                        '.DS_Store', '.coverage', '.cache'
                    }
                    for name in names:
                        if name in patterns:
                            ignore_names.add(name)
                    return ignore_names

                shutil.copytree(common_root, sandbox_project_root, dirs_exist_ok=False, ignore=_ignore)

                # Compute sandboxed paths
                rel_src = original_path.relative_to(common_root)
                sandbox_src = sandbox_project_root / rel_src

                # Overwrite sandboxed source with mutant contents via SafeFS
                mutated_code = Path(mutant_file).read_text(encoding='utf-8')
                safe_fs = SafeFS(sandbox_project_root)
                # Copy-on-write: write to temp then atomic replace into place
                tmp_dst = sandbox_src.with_suffix(sandbox_src.suffix + ".mutant.tmp")
                safe_fs.write_text(tmp_dst, mutated_code, encoding='utf-8')
                safe_fs.replace(tmp_dst, sandbox_src)

                # Map test files into sandbox (preserve relative layout)
                sandbox_tests: List[str] = []
                for tp in test_paths:
                    try:
                        rel = tp.relative_to(common_root)
                        sandbox_tests.append(str((sandbox_project_root / rel)))
                    except ValueError:
                        # If a test path is outside common_root, run by name from cwd
                        sandbox_tests.append(str(tp.name))

                # Run tests inside sandbox
                cmd = ['python', '-m', 'pytest'] + sandbox_tests + ['--tb=short', '-q']
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=str(sandbox_project_root)
                )

                killed = result.returncode != 0
                failing_tests = self._extract_failing_tests(result.stdout, result.stderr)
                error_message = result.stderr if result.stderr else None

                return killed, failing_tests, error_message

        except subprocess.TimeoutExpired:
            return False, [], "Test execution timeout"
        except Exception as e:
            return False, [], str(e)
    
    def _extract_failing_tests(self, stdout: str, stderr: str) -> List[str]:
        """Extract failing test names from pytest output."""
        failing_tests = []
        
        # Look for test failures in output
        test_pattern = r'(\w+\.py::\w+(?:::\w+)?)\s+FAILED'
        matches = re.findall(test_pattern, stdout + stderr)
        failing_tests.extend(matches)
        
        return failing_tests
    
    def _calculate_mutation_score(self, source_file: str, test_files: List[str], 
                                 results: List[MutationResult], start_time: float) -> MutationScore:
        """Calculate mutation testing score and metrics."""
        total_mutants = len(results)
        killed_mutants = sum(1 for r in results if r.killed)
        surviving_mutants = sum(1 for r in results if not r.killed)
        timeout_mutants = sum(1 for r in results if "timeout" in (r.error_message or "").lower())
        error_mutants = sum(1 for r in results if r.error_message and "timeout" not in r.error_message.lower())
        
        mutation_score = (killed_mutants / total_mutants * 100) if total_mutants > 0 else 0
        effective_mutants = total_mutants - timeout_mutants - error_mutants
        mutation_score_effective = (killed_mutants / effective_mutants * 100) if effective_mutants > 0 else 0
        
        # Calculate mutation distribution (robust to mocks missing attributes)
        mutation_distribution: Dict[Any, int] = {}
        for result in results:
            mut_type = getattr(result.mutant, 'mutation_type', 'unknown')
            mutation_distribution[mut_type] = mutation_distribution.get(mut_type, 0) + 1
        
        # Performance metrics
        total_execution_time = time.time() - start_time
        average_test_time = sum(r.execution_time for r in results) / len(results) if results else 0
        
        return MutationScore(
            source_file=source_file,
            test_files=test_files,
            total_mutants=total_mutants,
            killed_mutants=killed_mutants,
            surviving_mutants=surviving_mutants,
            timeout_mutants=timeout_mutants,
            error_mutants=error_mutants,
            mutation_score=mutation_score,
            mutation_score_effective=mutation_score_effective,
            mutant_results=results,
            mutation_distribution=mutation_distribution,
            total_execution_time=total_execution_time,
            average_test_time=average_test_time
        )
    
    def _identify_weak_spots(self, results: List[MutationResult]) -> List[WeakSpot]:
        """Identify weak spots based on surviving mutants."""
        weak_spots = []
        
        # Group surviving mutants by location and type
        surviving_mutants = [r.mutant for r in results if not r.killed]
        location_groups = {}
        
        for mutant in surviving_mutants:
            location = f"{mutant.line_number}:{mutant.column_start}"
            if location not in location_groups:
                location_groups[location] = []
            location_groups[location].append(mutant)
        
        # Create weak spots for locations with multiple surviving mutants
        for location, mutants in location_groups.items():
            if len(mutants) >= 1:  # Even single surviving mutant is a weak spot
                # Determine predominant mutation type
                mutation_types = [m.mutation_type for m in mutants]
                predominant_type = max(set(mutation_types), key=mutation_types.count)
                
                # Create suggested tests
                suggested_tests = self._generate_test_suggestions(mutants, predominant_type)
                
                # Determine severity
                severity = self._determine_weakness_severity(mutants)
                
                weak_spot = WeakSpot(
                    location=location,
                    mutation_type=predominant_type,
                    surviving_mutants=mutants,
                    description=f"{len(mutants)} surviving mutants at {location}",
                    suggested_tests=suggested_tests,
                    severity=severity,
                    function_name=mutants[0].context.get('function_context') if mutants[0].context else None,
                    surrounding_code='\n'.join(mutants[0].context.get('surrounding_code', [])) if mutants[0].context else ""
                )
                weak_spots.append(weak_spot)
        
        return sorted(weak_spots, key=lambda ws: (ws.severity, len(ws.surviving_mutants)), reverse=True)
    
    def _generate_test_suggestions(self, mutants: List[Mutant], mutation_type: MutationType) -> List[str]:
        """Generate test suggestions based on surviving mutants."""
        suggestions = []
        
        if mutation_type == MutationType.ARITHMETIC_OPERATOR:
            suggestions.append("Add tests that verify exact arithmetic results, not just non-zero values")
            suggestions.append("Test edge cases where different operators would produce different results")
        elif mutation_type == MutationType.COMPARISON_OPERATOR:
            suggestions.append("Add boundary value tests (equal, just above, just below)")
            suggestions.append("Test all comparison edge cases explicitly")
        elif mutation_type == MutationType.LOGICAL_OPERATOR:
            suggestions.append("Test both true and false branches of logical expressions")
            suggestions.append("Add tests where AND vs OR would produce different results")
        elif mutation_type == MutationType.CONSTANT_VALUE:
            suggestions.append("Test with the exact expected values, not just type checking")
            suggestions.append("Add tests for boundary values and special constants")
        elif mutation_type == MutationType.BOUNDARY_VALUE:
            suggestions.append("Add off-by-one error tests")
            suggestions.append("Test exact boundary conditions")
        
        return suggestions
    
    def _determine_weakness_severity(self, mutants: List[Mutant]) -> str:
        """Determine severity of a weak spot based on mutants."""
        severities = [m.severity for m in mutants]
        if "critical" in severities:
            return "critical"
        elif "high" in severities:
            return "high"
        elif len(mutants) > 3:
            return "high"  # Many surviving mutants = high severity
        else:
            return "medium"
    
    def _deduplicate_mutants(self, mutants: List[Mutant]) -> List[Mutant]:
        """Remove duplicate mutants."""
        seen = set()
        unique_mutants = []
        
        for mutant in mutants:
            key = (mutant.line_number, mutant.column_start, mutant.original_code, mutant.mutated_code)
            if key not in seen:
                seen.add(key)
                unique_mutants.append(mutant)
        
        return unique_mutants
    
    def _create_empty_mutation_score(self, source_file: str, test_files: List[str]) -> MutationScore:
        """Create empty mutation score for failed analysis."""
        return MutationScore(
            source_file=source_file,
            test_files=test_files,
            total_mutants=0,
            killed_mutants=0,
            surviving_mutants=0,
            timeout_mutants=0,
            error_mutants=0,
            mutation_score=0.0,
            mutation_score_effective=0.0,
            mutant_results=[],
            weak_spots=[]
        ) 
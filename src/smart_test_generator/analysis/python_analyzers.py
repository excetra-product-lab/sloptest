"""Python-specific quality and mutation analyzers."""

import ast
import re
import logging
from typing import Dict, List, Optional, Set, Any, Tuple

from smart_test_generator.models.data_models import (
    QualityDimension, QualityScore, MutationType, Mutant
)
from smart_test_generator.analysis.quality_analyzer import QualityAnalyzer
from smart_test_generator.analysis.mutation_engine import MutationOperator

logger = logging.getLogger(__name__)


class PythonIdiomAnalyzer(QualityAnalyzer):
    """Analyze Python-specific testing idioms and best practices."""
    
    def __init__(self):
        self.python_test_patterns = {
            'pytest_fixtures': [
                r'@pytest\.fixture', r'def\s+\w+\(\s*\):\s*\n\s*yield',
                r'@fixture', r'conftest\.py'
            ],
            'pytest_parametrize': [
                r'@pytest\.mark\.parametrize', r'@parametrize',
                r'pytest\.param\('
            ],
            'pytest_markers': [
                r'@pytest\.mark\.\w+', r'@mark\.\w+',
                r'pytest\.mark\.skip', r'pytest\.mark\.xfail'
            ],
            'context_managers': [
                r'with\s+pytest\.raises', r'with\s+\w+\(',
                r'with\s+mock\.patch', r'with\s+tempfile\.'
            ],
            'mock_usage': [
                r'from\s+unittest\.mock\s+import', r'mock\.Mock\(',
                r'mock\.patch', r'@patch', r'MagicMock',
                r'\.assert_called', r'\.call_count'
            ],
            'async_testing': [
                r'async\s+def\s+test_', r'await\s+',
                r'@pytest\.mark\.asyncio', r'asyncio\.run'
            ]
        }
        
        self.anti_patterns = {
            'unittest_in_pytest': r'from\s+unittest\s+import.*TestCase',
            'hard_coded_paths': r'["\'][/\\]?(?:Users|home|C:|D:)[/\\]',
            'print_debugging': r'\bprint\s*\(',
            'bare_except': r'except\s*:',
            'sleep_in_tests': r'time\.sleep\(',
            'hardcoded_dates': r'datetime\(\d{4},\s*\d{1,2},\s*\d{1,2}\)'
        }
    
    def get_dimension(self) -> QualityDimension:
        return QualityDimension.READABILITY
    
    def analyze(self, test_code: str, source_code: str = "") -> QualityScore:
        """Analyze Python-specific testing patterns."""
        score = 0
        max_score = 100
        details = {}
        suggestions = []
        
        # Check for modern pytest patterns
        pytest_patterns_found = 0
        for category, patterns in self.python_test_patterns.items():
            found = sum(len(re.findall(pattern, test_code, re.IGNORECASE | re.MULTILINE)) 
                       for pattern in patterns)
            if found > 0:
                pytest_patterns_found += 1
                details[f'{category}_count'] = found
        
        # Score based on modern pytest usage
        if pytest_patterns_found > 0:
            score += min(pytest_patterns_found * 15, 60)  # Up to 60 points for good patterns
            details['uses_modern_pytest'] = True
        else:
            suggestions.append("Consider using pytest fixtures, parametrize, and markers for better tests")
            details['uses_modern_pytest'] = False
        
        # Check for anti-patterns
        anti_pattern_count = 0
        for pattern_name, pattern in self.anti_patterns.items():
            if re.search(pattern, test_code, re.IGNORECASE):
                anti_pattern_count += 1
                suggestions.append(f"Avoid {pattern_name.replace('_', ' ')}")
                details[f'has_{pattern_name}'] = True
        
        # Penalty for anti-patterns
        score -= anti_pattern_count * 10
        
        # Check for proper test isolation
        isolation_score = self._check_test_isolation(test_code)
        score += isolation_score
        details['isolation_score'] = isolation_score
        
        # Check for proper assertion usage
        assertion_score = self._check_python_assertions(test_code)
        score += assertion_score
        details['assertion_score'] = assertion_score
        
        if isolation_score < 10:
            suggestions.append("Use fixtures and context managers for better test isolation")
        if assertion_score < 15:
            suggestions.append("Use more specific pytest assertions like assert_called_with()")
        
        return QualityScore(
            dimension=self.get_dimension(),
            score=max(0, min(score, max_score)),
            details=details,
            suggestions=suggestions
        )
    
    def _check_test_isolation(self, test_code: str) -> float:
        """Check for proper test isolation patterns."""
        score = 0
        
        # Fixtures usage
        if re.search(r'@pytest\.fixture|@fixture', test_code):
            score += 10
        
        # Context managers
        if re.search(r'with\s+\w+', test_code):
            score += 5
        
        # Setup/teardown patterns
        if re.search(r'def\s+setup_|def\s+teardown_', test_code):
            score += 5
        
        # Mock patches as context managers
        if re.search(r'with\s+mock\.patch', test_code):
            score += 5
        
        return min(score, 20)
    
    def _check_python_assertions(self, test_code: str) -> float:
        """Check for proper Python assertion usage."""
        score = 0
        
        # Pytest-specific assertions
        pytest_assertions = [
            r'assert\s+\w+\s+in\s+', r'assert\s+\w+\s+not\s+in\s+',
            r'assert\s+isinstance\(', r'assert\s+issubclass\(',
            r'pytest\.approx\(', r'assert.*pytest\.approx'
        ]
        
        for pattern in pytest_assertions:
            if re.search(pattern, test_code):
                score += 3
        
        # Mock assertions
        mock_assertions = [
            r'\.assert_called\(', r'\.assert_called_with\(',
            r'\.assert_called_once\(', r'\.assert_not_called\(',
            r'\.assert_has_calls\('
        ]
        
        for pattern in mock_assertions:
            if re.search(pattern, test_code):
                score += 2
        
        return min(score, 20)


class PythonExceptionMutator(MutationOperator):
    """Python-specific exception handling mutations."""
    
    def get_mutation_type(self) -> MutationType:
        return MutationType.EXCEPTION_HANDLING
    
    def is_applicable(self, node: Any) -> bool:
        return isinstance(node, (ast.Try, ast.Raise, ast.ExceptHandler))
    
    def generate_mutants(self, source_code: str, filepath: str) -> List[Mutant]:
        """Generate exception handling mutants."""
        mutants = []
        
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    mutants.extend(self._mutate_try_block(node, source_code))
                elif isinstance(node, ast.Raise):
                    mutants.extend(self._mutate_raise_statement(node, source_code))
                elif isinstance(node, ast.ExceptHandler):
                    mutants.extend(self._mutate_except_handler(node, source_code))
                    
        except Exception as e:
            logger.error(f"Error generating exception mutants for {filepath}: {e}")
        
        return mutants
    
    def _mutate_try_block(self, node: ast.Try, source_code: str) -> List[Mutant]:
        """Mutate try blocks."""
        mutants = []
        
        # Remove try-except (dangerous mutation)
        mutant = Mutant(
            id=f"EXC_TRY_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="try:",
            mutated_code="# try removed",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 4,
            description=f"Remove try-except block at line {node.lineno}",
            severity="critical",
            language="python"
        )
        mutants.append(mutant)
        
        return mutants
    
    def _mutate_raise_statement(self, node: ast.Raise, source_code: str) -> List[Mutant]:
        """Mutate raise statements."""
        mutants = []
        
        # Remove raise statement
        mutant = Mutant(
            id=f"EXC_RAISE_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="raise",
            mutated_code="# raise removed",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 5,
            description=f"Remove raise statement at line {node.lineno}",
            severity="high",
            language="python"
        )
        mutants.append(mutant)
        
        return mutants
    
    def _mutate_except_handler(self, node: ast.ExceptHandler, source_code: str) -> List[Mutant]:
        """Mutate except handlers."""
        mutants = []
        
        if node.type:  # Specific exception type
            # Change to bare except
            mutant = Mutant(
                id=f"EXC_BARE_{node.lineno}",
                mutation_type=self.get_mutation_type(),
                original_code=f"except {ast.unparse(node.type)}:",
                mutated_code="except:",
                line_number=node.lineno,
                column_start=node.col_offset,
                column_end=node.col_offset + 6,
                description=f"Change specific exception to bare except at line {node.lineno}",
                severity="medium",
                language="python"
            )
            mutants.append(mutant)
        
        return mutants


class PythonMethodCallMutator(MutationOperator):
    """Mutate Python method calls."""
    
    def __init__(self):
        self.method_mutations = {
            'append': ['extend', 'insert'],
            'extend': ['append'],
            'insert': ['append'],
            'remove': ['pop', 'discard'],
            'pop': ['remove'],
            'add': ['update', 'discard'],
            'update': ['add'],
            'get': ['pop', '__getitem__'],
            'keys': ['values', 'items'],
            'values': ['keys', 'items'],
            'items': ['keys', 'values'],
            'sort': ['reverse'],
            'reverse': ['sort'],
            'upper': ['lower', 'title'],
            'lower': ['upper', 'title'],
            'strip': ['lstrip', 'rstrip'],
            'lstrip': ['strip', 'rstrip'],
            'rstrip': ['strip', 'lstrip']
        }
    
    def get_mutation_type(self) -> MutationType:
        return MutationType.METHOD_CALL
    
    def is_applicable(self, node: Any) -> bool:
        return (isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.attr, str))
    
    def generate_mutants(self, source_code: str, filepath: str) -> List[Mutant]:
        """Generate method call mutants."""
        mutants = []
        
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if self.is_applicable(node):
                    method_name = node.func.attr
                    
                    if method_name in self.method_mutations:
                        for replacement in self.method_mutations[method_name]:
                            mutant_id = f"MCR_{node.lineno}_{node.col_offset}_{replacement}"
                            
                            mutant = Mutant(
                                id=mutant_id,
                                mutation_type=self.get_mutation_type(),
                                original_code=method_name,
                                mutated_code=replacement,
                                line_number=node.lineno,
                                column_start=node.func.end_col_offset - len(method_name) if hasattr(node.func, 'end_col_offset') else node.col_offset,
                                column_end=node.func.end_col_offset if hasattr(node.func, 'end_col_offset') else node.col_offset + len(method_name),
                                description=f"Replace method '{method_name}' with '{replacement}' at line {node.lineno}",
                                severity="medium",
                                language="python"
                            )
                            mutants.append(mutant)
                            
        except Exception as e:
            logger.error(f"Error generating method call mutants for {filepath}: {e}")
        
        return mutants


class PythonLoopMutator(MutationOperator):
    """Mutate Python loop constructs."""
    
    def get_mutation_type(self) -> MutationType:
        return MutationType.LOOP_MUTATION
    
    def is_applicable(self, node: Any) -> bool:
        return isinstance(node, (ast.For, ast.While, ast.Break, ast.Continue))
    
    def generate_mutants(self, source_code: str, filepath: str) -> List[Mutant]:
        """Generate loop-related mutants."""
        mutants = []
        
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    mutants.extend(self._mutate_for_loop(node))
                elif isinstance(node, ast.While):
                    mutants.extend(self._mutate_while_loop(node))
                elif isinstance(node, ast.Break):
                    mutants.extend(self._mutate_break_statement(node))
                elif isinstance(node, ast.Continue):
                    mutants.extend(self._mutate_continue_statement(node))
                    
        except Exception as e:
            logger.error(f"Error generating loop mutants for {filepath}: {e}")
        
        return mutants
    
    def _mutate_for_loop(self, node: ast.For) -> List[Mutant]:
        """Mutate for loops."""
        mutants = []
        
        # Change range bounds (if using range)
        if (isinstance(node.iter, ast.Call) and 
            isinstance(node.iter.func, ast.Name) and 
            node.iter.func.id == 'range'):
            
            # Mutate range arguments
            if len(node.iter.args) >= 1:
                mutant = Mutant(
                    id=f"LOOP_RANGE_{node.lineno}",
                    mutation_type=self.get_mutation_type(),
                    original_code="range",
                    mutated_code="range_mutated",  # This would need more sophisticated replacement
                    line_number=node.lineno,
                    column_start=node.col_offset,
                    column_end=node.col_offset + 3,
                    description=f"Mutate range bounds in for loop at line {node.lineno}",
                    severity="medium",
                    language="python"
                )
                mutants.append(mutant)
        
        return mutants
    
    def _mutate_while_loop(self, node: ast.While) -> List[Mutant]:
        """Mutate while loops."""
        mutants = []
        
        # Change while condition
        mutant = Mutant(
            id=f"LOOP_WHILE_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="while",
            mutated_code="if",  # Change while to if (execute only once)
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 5,
            description=f"Change while loop to if statement at line {node.lineno}",
            severity="high",
            language="python"
        )
        mutants.append(mutant)
        
        return mutants
    
    def _mutate_break_statement(self, node: ast.Break) -> List[Mutant]:
        """Mutate break statements."""
        mutants = []
        
        # Remove break
        mutant = Mutant(
            id=f"LOOP_BREAK_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="break",
            mutated_code="# break removed",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 5,
            description=f"Remove break statement at line {node.lineno}",
            severity="high",
            language="python"
        )
        mutants.append(mutant)
        
        # Change break to continue
        mutant = Mutant(
            id=f"LOOP_BREAK_CONT_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="break",
            mutated_code="continue",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 5,
            description=f"Change break to continue at line {node.lineno}",
            severity="medium",
            language="python"
        )
        mutants.append(mutant)
        
        return mutants
    
    def _mutate_continue_statement(self, node: ast.Continue) -> List[Mutant]:
        """Mutate continue statements."""
        mutants = []
        
        # Remove continue
        mutant = Mutant(
            id=f"LOOP_CONT_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="continue",
            mutated_code="# continue removed",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 8,
            description=f"Remove continue statement at line {node.lineno}",
            severity="medium",
            language="python"
        )
        mutants.append(mutant)
        
        # Change continue to break
        mutant = Mutant(
            id=f"LOOP_CONT_BREAK_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="continue",
            mutated_code="break",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 8,
            description=f"Change continue to break at line {node.lineno}",
            severity="medium",
            language="python"
        )
        mutants.append(mutant)
        
        return mutants


class PythonConditionalMutator(MutationOperator):
    """Mutate Python conditional statements."""
    
    def get_mutation_type(self) -> MutationType:
        return MutationType.CONDITIONAL
    
    def is_applicable(self, node: Any) -> bool:
        return isinstance(node, (ast.If, ast.IfExp))
    
    def generate_mutants(self, source_code: str, filepath: str) -> List[Mutant]:
        """Generate conditional mutants."""
        mutants = []
        
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    mutants.extend(self._mutate_if_statement(node))
                elif isinstance(node, ast.IfExp):
                    mutants.extend(self._mutate_ternary_operator(node))
                    
        except Exception as e:
            logger.error(f"Error generating conditional mutants for {filepath}: {e}")
        
        return mutants
    
    def _mutate_if_statement(self, node: ast.If) -> List[Mutant]:
        """Mutate if statements."""
        mutants = []
        
        # Negate condition
        mutant = Mutant(
            id=f"COND_NEG_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="if",
            mutated_code="if not",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 2,
            description=f"Negate if condition at line {node.lineno}",
            severity="high",
            language="python"
        )
        mutants.append(mutant)
        
        # Remove if condition (make it always true)
        mutant = Mutant(
            id=f"COND_TRUE_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="if condition:",
            mutated_code="if True:",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 2,
            description=f"Make if condition always True at line {node.lineno}",
            severity="critical",
            language="python"
        )
        mutants.append(mutant)
        
        # Make condition always false
        mutant = Mutant(
            id=f"COND_FALSE_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="if condition:",
            mutated_code="if False:",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 2,
            description=f"Make if condition always False at line {node.lineno}",
            severity="critical",
            language="python"
        )
        mutants.append(mutant)
        
        return mutants
    
    def _mutate_ternary_operator(self, node: ast.IfExp) -> List[Mutant]:
        """Mutate ternary operators (conditional expressions)."""
        mutants = []
        
        # Negate condition
        mutant = Mutant(
            id=f"TERN_NEG_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="ternary_condition",
            mutated_code="not ternary_condition",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 1,
            description=f"Negate ternary condition at line {node.lineno}",
            severity="medium",
            language="python"
        )
        mutants.append(mutant)
        
        return mutants


def get_python_quality_analyzers() -> List[QualityAnalyzer]:
    """Get all Python-specific quality analyzers."""
    from smart_test_generator.analysis.quality_analyzer import (
        EdgeCaseAnalyzer, AssertionStrengthAnalyzer, 
        MaintainabilityAnalyzer, BugDetectionAnalyzer
    )
    
    return [
        EdgeCaseAnalyzer(),
        AssertionStrengthAnalyzer(),
        MaintainabilityAnalyzer(),
        BugDetectionAnalyzer(),
        PythonIdiomAnalyzer()
    ]


def get_python_mutation_operators() -> List[MutationOperator]:
    """Get all Python-specific mutation operators."""
    from smart_test_generator.analysis.mutation_engine import (
        ArithmeticOperatorMutator, ComparisonOperatorMutator,
        LogicalOperatorMutator, ConstantValueMutator, BoundaryValueMutator
    )
    
    return [
        ArithmeticOperatorMutator(),
        ComparisonOperatorMutator(),
        LogicalOperatorMutator(),
        ConstantValueMutator(),
        BoundaryValueMutator(),
        PythonExceptionMutator(),
        PythonMethodCallMutator(),
        PythonLoopMutator(),
        PythonConditionalMutator()
    ] 
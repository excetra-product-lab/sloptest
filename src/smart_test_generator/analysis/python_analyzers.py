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
    """Enhanced Python-specific exception handling mutations."""
    
    def __init__(self):
        # Exception type substitutions
        self.exception_substitutions = {
            'ValueError': ['TypeError', 'AttributeError', 'KeyError', 'IndexError'],
            'TypeError': ['ValueError', 'AttributeError', 'KeyError'],
            'AttributeError': ['TypeError', 'ValueError', 'KeyError'],
            'KeyError': ['ValueError', 'TypeError', 'IndexError'],
            'IndexError': ['KeyError', 'ValueError', 'TypeError'],
            'FileNotFoundError': ['PermissionError', 'IsADirectoryError'],
            'PermissionError': ['FileNotFoundError', 'IsADirectoryError'],
            'ConnectionError': ['TimeoutError', 'OSError'],
            'TimeoutError': ['ConnectionError', 'OSError'],
            'RuntimeError': ['SystemError', 'OSError'],
            'SystemError': ['RuntimeError', 'OSError'],
            'Exception': ['ValueError', 'TypeError', 'RuntimeError']
        }
        
        # Common exception messages to mutate
        self.message_mutations = {
            'Invalid input': 'Valid input',
            'Not found': 'Found',
            'Access denied': 'Access granted',
            'Connection failed': 'Connection succeeded',
            'Timeout': 'No timeout',
            'Error': 'Success',
            'Failed': 'Succeeded',
            'Invalid': 'Valid'
        }
    
    def get_mutation_type(self) -> MutationType:
        return MutationType.EXCEPTION_HANDLING
    
    def is_applicable(self, node: Any) -> bool:
        return isinstance(node, (ast.Try, ast.Raise, ast.ExceptHandler, ast.With, ast.withitem))
    
    def generate_mutants(self, source_code: str, filepath: str) -> List[Mutant]:
        """Generate enhanced exception handling mutants."""
        mutants = []
        
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    mutants.extend(self._mutate_try_block(node, source_code))
                    mutants.extend(self._mutate_finally_block(node, source_code))
                elif isinstance(node, ast.Raise):
                    mutants.extend(self._mutate_raise_statement(node, source_code))
                elif isinstance(node, ast.ExceptHandler):
                    mutants.extend(self._mutate_except_handler(node, source_code))
                elif isinstance(node, ast.With):
                    mutants.extend(self._mutate_context_manager(node, source_code))
                    
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
        """Enhanced mutation of raise statements."""
        mutants = []
        
        # Remove raise statement
        mutant = Mutant(
            id=f"EXC_RAISE_REMOVE_{node.lineno}",
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
        
        # Mutate exception type if present
        if node.exc and isinstance(node.exc, ast.Call):
            if isinstance(node.exc.func, ast.Name):
                exception_name = node.exc.func.id
                if exception_name in self.exception_substitutions:
                    for substitute in self.exception_substitutions[exception_name]:
                        mutant = Mutant(
                            id=f"EXC_TYPE_SUB_{node.lineno}_{substitute}",
                            mutation_type=self.get_mutation_type(),
                            original_code=exception_name,
                            mutated_code=substitute,
                            line_number=node.lineno,
                            column_start=node.col_offset,
                            column_end=node.col_offset + len(exception_name),
                            description=f"Replace {exception_name} with {substitute} at line {node.lineno}",
                            severity="medium",
                            language="python"
                        )
                        mutants.append(mutant)
            
            # Mutate exception message if present
            if node.exc.args and len(node.exc.args) > 0:
                first_arg = node.exc.args[0]
                if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                    original_msg = first_arg.value
                    
                    # Remove message
                    mutant = Mutant(
                        id=f"EXC_MSG_REMOVE_{node.lineno}",
                        mutation_type=self.get_mutation_type(),
                        original_code=f'"{original_msg}"',
                        mutated_code='""',
                        line_number=node.lineno,
                        column_start=first_arg.col_offset,
                        column_end=getattr(first_arg, 'end_col_offset', first_arg.col_offset + len(original_msg) + 2),
                        description=f"Remove exception message at line {node.lineno}",
                        severity="low",
                        language="python"
                    )
                    mutants.append(mutant)
                    
                    # Mutate specific message content
                    for old_msg, new_msg in self.message_mutations.items():
                        if old_msg.lower() in original_msg.lower():
                            mutated_msg = original_msg.replace(old_msg, new_msg)
                            mutant = Mutant(
                                id=f"EXC_MSG_MUT_{node.lineno}",
                                mutation_type=self.get_mutation_type(),
                                original_code=f'"{original_msg}"',
                                mutated_code=f'"{mutated_msg}"',
                                line_number=node.lineno,
                                column_start=first_arg.col_offset,
                                column_end=getattr(first_arg, 'end_col_offset', first_arg.col_offset + len(original_msg) + 2),
                                description=f"Mutate exception message at line {node.lineno}",
                                severity="low",
                                language="python"
                            )
                            mutants.append(mutant)
                            break
        
        return mutants
    
    def _mutate_except_handler(self, node: ast.ExceptHandler, source_code: str) -> List[Mutant]:
        """Enhanced mutation of except handlers."""
        mutants = []
        
        # Enhanced except handler mutations
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
            
            # Substitute exception types
            if isinstance(node.type, ast.Name):
                exception_name = node.type.id
                if exception_name in self.exception_substitutions:
                    for substitute in self.exception_substitutions[exception_name][:2]:  # Limit to avoid too many mutants
                        mutant = Mutant(
                            id=f"EXC_HANDLER_SUB_{node.lineno}_{substitute}",
                            mutation_type=self.get_mutation_type(),
                            original_code=f"except {exception_name}:",
                            mutated_code=f"except {substitute}:",
                            line_number=node.lineno,
                            column_start=node.col_offset,
                            column_end=node.col_offset + 6 + len(exception_name),
                            description=f"Replace {exception_name} with {substitute} in except handler at line {node.lineno}",
                            severity="medium",
                            language="python"
                        )
                        mutants.append(mutant)
        else:
            # Change bare except to specific exception
            mutant = Mutant(
                id=f"EXC_SPECIFIC_{node.lineno}",
                mutation_type=self.get_mutation_type(),
                original_code="except:",
                mutated_code="except Exception:",
                line_number=node.lineno,
                column_start=node.col_offset,
                column_end=node.col_offset + 7,
                description=f"Change bare except to specific Exception at line {node.lineno}",
                severity="low",
                language="python"
            )
            mutants.append(mutant)
        
        return mutants
    
    def _mutate_finally_block(self, node: ast.Try, source_code: str) -> List[Mutant]:
        """Mutate finally blocks."""
        mutants = []
        
        if node.finalbody:  # Has finally block
            # Remove finally block
            mutant = Mutant(
                id=f"EXC_FINALLY_REMOVE_{node.lineno}",
                mutation_type=self.get_mutation_type(),
                original_code="finally:",
                mutated_code="# finally removed",
                line_number=node.lineno + len(node.body) + len(node.handlers) + len(node.orelse),
                column_start=0,
                column_end=8,
                description=f"Remove finally block at line {node.lineno}",
                severity="medium",
                language="python"
            )
            mutants.append(mutant)
        
        return mutants
    
    def _mutate_context_manager(self, node: ast.With, source_code: str) -> List[Mutant]:
        """Mutate context manager exception handling."""
        mutants = []
        
        # Check if it's an exception-related context manager
        for item in node.items:
            if isinstance(item.context_expr, ast.Call):
                if isinstance(item.context_expr.func, ast.Attribute):
                    # pytest.raises, assertRaises, etc.
                    if item.context_expr.func.attr in ['raises', 'assertRaises']:
                        # Mutate the expected exception
                        if item.context_expr.args:
                            first_arg = item.context_expr.args[0]
                            if isinstance(first_arg, ast.Name):
                                exception_name = first_arg.id
                                if exception_name in self.exception_substitutions:
                                    for substitute in self.exception_substitutions[exception_name][:1]:
                                        mutant = Mutant(
                                            id=f"EXC_CTX_SUB_{node.lineno}_{substitute}",
                                            mutation_type=self.get_mutation_type(),
                                            original_code=exception_name,
                                            mutated_code=substitute,
                                            line_number=node.lineno,
                                            column_start=first_arg.col_offset,
                                            column_end=first_arg.col_offset + len(exception_name),
                                            description=f"Replace {exception_name} with {substitute} in context manager at line {node.lineno}",
                                            severity="medium",
                                            language="python"
                                        )
                                        mutants.append(mutant)
        
        return mutants


class PythonMethodCallMutator(MutationOperator):
    """Enhanced Python method call mutations."""
    
    def __init__(self):
        # Basic method mutations
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
        
        # Magic method mutations
        self.magic_method_mutations = {
            '__len__': ['__bool__', '__nonzero__'],
            '__str__': ['__repr__', '__unicode__'],
            '__repr__': ['__str__'],
            '__bool__': ['__len__'],
            '__nonzero__': ['__bool__'],  # Python 2 compatibility
            '__iter__': ['__reversed__'],
            '__reversed__': ['__iter__'],
            '__getitem__': ['__getattr__', 'get'],
            '__setitem__': ['__setattr__'],
            '__contains__': ['__eq__'],
            '__eq__': ['__ne__', '__contains__'],
            '__ne__': ['__eq__'],
            '__lt__': ['__le__', '__gt__', '__ge__'],
            '__le__': ['__lt__', '__gt__', '__ge__'],
            '__gt__': ['__ge__', '__lt__', '__le__'],
            '__ge__': ['__gt__', '__lt__', '__le__']
        }
        
        # Property vs method access patterns
        self.property_method_pairs = {
            'length': '__len__()',
            'size': '__len__()',
            'count': '__len__()',
            'string': '__str__()',
            'representation': '__repr__()'
        }
        
        # Common parameter mutations
        self.parameter_mutations = {
            'remove_first_param': 'Remove first parameter',
            'remove_last_param': 'Remove last parameter',
            'add_none_param': 'Add None parameter',
            'swap_first_two': 'Swap first two parameters',
            'duplicate_first': 'Duplicate first parameter'
        }
    
    def get_mutation_type(self) -> MutationType:
        return MutationType.METHOD_CALL
    
    def is_applicable(self, node: Any) -> bool:
        return (isinstance(node, (ast.Call, ast.Attribute)) and
                (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.attr, str)) or
                (isinstance(node, ast.Attribute) and isinstance(node.attr, str)))
    
    def generate_mutants(self, source_code: str, filepath: str) -> List[Mutant]:
        """Generate enhanced method call mutants."""
        mutants = []
        
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    method_name = node.func.attr
                    
                    # Basic method substitutions
                    if method_name in self.method_mutations:
                        mutants.extend(self._mutate_basic_methods(node, method_name))
                    
                    # Magic method mutations
                    if method_name in self.magic_method_mutations:
                        mutants.extend(self._mutate_magic_methods(node, method_name))
                    
                    # Parameter mutations
                    mutants.extend(self._mutate_parameters(node, method_name))
                    
                    # Method chain mutations
                    mutants.extend(self._mutate_method_chains(node, method_name))
                
                # Property access mutations
                elif isinstance(node, ast.Attribute):
                    mutants.extend(self._mutate_property_access(node))
                    
        except Exception as e:
            logger.error(f"Error generating method call mutants for {filepath}: {e}")
        
        return mutants
    
    def _mutate_basic_methods(self, node: ast.Call, method_name: str) -> List[Mutant]:
        """Generate basic method substitution mutants."""
        mutants = []
        
        for replacement in self.method_mutations[method_name]:
            mutant_id = f"MCR_BASIC_{node.lineno}_{node.col_offset}_{replacement}"
            
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
        
        return mutants
    
    def _mutate_magic_methods(self, node: ast.Call, method_name: str) -> List[Mutant]:
        """Generate magic method substitution mutants."""
        mutants = []
        
        for replacement in self.magic_method_mutations[method_name]:
            mutant_id = f"MCR_MAGIC_{node.lineno}_{node.col_offset}_{replacement}"
            
            mutant = Mutant(
                id=mutant_id,
                mutation_type=self.get_mutation_type(),
                original_code=method_name,
                mutated_code=replacement,
                line_number=node.lineno,
                column_start=node.func.end_col_offset - len(method_name) if hasattr(node.func, 'end_col_offset') else node.col_offset,
                column_end=node.func.end_col_offset if hasattr(node.func, 'end_col_offset') else node.col_offset + len(method_name),
                description=f"Replace magic method '{method_name}' with '{replacement}' at line {node.lineno}",
                severity="medium",
                language="python"
            )
            mutants.append(mutant)
        
        return mutants
    
    def _mutate_parameters(self, node: ast.Call, method_name: str) -> List[Mutant]:
        """Generate parameter mutation mutants."""
        mutants = []
        
        if node.args:
            # Remove first parameter
            if len(node.args) > 1:
                mutant = Mutant(
                    id=f"MCR_PARAM_REMOVE_FIRST_{node.lineno}",
                    mutation_type=self.get_mutation_type(),
                    original_code=f"{method_name}(args...)",
                    mutated_code=f"{method_name}(args[1:]...)",
                    line_number=node.lineno,
                    column_start=node.col_offset,
                    column_end=getattr(node, 'end_col_offset', node.col_offset + 10),
                    description=f"Remove first parameter from {method_name} at line {node.lineno}",
                    severity="medium",
                    language="python"
                )
                mutants.append(mutant)
            
            # Remove last parameter
            if len(node.args) > 1:
                mutant = Mutant(
                    id=f"MCR_PARAM_REMOVE_LAST_{node.lineno}",
                    mutation_type=self.get_mutation_type(),
                    original_code=f"{method_name}(args...)",
                    mutated_code=f"{method_name}(args[:-1]...)",
                    line_number=node.lineno,
                    column_start=node.col_offset,
                    column_end=getattr(node, 'end_col_offset', node.col_offset + 10),
                    description=f"Remove last parameter from {method_name} at line {node.lineno}",
                    severity="medium",
                    language="python"
                )
                mutants.append(mutant)
        
        # Add None parameter
        mutant = Mutant(
            id=f"MCR_PARAM_ADD_NONE_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code=f"{method_name}(args)",
            mutated_code=f"{method_name}(args, None)",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=getattr(node, 'end_col_offset', node.col_offset + 10),
            description=f"Add None parameter to {method_name} at line {node.lineno}",
            severity="low",
            language="python"
        )
        mutants.append(mutant)
        
        return mutants
    
    def _mutate_method_chains(self, node: ast.Call, method_name: str) -> List[Mutant]:
        """Generate method chain mutation mutants."""
        mutants = []
        
        # Check if this is part of a method chain (parent is also a method call)
        parent = getattr(node, 'parent', None)
        if isinstance(parent, ast.Attribute) and isinstance(parent.value, ast.Call):
            # Break the chain by removing this method call
            mutant = Mutant(
                id=f"MCR_CHAIN_BREAK_{node.lineno}",
                mutation_type=self.get_mutation_type(),
                original_code=f"obj.{method_name}()",
                mutated_code="obj",
                line_number=node.lineno,
                column_start=node.col_offset,
                column_end=getattr(node, 'end_col_offset', node.col_offset + len(method_name) + 2),
                description=f"Break method chain by removing {method_name} at line {node.lineno}",
                severity="high",
                language="python"
            )
            mutants.append(mutant)
        
        return mutants
    
    def _mutate_property_access(self, node: ast.Attribute) -> List[Mutant]:
        """Generate property access mutation mutants."""
        mutants = []
        
        attr_name = node.attr
        
        # Convert property access to method calls where applicable
        if attr_name in self.property_method_pairs:
            method_call = self.property_method_pairs[attr_name]
            mutant = Mutant(
                id=f"MCR_PROP_TO_METHOD_{node.lineno}_{attr_name}",
                mutation_type=self.get_mutation_type(),
                original_code=f"obj.{attr_name}",
                mutated_code=f"obj.{method_call}",
                line_number=node.lineno,
                column_start=node.col_offset,
                column_end=getattr(node, 'end_col_offset', node.col_offset + len(attr_name)),
                description=f"Convert property {attr_name} to method call at line {node.lineno}",
                severity="medium",
                language="python"
            )
            mutants.append(mutant)
        
        return mutants


class PythonLoopMutator(MutationOperator):
    """Enhanced Python loop construct mutations."""
    
    def __init__(self):
        # Iterator function mutations
        self.iterator_mutations = {
            'range': ['reversed', 'enumerate'],
            'enumerate': ['range', 'zip'],
            'zip': ['enumerate'],
            'reversed': ['sorted', 'range'],
            'sorted': ['reversed'],
            'iter': ['list', 'tuple'],
            'next': ['iter']
        }
        
        # Range parameter mutations
        self.range_mutations = {
            'increment_start': '+1',
            'decrement_start': '-1',
            'increment_stop': '+1',
            'decrement_stop': '-1',
            'increment_step': '+1',
            'decrement_step': '-1',
            'negate_step': '-step'
        }
    
    def get_mutation_type(self) -> MutationType:
        return MutationType.LOOP_MUTATION
    
    def is_applicable(self, node: Any) -> bool:
        return isinstance(node, (ast.For, ast.While, ast.Break, ast.Continue, ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp))
    
    def generate_mutants(self, source_code: str, filepath: str) -> List[Mutant]:
        """Generate enhanced loop-related mutants."""
        mutants = []
        
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.For):
                    mutants.extend(self._mutate_for_loop(node))
                    mutants.extend(self._mutate_loop_else(node))
                elif isinstance(node, ast.While):
                    mutants.extend(self._mutate_while_loop(node))
                    mutants.extend(self._mutate_loop_else(node))
                elif isinstance(node, ast.Break):
                    mutants.extend(self._mutate_break_statement(node))
                elif isinstance(node, ast.Continue):
                    mutants.extend(self._mutate_continue_statement(node))
                elif isinstance(node, ast.ListComp):
                    mutants.extend(self._mutate_list_comprehension(node))
                elif isinstance(node, ast.GeneratorExp):
                    mutants.extend(self._mutate_generator_expression(node))
                elif isinstance(node, (ast.SetComp, ast.DictComp)):
                    mutants.extend(self._mutate_comprehension(node))
                    
        except Exception as e:
            logger.error(f"Error generating loop mutants for {filepath}: {e}")
        
        return mutants
    
    def _mutate_for_loop(self, node: ast.For) -> List[Mutant]:
        """Enhanced mutation of for loops."""
        mutants = []
        
        # Enhanced range mutations
        if (isinstance(node.iter, ast.Call) and 
            isinstance(node.iter.func, ast.Name) and 
            node.iter.func.id == 'range'):
            
            mutants.extend(self._mutate_range_parameters(node))
        
        # Iterator function mutations
        if (isinstance(node.iter, ast.Call) and 
            isinstance(node.iter.func, ast.Name) and 
            node.iter.func.id in self.iterator_mutations):
            
            func_name = node.iter.func.id
            for replacement in self.iterator_mutations[func_name]:
                mutant = Mutant(
                    id=f"LOOP_ITER_SUB_{node.lineno}_{replacement}",
                    mutation_type=self.get_mutation_type(),
                    original_code=func_name,
                    mutated_code=replacement,
                    line_number=node.lineno,
                    column_start=node.iter.func.col_offset,
                    column_end=node.iter.func.col_offset + len(func_name),
                    description=f"Replace iterator function {func_name} with {replacement} at line {node.lineno}",
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
    
    def _mutate_range_parameters(self, node: ast.For) -> List[Mutant]:
        """Generate enhanced range parameter mutations."""
        mutants = []
        
        if (isinstance(node.iter, ast.Call) and 
            isinstance(node.iter.func, ast.Name) and 
            node.iter.func.id == 'range' and node.iter.args):
            
            # Mutate start parameter (if 2+ args)
            if len(node.iter.args) >= 2:
                start_arg = node.iter.args[0]
                if isinstance(start_arg, ast.Constant) and isinstance(start_arg.value, int):
                    mutant = Mutant(
                        id=f"LOOP_RANGE_START_{node.lineno}",
                        mutation_type=self.get_mutation_type(),
                        original_code=str(start_arg.value),
                        mutated_code=str(start_arg.value + 1),
                        line_number=node.lineno,
                        column_start=start_arg.col_offset,
                        column_end=getattr(start_arg, 'end_col_offset', start_arg.col_offset + len(str(start_arg.value))),
                        description=f"Increment range start parameter at line {node.lineno}",
                        severity="medium",
                        language="python"
                    )
                    mutants.append(mutant)
            
            # Mutate stop parameter
            stop_arg_index = 1 if len(node.iter.args) >= 2 else 0
            if stop_arg_index < len(node.iter.args):
                stop_arg = node.iter.args[stop_arg_index]
                if isinstance(stop_arg, ast.Constant) and isinstance(stop_arg.value, int):
                    mutant = Mutant(
                        id=f"LOOP_RANGE_STOP_{node.lineno}",
                        mutation_type=self.get_mutation_type(),
                        original_code=str(stop_arg.value),
                        mutated_code=str(stop_arg.value - 1),
                        line_number=node.lineno,
                        column_start=stop_arg.col_offset,
                        column_end=getattr(stop_arg, 'end_col_offset', stop_arg.col_offset + len(str(stop_arg.value))),
                        description=f"Decrement range stop parameter at line {node.lineno}",
                        severity="medium",
                        language="python"
                    )
                    mutants.append(mutant)
            
            # Mutate step parameter (if 3 args)
            if len(node.iter.args) >= 3:
                step_arg = node.iter.args[2]
                if isinstance(step_arg, ast.Constant) and isinstance(step_arg.value, int):
                    mutant = Mutant(
                        id=f"LOOP_RANGE_STEP_{node.lineno}",
                        mutation_type=self.get_mutation_type(),
                        original_code=str(step_arg.value),
                        mutated_code=str(-step_arg.value),
                        line_number=node.lineno,
                        column_start=step_arg.col_offset,
                        column_end=getattr(step_arg, 'end_col_offset', step_arg.col_offset + len(str(step_arg.value))),
                        description=f"Negate range step parameter at line {node.lineno}",
                        severity="high",
                        language="python"
                    )
                    mutants.append(mutant)
        
        return mutants
    
    def _mutate_loop_else(self, node) -> List[Mutant]:
        """Mutate loop else clauses."""
        mutants = []
        
        if node.orelse:  # Has else clause
            # Remove else clause
            mutant = Mutant(
                id=f"LOOP_ELSE_REMOVE_{node.lineno}",
                mutation_type=self.get_mutation_type(),
                original_code="else:",
                mutated_code="# else removed",
                line_number=node.lineno + len(node.body),
                column_start=0,
                column_end=5,
                description=f"Remove loop else clause at line {node.lineno}",
                severity="medium",
                language="python"
            )
            mutants.append(mutant)
        
        return mutants
    
    def _mutate_list_comprehension(self, node: ast.ListComp) -> List[Mutant]:
        """Mutate list comprehensions."""
        mutants = []
        
        # Convert list comprehension to generator expression
        mutant = Mutant(
            id=f"LOOP_LISTCOMP_TO_GEN_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="[expr for ...]",
            mutated_code="(expr for ...)",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=getattr(node, 'end_col_offset', node.col_offset + 10),
            description=f"Convert list comprehension to generator expression at line {node.lineno}",
            severity="medium",
            language="python"
        )
        mutants.append(mutant)
        
        # Mutate comprehension condition if present
        for generator in node.generators:
            if generator.ifs:
                for i, if_clause in enumerate(generator.ifs):
                    mutant = Mutant(
                        id=f"LOOP_LISTCOMP_COND_{node.lineno}_{i}",
                        mutation_type=self.get_mutation_type(),
                        original_code="if condition",
                        mutated_code="if not condition",
                        line_number=node.lineno,
                        column_start=node.col_offset,
                        column_end=getattr(node, 'end_col_offset', node.col_offset + 10),
                        description=f"Negate list comprehension condition at line {node.lineno}",
                        severity="high",
                        language="python"
                    )
                    mutants.append(mutant)
        
        return mutants
    
    def _mutate_generator_expression(self, node: ast.GeneratorExp) -> List[Mutant]:
        """Mutate generator expressions."""
        mutants = []
        
        # Convert generator expression to list comprehension
        mutant = Mutant(
            id=f"LOOP_GEN_TO_LIST_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="(expr for ...)",
            mutated_code="[expr for ...]",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=getattr(node, 'end_col_offset', node.col_offset + 10),
            description=f"Convert generator expression to list comprehension at line {node.lineno}",
            severity="medium",
            language="python"
        )
        mutants.append(mutant)
        
        return mutants
    
    def _mutate_comprehension(self, node) -> List[Mutant]:
        """Mutate set/dict comprehensions."""
        mutants = []
        
        comp_type = "set" if isinstance(node, ast.SetComp) else "dict"
        
        # Convert to list comprehension
        mutant = Mutant(
            id=f"LOOP_{comp_type.upper()}COMP_TO_LIST_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code=f"{{{comp_type} comprehension}}",
            mutated_code="[list comprehension]",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=getattr(node, 'end_col_offset', node.col_offset + 10),
            description=f"Convert {comp_type} comprehension to list comprehension at line {node.lineno}",
            severity="medium",
            language="python"
        )
        mutants.append(mutant)
        
        return mutants


class PythonConditionalMutator(MutationOperator):
    """Enhanced Python conditional statement mutations."""
    
    def __init__(self):
        # Boolean operator mutations for short-circuit evaluation
        self.boolean_operators = {
            'and': ['or'],
            'or': ['and']
        }
        
        # Comparison operator mutations
        self.comparison_mutations = {
            '==': ['!=', 'is', 'is not'],
            '!=': ['==', 'is', 'is not'],
            '<': ['<=', '>', '>='],
            '<=': ['<', '>', '>='],
            '>': ['>=', '<', '<='],
            '>=': ['>', '<', '<='],
            'in': ['not in'],
            'not in': ['in'],
            'is': ['is not', '=='],
            'is not': ['is', '!=']
        }
        
        # Truthiness test mutations
        self.truthiness_mutations = {
            'bool_explicit': 'Add explicit bool() cast',
            'not_negation': 'Add not negation',
            'double_negation': 'Add double negation (not not)',
            'identity_check': 'Replace with is None/is not None'
        }
    
    def get_mutation_type(self) -> MutationType:
        return MutationType.CONDITIONAL
    
    def is_applicable(self, node: Any) -> bool:
        return isinstance(node, (ast.If, ast.IfExp, ast.BoolOp, ast.Compare, ast.UnaryOp, ast.Match)) or \
               (hasattr(ast, 'Match') and isinstance(node, ast.Match))
    
    def generate_mutants(self, source_code: str, filepath: str) -> List[Mutant]:
        """Generate enhanced conditional mutants."""
        mutants = []
        
        try:
            tree = ast.parse(source_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    mutants.extend(self._mutate_if_statement(node))
                    mutants.extend(self._mutate_truthiness_tests(node))
                elif isinstance(node, ast.IfExp):
                    mutants.extend(self._mutate_ternary_operator(node))
                elif isinstance(node, ast.BoolOp):
                    mutants.extend(self._mutate_boolean_operators(node))
                elif isinstance(node, ast.Compare):
                    mutants.extend(self._mutate_comparison_operators(node))
                elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                    mutants.extend(self._mutate_not_operator(node))
                elif hasattr(ast, 'Match') and isinstance(node, ast.Match):
                    mutants.extend(self._mutate_match_statement(node))
                    
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
        """Enhanced mutation of ternary operators (conditional expressions)."""
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
        
        # Swap branches (body and orelse)
        mutant = Mutant(
            id=f"TERN_SWAP_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="x if condition else y",
            mutated_code="y if condition else x",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=getattr(node, 'end_col_offset', node.col_offset + 10),
            description=f"Swap ternary operator branches at line {node.lineno}",
            severity="high",
            language="python"
        )
        mutants.append(mutant)
        
        # Force True branch
        mutant = Mutant(
            id=f"TERN_TRUE_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="x if condition else y",
            mutated_code="x if True else y",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=getattr(node, 'end_col_offset', node.col_offset + 10),
            description=f"Force ternary condition to True at line {node.lineno}",
            severity="high",
            language="python"
        )
        mutants.append(mutant)
        
        # Force False branch
        mutant = Mutant(
            id=f"TERN_FALSE_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="x if condition else y",
            mutated_code="x if False else y",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=getattr(node, 'end_col_offset', node.col_offset + 10),
            description=f"Force ternary condition to False at line {node.lineno}",
            severity="high",
            language="python"
        )
        mutants.append(mutant)
        
        return mutants
    
    def _mutate_truthiness_tests(self, node: ast.If) -> List[Mutant]:
        """Mutate truthiness tests in if statements."""
        mutants = []
        
        # Add explicit bool() cast
        mutant = Mutant(
            id=f"COND_BOOL_CAST_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="if condition:",
            mutated_code="if bool(condition):",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 2,
            description=f"Add explicit bool() cast to condition at line {node.lineno}",
            severity="low",
            language="python"
        )
        mutants.append(mutant)
        
        # Add identity check with None
        mutant = Mutant(
            id=f"COND_IS_NONE_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="if condition:",
            mutated_code="if condition is None:",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 2,
            description=f"Replace truthiness with 'is None' check at line {node.lineno}",
            severity="medium",
            language="python"
        )
        mutants.append(mutant)
        
        # Add identity check with not None
        mutant = Mutant(
            id=f"COND_IS_NOT_NONE_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="if condition:",
            mutated_code="if condition is not None:",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 2,
            description=f"Replace truthiness with 'is not None' check at line {node.lineno}",
            severity="medium",
            language="python"
        )
        mutants.append(mutant)
        
        return mutants
    
    def _mutate_boolean_operators(self, node: ast.BoolOp) -> List[Mutant]:
        """Mutate boolean operators for short-circuit evaluation."""
        mutants = []
        
        op_name = 'and' if isinstance(node.op, ast.And) else 'or'
        
        if op_name in self.boolean_operators:
            for replacement in self.boolean_operators[op_name]:
                mutant = Mutant(
                    id=f"BOOL_OP_{op_name}_TO_{replacement}_{node.lineno}",
                    mutation_type=self.get_mutation_type(),
                    original_code=op_name,
                    mutated_code=replacement,
                    line_number=node.lineno,
                    column_start=node.col_offset,
                    column_end=getattr(node, 'end_col_offset', node.col_offset + len(op_name)),
                    description=f"Replace boolean operator '{op_name}' with '{replacement}' at line {node.lineno}",
                    severity="high",
                    language="python"
                )
                mutants.append(mutant)
        
        return mutants
    
    def _mutate_comparison_operators(self, node: ast.Compare) -> List[Mutant]:
        """Mutate comparison operators."""
        mutants = []
        
        for i, op in enumerate(node.ops):
            op_str = self._get_comparison_op_string(op)
            if op_str in self.comparison_mutations:
                for replacement in self.comparison_mutations[op_str]:
                    mutant = Mutant(
                        id=f"COMP_OP_{op_str}_TO_{replacement}_{node.lineno}_{i}",
                        mutation_type=self.get_mutation_type(),
                        original_code=op_str,
                        mutated_code=replacement,
                        line_number=node.lineno,
                        column_start=node.col_offset,
                        column_end=getattr(node, 'end_col_offset', node.col_offset + len(op_str)),
                        description=f"Replace comparison operator '{op_str}' with '{replacement}' at line {node.lineno}",
                        severity="medium",
                        language="python"
                    )
                    mutants.append(mutant)
        
        return mutants
    
    def _mutate_not_operator(self, node: ast.UnaryOp) -> List[Mutant]:
        """Mutate not operators."""
        mutants = []
        
        # Remove not operator
        mutant = Mutant(
            id=f"NOT_REMOVE_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="not condition",
            mutated_code="condition",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 3,
            description=f"Remove not operator at line {node.lineno}",
            severity="high",
            language="python"
        )
        mutants.append(mutant)
        
        # Add double negation
        mutant = Mutant(
            id=f"NOT_DOUBLE_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="not condition",
            mutated_code="not not condition",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 3,
            description=f"Add double negation at line {node.lineno}",
            severity="low",
            language="python"
        )
        mutants.append(mutant)
        
        return mutants
    
    def _mutate_match_statement(self, node) -> List[Mutant]:
        """Mutate match/case statements (Python 3.10+)."""
        mutants = []
        
        # Remove match statement (convert to if-elif chain)
        mutant = Mutant(
            id=f"MATCH_TO_IF_{node.lineno}",
            mutation_type=self.get_mutation_type(),
            original_code="match value:",
            mutated_code="if True:  # match converted to if",
            line_number=node.lineno,
            column_start=node.col_offset,
            column_end=node.col_offset + 5,
            description=f"Convert match statement to if statement at line {node.lineno}",
            severity="medium",
            language="python"
        )
        mutants.append(mutant)
        
        return mutants
    
    def _get_comparison_op_string(self, op) -> str:
        """Convert AST comparison operator to string."""
        op_map = {
            ast.Eq: '==',
            ast.NotEq: '!=',
            ast.Lt: '<',
            ast.LtE: '<=',
            ast.Gt: '>',
            ast.GtE: '>=',
            ast.Is: 'is',
            ast.IsNot: 'is not',
            ast.In: 'in',
            ast.NotIn: 'not in'
        }
        return op_map.get(type(op), str(type(op)))


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
import pytest
import ast
from unittest.mock import Mock, patch, MagicMock

from smart_test_generator.analysis.python_analyzers import (
    PythonIdiomAnalyzer,
    PythonExceptionMutator,
    PythonMethodCallMutator,
    PythonLoopMutator,
    PythonConditionalMutator,
    get_python_quality_analyzers,
    get_python_mutation_operators
)
from smart_test_generator.models.data_models import (
    QualityDimension, QualityScore, MutationType, Mutant
)


class TestPythonIdiomAnalyzer:
    """Test PythonIdiomAnalyzer class."""
    
    def test_init_sets_up_patterns_correctly(self):
        """Test that __init__ initializes pattern dictionaries correctly."""
        # Act
        analyzer = PythonIdiomAnalyzer()
        
        # Assert
        assert hasattr(analyzer, 'python_test_patterns')
        assert hasattr(analyzer, 'anti_patterns')
        assert 'pytest_fixtures' in analyzer.python_test_patterns
        assert 'mock_usage' in analyzer.python_test_patterns
        assert 'unittest_in_pytest' in analyzer.anti_patterns
        assert 'print_debugging' in analyzer.anti_patterns
    
    def test_get_dimension_returns_readability(self):
        """Test that get_dimension returns READABILITY dimension."""
        # Arrange
        analyzer = PythonIdiomAnalyzer()
        
        # Act
        dimension = analyzer.get_dimension()
        
        # Assert
        assert dimension == QualityDimension.READABILITY
    
    def test_analyze_with_modern_pytest_patterns(self):
        """Test analyze method with modern pytest patterns."""
        # Arrange
        analyzer = PythonIdiomAnalyzer()
        test_code = """
@pytest.fixture
def sample_data():
    return {'key': 'value'}

@pytest.mark.parametrize('input,expected', [(1, 2), (2, 3)])
def test_function(input, expected):
    with pytest.raises(ValueError):
        raise ValueError()
    assert result == expected
"""
        
        # Act
        result = analyzer.analyze(test_code)
        
        # Assert
        assert isinstance(result, QualityScore)
        assert result.dimension == QualityDimension.READABILITY
        assert result.score > 0
        assert result.details['uses_modern_pytest'] is True
        assert 'pytest_fixtures_count' in result.details
        assert 'pytest_parametrize_count' in result.details
    
    def test_analyze_with_anti_patterns(self):
        """Test analyze method detects anti-patterns."""
        # Arrange
        analyzer = PythonIdiomAnalyzer()
        test_code = """
from unittest import TestCase

def test_bad_practice():
    print("debugging")
    try:
        pass
    except:
        pass
    time.sleep(1)
"""
        
        # Act
        result = analyzer.analyze(test_code)
        
        # Assert
        assert isinstance(result, QualityScore)
        assert len(result.suggestions) > 0
        assert result.details['has_unittest_in_pytest'] is True
        assert result.details['has_print_debugging'] is True
        assert result.details['has_bare_except'] is True
    
    def test_analyze_with_empty_code(self):
        """Test analyze method with empty test code."""
        # Arrange
        analyzer = PythonIdiomAnalyzer()
        test_code = ""
        
        # Act
        result = analyzer.analyze(test_code)
        
        # Assert
        assert isinstance(result, QualityScore)
        assert result.score >= 0
        assert result.details['uses_modern_pytest'] is False
    
    def test_analyze_with_mock_patterns(self):
        """Test analyze method recognizes mock usage patterns."""
        # Arrange
        analyzer = PythonIdiomAnalyzer()
        test_code = """
from unittest.mock import Mock, patch

def test_with_mocks():
    mock_obj = Mock()
    with patch('module.function') as mock_func:
        mock_func.assert_called_once()
        assert mock_obj.call_count == 1
"""
        
        # Act
        result = analyzer.analyze(test_code)
        
        # Assert
        assert result.details['uses_modern_pytest'] is True
        assert 'mock_usage_count' in result.details
        assert result.details['mock_usage_count'] > 0


class TestPythonExceptionMutator:
    """Test PythonExceptionMutator class."""
    
    def test_get_mutation_type_returns_exception_handling(self):
        """Test that get_mutation_type returns EXCEPTION_HANDLING."""
        # Arrange
        mutator = PythonExceptionMutator()
        
        # Act
        mutation_type = mutator.get_mutation_type()
        
        # Assert
        assert mutation_type == MutationType.EXCEPTION_HANDLING
    
    def test_is_applicable_with_try_node(self):
        """Test is_applicable returns True for Try nodes."""
        # Arrange
        mutator = PythonExceptionMutator()
        try_node = ast.Try(body=[], handlers=[], orelse=[], finalbody=[])
        
        # Act
        result = mutator.is_applicable(try_node)
        
        # Assert
        assert result is True
    
    def test_is_applicable_with_raise_node(self):
        """Test is_applicable returns True for Raise nodes."""
        # Arrange
        mutator = PythonExceptionMutator()
        raise_node = ast.Raise(exc=None, cause=None)
        
        # Act
        result = mutator.is_applicable(raise_node)
        
        # Assert
        assert result is True
    
    def test_is_applicable_with_except_handler_node(self):
        """Test is_applicable returns True for ExceptHandler nodes."""
        # Arrange
        mutator = PythonExceptionMutator()
        except_node = ast.ExceptHandler(type=None, name=None, body=[])
        
        # Act
        result = mutator.is_applicable(except_node)
        
        # Assert
        assert result is True
    
    def test_is_applicable_with_other_node(self):
        """Test is_applicable returns False for other node types."""
        # Arrange
        mutator = PythonExceptionMutator()
        assign_node = ast.Assign(targets=[], value=ast.Constant(value=1))
        
        # Act
        result = mutator.is_applicable(assign_node)
        
        # Assert
        assert result is False
    
    def test_generate_mutants_with_try_block(self):
        """Test generate_mutants creates mutants for try blocks."""
        # Arrange
        mutator = PythonExceptionMutator()
        source_code = """
try:
    risky_operation()
except ValueError:
    handle_error()
"""
        
        # Act
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Assert
        assert len(mutants) > 0
        try_mutants = [m for m in mutants if 'TRY' in m.id]
        assert len(try_mutants) > 0
        assert all(m.mutation_type == MutationType.EXCEPTION_HANDLING for m in try_mutants)
    
    def test_generate_mutants_with_raise_statement(self):
        """Test generate_mutants creates mutants for raise statements."""
        # Arrange
        mutator = PythonExceptionMutator()
        source_code = """
def function():
    if condition:
        raise ValueError("error")
"""
        
        # Act
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Assert
        assert len(mutants) > 0
        raise_mutants = [m for m in mutants if 'RAISE' in m.id]
        assert len(raise_mutants) > 0
        assert all(m.mutation_type == MutationType.EXCEPTION_HANDLING for m in raise_mutants)
    
    def test_generate_mutants_with_invalid_syntax(self):
        """Test generate_mutants handles invalid syntax gracefully."""
        # Arrange
        mutator = PythonExceptionMutator()
        source_code = "invalid python syntax {"
        
        # Act
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Assert
        assert mutants == []


class TestPythonMethodCallMutator:
    """Test PythonMethodCallMutator class."""
    
    def test_init_sets_up_method_mutations(self):
        """Test that __init__ initializes method_mutations dictionary."""
        # Act
        mutator = PythonMethodCallMutator()
        
        # Assert
        assert hasattr(mutator, 'method_mutations')
        assert 'append' in mutator.method_mutations
        assert 'extend' in mutator.method_mutations['append']
        assert 'get' in mutator.method_mutations
        assert 'upper' in mutator.method_mutations
    
    def test_get_mutation_type_returns_method_call(self):
        """Test that get_mutation_type returns METHOD_CALL."""
        # Arrange
        mutator = PythonMethodCallMutator()
        
        # Act
        mutation_type = mutator.get_mutation_type()
        
        # Assert
        assert mutation_type == MutationType.METHOD_CALL
    
    def test_is_applicable_with_method_call(self):
        """Test is_applicable returns True for method calls."""
        # Arrange
        mutator = PythonMethodCallMutator()
        # Create a method call node: obj.method()
        call_node = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='obj', ctx=ast.Load()),
                attr='append',
                ctx=ast.Load()
            ),
            args=[],
            keywords=[]
        )
        
        # Act
        result = mutator.is_applicable(call_node)
        
        # Assert
        assert result is True
    
    def test_is_applicable_with_function_call(self):
        """Test is_applicable returns False for function calls."""
        # Arrange
        mutator = PythonMethodCallMutator()
        # Create a function call node: function()
        call_node = ast.Call(
            func=ast.Name(id='function', ctx=ast.Load()),
            args=[],
            keywords=[]
        )
        
        # Act
        result = mutator.is_applicable(call_node)
        
        # Assert
        assert result is False
    
    def test_generate_mutants_with_list_methods(self):
        """Test generate_mutants creates mutants for list methods."""
        # Arrange
        mutator = PythonMethodCallMutator()
        source_code = """
my_list = [1, 2, 3]
my_list.append(4)
my_list.extend([5, 6])
"""
        
        # Act
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Assert
        assert len(mutants) > 0
        append_mutants = [m for m in mutants if 'append' in m.original_code]
        extend_mutants = [m for m in mutants if 'extend' in m.original_code]
        assert len(append_mutants) > 0
        assert len(extend_mutants) > 0
        assert all(m.mutation_type == MutationType.METHOD_CALL for m in mutants)
    
    def test_generate_mutants_with_string_methods(self):
        """Test generate_mutants creates mutants for string methods."""
        # Arrange
        mutator = PythonMethodCallMutator()
        source_code = """
text = "Hello World"
result = text.upper()
stripped = text.strip()
"""
        
        # Act
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Assert
        assert len(mutants) > 0
        upper_mutants = [m for m in mutants if 'upper' in m.original_code]
        strip_mutants = [m for m in mutants if 'strip' in m.original_code]
        assert len(upper_mutants) > 0
        assert len(strip_mutants) > 0


class TestPythonLoopMutator:
    """Test PythonLoopMutator class."""
    
    def test_get_mutation_type_returns_loop_mutation(self):
        """Test that get_mutation_type returns LOOP_MUTATION."""
        # Arrange
        mutator = PythonLoopMutator()
        
        # Act
        mutation_type = mutator.get_mutation_type()
        
        # Assert
        assert mutation_type == MutationType.LOOP_MUTATION
    
    def test_is_applicable_with_for_loop(self):
        """Test is_applicable returns True for For loops."""
        # Arrange
        mutator = PythonLoopMutator()
        for_node = ast.For(
            target=ast.Name(id='i', ctx=ast.Store()),
            iter=ast.Name(id='items', ctx=ast.Load()),
            body=[],
            orelse=[]
        )
        
        # Act
        result = mutator.is_applicable(for_node)
        
        # Assert
        assert result is True
    
    def test_is_applicable_with_while_loop(self):
        """Test is_applicable returns True for While loops."""
        # Arrange
        mutator = PythonLoopMutator()
        while_node = ast.While(
            test=ast.Constant(value=True),
            body=[],
            orelse=[]
        )
        
        # Act
        result = mutator.is_applicable(while_node)
        
        # Assert
        assert result is True
    
    def test_is_applicable_with_break_statement(self):
        """Test is_applicable returns True for Break statements."""
        # Arrange
        mutator = PythonLoopMutator()
        break_node = ast.Break()
        
        # Act
        result = mutator.is_applicable(break_node)
        
        # Assert
        assert result is True
    
    def test_is_applicable_with_continue_statement(self):
        """Test is_applicable returns True for Continue statements."""
        # Arrange
        mutator = PythonLoopMutator()
        continue_node = ast.Continue()
        
        # Act
        result = mutator.is_applicable(continue_node)
        
        # Assert
        assert result is True
    
    def test_generate_mutants_with_for_loop(self):
        """Test generate_mutants creates mutants for for loops."""
        # Arrange
        mutator = PythonLoopMutator()
        source_code = """
for i in range(10):
    print(i)
"""
        
        # Act
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Assert
        assert len(mutants) > 0
        range_mutants = [m for m in mutants if 'RANGE' in m.id]
        assert len(range_mutants) > 0
        assert all(m.mutation_type == MutationType.LOOP_MUTATION for m in mutants)
    
    def test_generate_mutants_with_while_loop(self):
        """Test generate_mutants creates mutants for while loops."""
        # Arrange
        mutator = PythonLoopMutator()
        source_code = """
while condition:
    do_something()
"""
        
        # Act
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Assert
        assert len(mutants) > 0
        while_mutants = [m for m in mutants if 'WHILE' in m.id]
        assert len(while_mutants) > 0
        assert all(m.mutation_type == MutationType.LOOP_MUTATION for m in mutants)
    
    def test_generate_mutants_with_break_continue(self):
        """Test generate_mutants creates mutants for break and continue."""
        # Arrange
        mutator = PythonLoopMutator()
        source_code = """
for i in range(10):
    if i == 5:
        break
    if i == 3:
        continue
    print(i)
"""
        
        # Act
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Assert
        assert len(mutants) > 0
        break_mutants = [m for m in mutants if 'BREAK' in m.id]
        continue_mutants = [m for m in mutants if 'CONT' in m.id]
        assert len(break_mutants) > 0
        assert len(continue_mutants) > 0


class TestPythonConditionalMutator:
    """Test PythonConditionalMutator class."""
    
    def test_get_mutation_type_returns_conditional(self):
        """Test that get_mutation_type returns CONDITIONAL."""
        # Arrange
        mutator = PythonConditionalMutator()
        
        # Act
        mutation_type = mutator.get_mutation_type()
        
        # Assert
        assert mutation_type == MutationType.CONDITIONAL
    
    def test_is_applicable_with_if_statement(self):
        """Test is_applicable returns True for If statements."""
        # Arrange
        mutator = PythonConditionalMutator()
        if_node = ast.If(
            test=ast.Constant(value=True),
            body=[],
            orelse=[]
        )
        
        # Act
        result = mutator.is_applicable(if_node)
        
        # Assert
        assert result is True
    
    def test_is_applicable_with_ternary_operator(self):
        """Test is_applicable returns True for ternary operators."""
        # Arrange
        mutator = PythonConditionalMutator()
        ifexp_node = ast.IfExp(
            test=ast.Constant(value=True),
            body=ast.Constant(value=1),
            orelse=ast.Constant(value=2)
        )
        
        # Act
        result = mutator.is_applicable(ifexp_node)
        
        # Assert
        assert result is True
    
    def test_is_applicable_with_other_node(self):
        """Test is_applicable returns False for other node types."""
        # Arrange
        mutator = PythonConditionalMutator()
        assign_node = ast.Assign(targets=[], value=ast.Constant(value=1))
        
        # Act
        result = mutator.is_applicable(assign_node)
        
        # Assert
        assert result is False
    
    def test_generate_mutants_with_if_statement(self):
        """Test generate_mutants creates mutants for if statements."""
        # Arrange
        mutator = PythonConditionalMutator()
        source_code = """
if condition:
    do_something()
else:
    do_something_else()
"""
        
        # Act
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Assert
        assert len(mutants) > 0
        neg_mutants = [m for m in mutants if 'NEG' in m.id]
        true_mutants = [m for m in mutants if 'TRUE' in m.id]
        false_mutants = [m for m in mutants if 'FALSE' in m.id]
        assert len(neg_mutants) > 0
        assert len(true_mutants) > 0
        assert len(false_mutants) > 0
        assert all(m.mutation_type == MutationType.CONDITIONAL for m in mutants)
    
    def test_generate_mutants_with_ternary_operator(self):
        """Test generate_mutants creates mutants for ternary operators."""
        # Arrange
        mutator = PythonConditionalMutator()
        source_code = """
result = value if condition else default
"""
        
        # Act
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Assert
        assert len(mutants) > 0
        ternary_mutants = [m for m in mutants if 'TERN' in m.id]
        assert len(ternary_mutants) > 0
        assert all(m.mutation_type == MutationType.CONDITIONAL for m in mutants)
    
    def test_generate_mutants_with_invalid_syntax(self):
        """Test generate_mutants handles invalid syntax gracefully."""
        # Arrange
        mutator = PythonConditionalMutator()
        source_code = "invalid python syntax {"
        
        # Act
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Assert
        assert mutants == []


class TestGetPythonQualityAnalyzers:
    """Test get_python_quality_analyzers function."""
    
    @patch('smart_test_generator.analysis.quality_analyzer.EdgeCaseAnalyzer')
    @patch('smart_test_generator.analysis.quality_analyzer.AssertionStrengthAnalyzer')
    @patch('smart_test_generator.analysis.quality_analyzer.MaintainabilityAnalyzer')
    @patch('smart_test_generator.analysis.quality_analyzer.BugDetectionAnalyzer')
    def test_returns_all_quality_analyzers(self, mock_bug, mock_maint, mock_assert, mock_edge):
        """Test that function returns all quality analyzers including Python-specific ones."""
        # Arrange
        mock_edge.return_value = Mock()
        mock_assert.return_value = Mock()
        mock_maint.return_value = Mock()
        mock_bug.return_value = Mock()
        
        # Act
        analyzers = get_python_quality_analyzers()
        
        # Assert
        assert len(analyzers) == 5
        assert any(isinstance(analyzer, PythonIdiomAnalyzer) for analyzer in analyzers)
        mock_edge.assert_called_once()
        mock_assert.assert_called_once()
        mock_maint.assert_called_once()
        mock_bug.assert_called_once()
    
    def test_returns_list_of_analyzers(self):
        """Test that function returns a list of analyzer instances."""
        # Act
        analyzers = get_python_quality_analyzers()
        
        # Assert
        assert isinstance(analyzers, list)
        assert len(analyzers) > 0
        # Check that PythonIdiomAnalyzer is included
        python_analyzers = [a for a in analyzers if isinstance(a, PythonIdiomAnalyzer)]
        assert len(python_analyzers) == 1


class TestGetPythonMutationOperators:
    """Test get_python_mutation_operators function."""
    
    @patch('smart_test_generator.analysis.mutation_engine.ArithmeticOperatorMutator')
    @patch('smart_test_generator.analysis.mutation_engine.ComparisonOperatorMutator')
    @patch('smart_test_generator.analysis.mutation_engine.LogicalOperatorMutator')
    @patch('smart_test_generator.analysis.mutation_engine.ConstantValueMutator')
    @patch('smart_test_generator.analysis.mutation_engine.BoundaryValueMutator')
    def test_returns_all_mutation_operators(self, mock_boundary, mock_constant, mock_logical, mock_comparison, mock_arithmetic):
        """Test that function returns all mutation operators including Python-specific ones."""
        # Arrange
        mock_arithmetic.return_value = Mock()
        mock_comparison.return_value = Mock()
        mock_logical.return_value = Mock()
        mock_constant.return_value = Mock()
        mock_boundary.return_value = Mock()
        
        # Act
        operators = get_python_mutation_operators()
        
        # Assert
        assert len(operators) == 9
        # Check Python-specific mutators are included
        assert any(isinstance(op, PythonExceptionMutator) for op in operators)
        assert any(isinstance(op, PythonMethodCallMutator) for op in operators)
        assert any(isinstance(op, PythonLoopMutator) for op in operators)
        assert any(isinstance(op, PythonConditionalMutator) for op in operators)
        
        # Check base mutators are called
        mock_arithmetic.assert_called_once()
        mock_comparison.assert_called_once()
        mock_logical.assert_called_once()
        mock_constant.assert_called_once()
        mock_boundary.assert_called_once()
    
    def test_returns_list_of_operators(self):
        """Test that function returns a list of mutation operator instances."""
        # Act
        operators = get_python_mutation_operators()
        
        # Assert
        assert isinstance(operators, list)
        assert len(operators) > 0
        # Check that all Python-specific mutators are included
        python_exception = [op for op in operators if isinstance(op, PythonExceptionMutator)]
        python_method = [op for op in operators if isinstance(op, PythonMethodCallMutator)]
        python_loop = [op for op in operators if isinstance(op, PythonLoopMutator)]
        python_conditional = [op for op in operators if isinstance(op, PythonConditionalMutator)]
        
        assert len(python_exception) == 1
        assert len(python_method) == 1
        assert len(python_loop) == 1
        assert len(python_conditional) == 1
    
    def test_all_operators_have_required_methods(self):
        """Test that all returned operators have required mutation methods."""
        # Act
        operators = get_python_mutation_operators()
        
        # Assert
        for operator in operators:
            assert hasattr(operator, 'get_mutation_type')
            assert hasattr(operator, 'is_applicable')
            assert hasattr(operator, 'generate_mutants')
            assert callable(operator.get_mutation_type)
            assert callable(operator.is_applicable)
            assert callable(operator.generate_mutants)
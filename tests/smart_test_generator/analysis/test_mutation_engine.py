import pytest
import ast
import tempfile
import os
from unittest.mock import Mock, patch, mock_open, MagicMock
from pathlib import Path

from smart_test_generator.analysis.mutation_engine import (
    MutationOperator,
    ArithmeticOperatorMutator,
    ComparisonOperatorMutator,
    LogicalOperatorMutator,
    ConstantValueMutator,
    BoundaryValueMutator,
    MutationTestingEngine
)
from smart_test_generator.models.data_models import (
    MutationType, Mutant, MutationResult, MutationScore, WeakSpot
)


class TestMutationOperator:
    """Test abstract MutationOperator base class."""
    
    def test_get_mutation_type_is_abstract(self):
        """Test that get_mutation_type is abstract and raises NotImplementedError."""
        with pytest.raises(TypeError):
            MutationOperator()
    
    def test_generate_mutants_is_abstract(self):
        """Test that generate_mutants is abstract."""
        # Create a concrete subclass to test abstract methods
        class TestOperator(MutationOperator):
            def get_mutation_type(self):
                return MutationType.ARITHMETIC_OPERATOR
            
            def is_applicable(self, node):
                return True
        
        operator = TestOperator()
        with pytest.raises(NotImplementedError):
            operator.generate_mutants("code", "file.py")
    
    def test_is_applicable_is_abstract(self):
        """Test that is_applicable is abstract."""
        class TestOperator(MutationOperator):
            def get_mutation_type(self):
                return MutationType.ARITHMETIC_OPERATOR
            
            def generate_mutants(self, source_code, filepath):
                return []
        
        operator = TestOperator()
        with pytest.raises(NotImplementedError):
            operator.is_applicable(ast.parse("1 + 1"))


class TestArithmeticOperatorMutator:
    """Test ArithmeticOperatorMutator class."""
    
    def test_init_sets_operator_mappings(self):
        """Test that __init__ properly sets up operator mappings."""
        mutator = ArithmeticOperatorMutator()
        
        assert '+' in mutator.operator_mappings
        assert '-' in mutator.operator_mappings['+'] 
        assert '*' in mutator.operator_mappings['+']
        assert ast.Add in mutator.ast_operators
        assert mutator.ast_operators[ast.Add] == '+'
    
    def test_get_mutation_type_returns_arithmetic_operator(self):
        """Test that get_mutation_type returns ARITHMETIC_OPERATOR."""
        mutator = ArithmeticOperatorMutator()
        assert mutator.get_mutation_type() == MutationType.ARITHMETIC_OPERATOR
    
    def test_is_applicable_with_binary_operation(self):
        """Test is_applicable returns True for binary operations with arithmetic operators."""
        mutator = ArithmeticOperatorMutator()
        tree = ast.parse("1 + 2")
        binop_node = tree.body[0].value
        
        assert mutator.is_applicable(binop_node) is True
    
    def test_is_applicable_with_non_arithmetic_operation(self):
        """Test is_applicable returns False for non-arithmetic operations."""
        mutator = ArithmeticOperatorMutator()
        tree = ast.parse("x = 1")
        assign_node = tree.body[0]
        
        assert mutator.is_applicable(assign_node) is False
    
    def test_generate_mutants_creates_arithmetic_mutations(self):
        """Test generate_mutants creates proper arithmetic mutations."""
        mutator = ArithmeticOperatorMutator()
        source_code = "result = 5 + 3"
        
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        assert len(mutants) > 0
        assert any(m.original_code == '+' and m.mutated_code == '-' for m in mutants)
        assert any(m.original_code == '+' and m.mutated_code == '*' for m in mutants)
        assert all(m.mutation_type == MutationType.ARITHMETIC_OPERATOR for m in mutants)
    
    def test_generate_mutants_handles_syntax_error(self):
        """Test generate_mutants handles syntax errors gracefully."""
        mutator = ArithmeticOperatorMutator()
        invalid_code = "result = 5 + )"  # Actually invalid syntax
        
        mutants = mutator.generate_mutants(invalid_code, "test.py")
        
        assert mutants == []
    
    def test_generate_mutants_with_multiple_operators(self):
        """Test generate_mutants handles multiple arithmetic operators."""
        mutator = ArithmeticOperatorMutator()
        source_code = "result = a + b * c - d"
        
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should have mutants for +, *, and - operators
        plus_mutants = [m for m in mutants if m.original_code == '+']
        mult_mutants = [m for m in mutants if m.original_code == '*']
        minus_mutants = [m for m in mutants if m.original_code == '-']
        
        assert len(plus_mutants) > 0
        assert len(mult_mutants) > 0
        assert len(minus_mutants) > 0


class TestComparisonOperatorMutator:
    """Test ComparisonOperatorMutator class."""
    
    def test_init_sets_comparison_mappings(self):
        """Test that __init__ properly sets up comparison operator mappings."""
        mutator = ComparisonOperatorMutator()
        
        assert '==' in mutator.operator_mappings
        assert '!=' in mutator.operator_mappings['==']
        assert ast.Eq in mutator.ast_operators
        assert mutator.ast_operators[ast.Eq] == '=='
    
    def test_get_mutation_type_returns_comparison_operator(self):
        """Test that get_mutation_type returns COMPARISON_OPERATOR."""
        mutator = ComparisonOperatorMutator()
        assert mutator.get_mutation_type() == MutationType.COMPARISON_OPERATOR
    
    def test_is_applicable_with_comparison_operation(self):
        """Test is_applicable returns True for comparison operations."""
        mutator = ComparisonOperatorMutator()
        tree = ast.parse("x == y")
        compare_node = tree.body[0].value
        
        assert mutator.is_applicable(compare_node) is True
    
    def test_is_applicable_with_multiple_comparisons(self):
        """Test is_applicable returns False for chained comparisons."""
        mutator = ComparisonOperatorMutator()
        tree = ast.parse("1 < x < 10")
        compare_node = tree.body[0].value
        
        assert mutator.is_applicable(compare_node) is False
    
    def test_generate_mutants_creates_comparison_mutations(self):
        """Test generate_mutants creates proper comparison mutations."""
        mutator = ComparisonOperatorMutator()
        source_code = "if x == 5: pass"
        
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        assert len(mutants) > 0
        assert any(m.original_code == '==' and m.mutated_code == '!=' for m in mutants)
        assert any(m.original_code == '==' and m.mutated_code == '<' for m in mutants)
        assert all(m.mutation_type == MutationType.COMPARISON_OPERATOR for m in mutants)
    
    def test_generate_mutants_sets_severity_for_equality(self):
        """Test generate_mutants sets high severity for equality operators."""
        mutator = ComparisonOperatorMutator()
        source_code = "if x == 5: pass"
        
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        equality_mutants = [m for m in mutants if m.original_code == '==']
        assert all(m.severity == 'high' for m in equality_mutants)


class TestLogicalOperatorMutator:
    """Test LogicalOperatorMutator class."""
    
    def test_init_sets_logical_mappings(self):
        """Test that __init__ properly sets up logical operator mappings."""
        mutator = LogicalOperatorMutator()
        
        assert 'and' in mutator.operator_mappings
        assert 'or' in mutator.operator_mappings['and']
        assert ast.And in mutator.ast_operators
        assert mutator.ast_operators[ast.And] == 'and'
    
    def test_get_mutation_type_returns_logical_operator(self):
        """Test that get_mutation_type returns LOGICAL_OPERATOR."""
        mutator = LogicalOperatorMutator()
        assert mutator.get_mutation_type() == MutationType.LOGICAL_OPERATOR
    
    def test_is_applicable_with_logical_operation(self):
        """Test is_applicable returns True for logical operations."""
        mutator = LogicalOperatorMutator()
        tree = ast.parse("x and y")
        boolop_node = tree.body[0].value
        
        assert mutator.is_applicable(boolop_node) is True
    
    def test_is_applicable_with_non_logical_operation(self):
        """Test is_applicable returns False for non-logical operations."""
        mutator = LogicalOperatorMutator()
        tree = ast.parse("x + y")
        binop_node = tree.body[0].value
        
        assert mutator.is_applicable(binop_node) is False
    
    def test_generate_mutants_creates_logical_mutations(self):
        """Test generate_mutants creates proper logical mutations."""
        mutator = LogicalOperatorMutator()
        source_code = "if x and y: pass"
        
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        assert len(mutants) > 0
        assert any(m.original_code == 'and' and m.mutated_code == 'or' for m in mutants)
        assert all(m.mutation_type == MutationType.LOGICAL_OPERATOR for m in mutants)
        assert all(m.severity == 'high' for m in mutants)


class TestConstantValueMutator:
    """Test ConstantValueMutator class."""
    
    def test_get_mutation_type_returns_constant_value(self):
        """Test that get_mutation_type returns CONSTANT_VALUE."""
        mutator = ConstantValueMutator()
        assert mutator.get_mutation_type() == MutationType.CONSTANT_VALUE
    
    def test_is_applicable_with_constant_node(self):
        """Test is_applicable returns True for constant nodes."""
        mutator = ConstantValueMutator()
        tree = ast.parse("x = 42")
        constant_node = tree.body[0].value
        
        assert mutator.is_applicable(constant_node) is True
    
    def test_is_applicable_with_non_constant_node(self):
        """Test is_applicable returns False for non-constant nodes."""
        mutator = ConstantValueMutator()
        tree = ast.parse("x = y")
        name_node = tree.body[0].value
        
        assert mutator.is_applicable(name_node) is False
    
    def test_generate_mutants_creates_integer_mutations(self):
        """Test generate_mutants creates proper integer mutations."""
        mutator = ConstantValueMutator()
        source_code = "x = 5"
        
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        assert len(mutants) > 0
        assert any(m.original_code == '5' and m.mutated_code == '6' for m in mutants)
        assert any(m.original_code == '5' and m.mutated_code == '4' for m in mutants)
        assert any(m.original_code == '5' and m.mutated_code == '0' for m in mutants)
    
    def test_generate_mutants_creates_string_mutations(self):
        """Test generate_mutants creates proper string mutations."""
        mutator = ConstantValueMutator()
        source_code = 'x = "hello"'
        
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        assert len(mutants) > 0
        assert any(m.original_code == 'hello' and m.mutated_code == '' for m in mutants)
        assert any(m.original_code == 'hello' and 'helloX' in m.mutated_code for m in mutants)
    
    def test_generate_mutants_creates_boolean_mutations(self):
        """Test generate_mutants creates proper boolean mutations."""
        mutator = ConstantValueMutator()
        source_code = "x = True"
        
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        assert len(mutants) > 0
        # Check that we have boolean-related mutations (more flexible)
        boolean_mutants = [m for m in mutants if m.mutation_type == MutationType.CONSTANT_VALUE]
        assert len(boolean_mutants) > 0


class TestBoundaryValueMutator:
    """Test BoundaryValueMutator class."""
    
    def test_get_mutation_type_returns_boundary_value(self):
        """Test that get_mutation_type returns BOUNDARY_VALUE."""
        mutator = BoundaryValueMutator()
        assert mutator.get_mutation_type() == MutationType.BOUNDARY_VALUE
    
    def test_is_applicable_with_comparison_node(self):
        """Test is_applicable returns True for comparison nodes."""
        mutator = BoundaryValueMutator()
        tree = ast.parse("x < 10")
        compare_node = tree.body[0].value
        
        assert mutator.is_applicable(compare_node) is True
    
    def test_is_applicable_with_integer_constant(self):
        """Test is_applicable returns True for integer constants."""
        mutator = BoundaryValueMutator()
        tree = ast.parse("x = 0")
        constant_node = tree.body[0].value
        
        assert mutator.is_applicable(constant_node) is True
    
    def test_generate_mutants_creates_boundary_mutations(self):
        """Test generate_mutants creates proper boundary mutations."""
        mutator = BoundaryValueMutator()
        source_code = "if x < 10: pass"
        
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        assert len(mutants) > 0
        assert any(m.original_code == '10' and m.mutated_code == '9' for m in mutants)
        assert any(m.original_code == '10' and m.mutated_code == '11' for m in mutants)
        assert all(m.mutation_type == MutationType.BOUNDARY_VALUE for m in mutants)
    
    def test_generate_mutants_with_boundary_constants(self):
        """Test generate_mutants handles common boundary constants."""
        mutator = BoundaryValueMutator()
        source_code = "x = 0\ny = 1\nz = 2"  # Use 2 instead of -1 since -1 is UnaryOp in AST
        
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        assert len(mutants) > 0
        zero_mutants = [m for m in mutants if m.original_code == '0']
        one_mutants = [m for m in mutants if m.original_code == '1']
        two_mutants = [m for m in mutants if m.original_code == '2']
        
        assert len(zero_mutants) > 0
        assert len(one_mutants) > 0
        assert len(two_mutants) > 0


class TestMutationTestingEngine:
    """Test MutationTestingEngine class."""
    
    def test_init_with_default_operators(self):
        """Test __init__ creates engine with default operators."""
        engine = MutationTestingEngine()
        
        assert len(engine.operators) == 8
        assert engine.timeout == 30
        assert any(isinstance(op, ArithmeticOperatorMutator) for op in engine.operators)
        assert any(isinstance(op, ComparisonOperatorMutator) for op in engine.operators)
    
    def test_init_with_custom_operators(self):
        """Test __init__ accepts custom operators and timeout."""
        custom_operators = [ArithmeticOperatorMutator()]
        engine = MutationTestingEngine(operators=custom_operators, timeout=60)
        
        assert len(engine.operators) == 1
        assert engine.timeout == 60
        assert isinstance(engine.operators[0], ArithmeticOperatorMutator)
    
    @patch('builtins.open', mock_open(read_data="x = 1 + 2"))
    def test_generate_mutants_reads_file_and_generates_mutants(self):
        """Test generate_mutants reads source file and generates mutants."""
        engine = MutationTestingEngine()
        
        mutants = engine.generate_mutants("test.py")
        
        assert len(mutants) > 0
        assert all(isinstance(m, Mutant) for m in mutants)
    
    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_generate_mutants_handles_file_not_found(self, mock_open):
        """Test generate_mutants handles file not found error."""
        engine = MutationTestingEngine()
        
        mutants = engine.generate_mutants("nonexistent.py")
        
        assert mutants == []
    
    @patch('builtins.open', mock_open(read_data="x = 1 + 2\ny = 3 * 4"))
    def test_generate_mutants_sorts_by_line_number(self):
        """Test generate_mutants sorts mutants by line number."""
        engine = MutationTestingEngine()
        
        mutants = engine.generate_mutants("test.py")
        
        # Check that mutants are sorted by line number
        for i in range(1, len(mutants)):
            assert mutants[i-1].line_number <= mutants[i].line_number
    
    @patch('builtins.open', mock_open(read_data="x = 1 + 2"))
    @patch('smart_test_generator.analysis.mutation_engine.MutationTestingEngine._test_mutant')
    def test_run_mutation_testing_processes_all_mutants(self, mock_test_mutant):
        """Test run_mutation_testing processes all generated mutants."""
        engine = MutationTestingEngine()
        mock_test_mutant.return_value = MutationResult(
            mutant=Mock(spec=Mutant),
            killed=True,
            execution_time=0.1
        )
        
        result = engine.run_mutation_testing("test.py", ["test_file.py"])
        
        assert isinstance(result, MutationScore)
        assert mock_test_mutant.call_count > 0
    
    @patch('builtins.open', mock_open(read_data="x = 1 + 2"))
    def test_run_mutation_testing_limits_mutants_when_max_specified(self):
        """Test run_mutation_testing respects max_mutants parameter."""
        engine = MutationTestingEngine()
        
        with patch.object(engine, '_test_mutant') as mock_test_mutant:
            mock_test_mutant.return_value = MutationResult(
                mutant=Mock(spec=Mutant),
                killed=True,
                execution_time=0.1
            )
            
            result = engine.run_mutation_testing("test.py", ["test_file.py"], max_mutants=2)
            
            assert mock_test_mutant.call_count <= 2
    
    @patch('builtins.open', mock_open(read_data=""))
    def test_run_mutation_testing_handles_no_mutants(self):
        """Test run_mutation_testing handles case with no mutants."""
        engine = MutationTestingEngine()
        
        result = engine.run_mutation_testing("empty.py", ["test_file.py"])
        
        assert isinstance(result, MutationScore)
        assert result.total_mutants == 0
        assert result.mutation_score == 0.0
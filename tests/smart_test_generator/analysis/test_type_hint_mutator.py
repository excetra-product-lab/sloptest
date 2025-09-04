"""Tests for TypeHintMutator - type hint mutations for modern Python."""

import ast
import pytest
from smart_test_generator.analysis.type_hint_mutator import TypeHintMutator
from smart_test_generator.models.data_models import MutationType


class TestTypeHintMutator:
    """Test the TypeHintMutator for type hint mutations."""
    
    @pytest.fixture
    def mutator(self):
        """Create a TypeHintMutator instance."""
        return TypeHintMutator()
    
    def test_get_mutation_type(self, mutator):
        """Test that the mutator returns correct mutation type."""
        assert mutator.get_mutation_type() == MutationType.TYPE_HINT
    
    def test_is_applicable(self, mutator):
        """Test that mutator is applicable to correct node types."""
        # Should be applicable to function definitions
        func_node = ast.parse("def func(x: int) -> str: pass").body[0]
        assert mutator.is_applicable(func_node)
        
        # Should be applicable to annotated assignments
        ann_assign_node = ast.parse("x: int = 5").body[0]
        assert mutator.is_applicable(ann_assign_node)
        
        # Should be applicable to class definitions
        class_node = ast.parse("@dataclass\nclass Test: pass").body[0]
        assert mutator.is_applicable(class_node)
        
        # Should not be applicable to simple assignments
        assign_node = ast.parse("x = 5").body[0]
        assert not mutator.is_applicable(assign_node)


class TestOptionalTypeMutations:
    """Test Optional[T] type mutations."""
    
    @pytest.fixture
    def mutator(self):
        return TypeHintMutator()
    
    def test_optional_function_parameter_mutation(self, mutator):
        """Test mutation of Optional parameters in function definitions."""
        source_code = '''
def process_data(value: Optional[str]) -> None:
    pass
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate mutations to remove Optional wrapper and force None
        optional_mutants = [m for m in mutants if "Optional" in m.original_code]
        assert len(optional_mutants) >= 2
        
        # Check for Optional[str] → str mutation
        remove_optional = [m for m in optional_mutants if m.mutated_code == "str"]
        assert len(remove_optional) >= 1
        
        # Check for Optional[str] → None mutation  
        force_none = [m for m in optional_mutants if m.mutated_code == "None"]
        assert len(force_none) >= 1
    
    def test_optional_return_type_mutation(self, mutator):
        """Test mutation of Optional return types."""
        source_code = '''
def get_user(user_id: int) -> Optional[User]:
    return None
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        optional_mutants = [m for m in mutants if "Optional" in m.original_code]
        assert len(optional_mutants) >= 2
        
        # Should generate User and None mutations
        mutated_codes = [m.mutated_code for m in optional_mutants]
        assert "User" in mutated_codes
        assert "None" in mutated_codes
    
    def test_optional_variable_annotation_mutation(self, mutator):
        """Test mutation of Optional variable annotations."""
        source_code = '''
user_name: Optional[str] = None
age: Optional[int] = 25
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        optional_mutants = [m for m in mutants if "Optional" in m.original_code]
        assert len(optional_mutants) >= 4  # 2 variables × 2 mutations each
        
        # Check mutations for both variables
        str_mutations = [m for m in optional_mutants if "str" in m.original_code]
        int_mutations = [m for m in optional_mutants if "int" in m.original_code]
        
        assert len(str_mutations) >= 2
        assert len(int_mutations) >= 2


class TestUnionTypeMutations:
    """Test Union[A, B, ...] type mutations."""
    
    @pytest.fixture
    def mutator(self):
        return TypeHintMutator()
    
    def test_simple_union_mutation(self, mutator):
        """Test mutation of Union[A, B] types."""
        source_code = '''
def handle_input(data: Union[str, int]) -> None:
    pass
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        union_mutants = [m for m in mutants if "Union" in m.original_code]
        assert len(union_mutants) >= 2
        
        # Should generate str and int mutations
        mutated_codes = [m.mutated_code for m in union_mutants]
        assert "str" in mutated_codes
        assert "int" in mutated_codes
    
    def test_complex_union_mutation(self, mutator):
        """Test mutation of Union[A, B, C] with multiple types."""
        source_code = '''
def process(value: Union[str, int, float]) -> Union[bool, None]:
    return True
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        union_mutants = [m for m in mutants if "Union" in m.original_code]
        assert len(union_mutants) >= 4  # 2 union types with multiple mutations each
        
        # Check parameter mutations
        param_mutants = [m for m in union_mutants if "str, int, float" in m.original_code]
        assert len(param_mutants) >= 3
        
        # Check return type mutations  
        return_mutants = [m for m in union_mutants if "bool, None" in m.original_code or "None" in m.original_code]
        assert len(return_mutants) >= 1
    
    def test_union_with_none_equivalent_to_optional(self, mutator):
        """Test that Union[T, None] is treated like Optional[T]."""
        source_code = '''
def get_data() -> Union[dict, None]:
    return None
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        union_mutants = [m for m in mutants if "Union" in m.original_code]
        assert len(union_mutants) >= 2
        
        mutated_codes = [m.mutated_code for m in union_mutants]
        assert "dict" in mutated_codes
        assert "None" in mutated_codes


class TestGenericTypeMutations:
    """Test generic type mutations like List[T], Dict[K, V]."""
    
    @pytest.fixture
    def mutator(self):
        return TypeHintMutator()
    
    def test_list_generic_mutation(self, mutator):
        """Test mutation of List[T] types."""
        source_code = '''
def process_items(items: List[str]) -> List[int]:
    return []
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        list_mutants = [m for m in mutants if "List" in m.original_code]
        assert len(list_mutants) >= 2
        
        # Should generate list and [] mutations
        mutated_codes = [m.mutated_code for m in list_mutants]
        assert "list" in mutated_codes or "[]" in mutated_codes
    
    def test_dict_generic_mutation(self, mutator):
        """Test mutation of Dict[K, V] types."""
        source_code = '''
def get_mapping() -> Dict[str, int]:
    return {}
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        dict_mutants = [m for m in mutants if "Dict" in m.original_code]
        assert len(dict_mutants) >= 1
        
        mutated_codes = [m.mutated_code for m in dict_mutants]
        assert "dict" in mutated_codes or "{}" in mutated_codes
    
    def test_complex_generic_mutations(self, mutator):
        """Test mutations of complex generic types."""
        source_code = '''
def complex_func(
    data: Dict[str, List[int]],
    callback: Callable[[str], bool]
) -> Tuple[str, int]:
    return ("", 0)
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate mutations for Dict, List, Callable, and Tuple
        generic_mutants = [m for m in mutants if any(
            generic in m.original_code 
            for generic in ["Dict", "List", "Callable", "Tuple"]
        )]
        assert len(generic_mutants) >= 3
    
    def test_sequence_and_mapping_mutations(self, mutator):
        """Test mutations of abstract base types."""
        source_code = '''
from typing import Sequence, Mapping, Iterable

def process(
    seq: Sequence[int],
    mapping: Mapping[str, float],
    iterable: Iterable[str]
) -> None:
    pass
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        abstract_mutants = [m for m in mutants if any(
            abstract in m.original_code
            for abstract in ["Sequence", "Mapping", "Iterable"]
        )]
        assert len(abstract_mutants) >= 3


class TestSimpleTypeMutations:
    """Test mutations of simple type annotations."""
    
    @pytest.fixture
    def mutator(self):
        return TypeHintMutator()
    
    def test_basic_type_substitutions(self, mutator):
        """Test substitution of basic types."""
        source_code = '''
def calculate(x: int, y: float) -> str:
    return str(x + y)
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate mutations for int, float, and str
        type_mutants = [m for m in mutants if m.original_code in ["int", "float", "str"]]
        assert len(type_mutants) >= 6  # 3 types × ~2-3 substitutions each
        
        # Check specific substitutions
        int_mutants = [m for m in type_mutants if m.original_code == "int"]
        assert any("float" in m.mutated_code for m in int_mutants)
        assert any("Any" in m.mutated_code for m in int_mutants)
    
    def test_bool_and_none_mutations(self, mutator):
        """Test mutation of bool and None types."""
        source_code = '''
def is_valid(flag: bool) -> None:
    return None
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        bool_mutants = [m for m in mutants if m.original_code == "bool"]
        none_mutants = [m for m in mutants if m.original_code == "None"]
        
        assert len(bool_mutants) >= 2
        assert len(none_mutants) >= 2
    
    def test_any_and_object_mutations(self, mutator):
        """Test mutation of Any and object types."""
        source_code = '''
from typing import Any

def handle_any(value: Any) -> object:
    return value
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        any_mutants = [m for m in mutants if m.original_code == "Any"]
        object_mutants = [m for m in mutants if m.original_code == "object"]
        
        assert len(any_mutants) >= 1
        assert len(object_mutants) >= 1


class TestAttributeAnnotationMutations:
    """Test mutations of typing module attribute annotations."""
    
    @pytest.fixture
    def mutator(self):
        return TypeHintMutator()
    
    def test_typing_module_simplification(self, mutator):
        """Test simplification of typing.X to X."""
        source_code = '''
import typing

def func(x: typing.Optional[str]) -> typing.List[int]:
    return []
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate simplifications for typing.Optional and typing.List
        typing_mutants = [m for m in mutants if "typing." in m.original_code]
        assert len(typing_mutants) >= 2
        
        # Check for simplification mutations
        simplified = [m for m in typing_mutants if "typing." not in m.mutated_code]
        assert len(simplified) >= 2


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling in type hint mutations."""
    
    @pytest.fixture
    def mutator(self):
        return TypeHintMutator()
    
    def test_invalid_syntax_handling(self, mutator):
        """Test that invalid syntax is handled gracefully."""
        invalid_code = '''
def func(x: invalid_syntax_here) -> ???:
    pass
'''
        # Should not raise an exception
        mutants = mutator.generate_mutants(invalid_code, "test.py")
        # May return empty list or handle gracefully
        assert isinstance(mutants, list)
    
    def test_complex_nested_types(self, mutator):
        """Test handling of deeply nested type annotations."""
        source_code = '''
from typing import Dict, List, Optional, Union, Callable

def complex_func(
    data: Dict[str, List[Optional[Union[int, float]]]],
    callback: Optional[Callable[[str], Dict[str, Any]]]
) -> Optional[List[Tuple[str, int]]]:
    return None
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should handle complex nested types without crashing
        assert len(mutants) >= 5  # At least some mutations should be generated
        
        # All mutants should have valid structure
        for mutant in mutants:
            assert hasattr(mutant, 'id')
            assert hasattr(mutant, 'mutation_type')
            assert hasattr(mutant, 'original_code')
            assert hasattr(mutant, 'mutated_code')
            assert mutant.mutation_type == MutationType.TYPE_HINT
    
    def test_no_type_hints_file(self, mutator):
        """Test file with no type hints."""
        source_code = '''
def old_style_func(x, y):
    return x + y

class OldClass:
    def method(self, value):
        return value
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should return empty list or very few mutants
        assert len(mutants) == 0
    
    def test_python_version_compatibility(self, mutator):
        """Test compatibility with different Python version syntax."""
        # Test Python 3.10+ Union syntax (X | Y)
        source_code = '''
def func(x: int | str) -> bool | None:
    return True
'''
        # Should handle gracefully even if running on older Python
        mutants = mutator.generate_mutants(source_code, "test.py")
        assert isinstance(mutants, list)  # Should not crash


class TestMutantQuality:
    """Test the quality and properties of generated mutants."""
    
    @pytest.fixture
    def mutator(self):
        return TypeHintMutator()
    
    def test_mutant_ids_are_unique(self, mutator):
        """Test that generated mutants have unique IDs."""
        source_code = '''
def func(x: Optional[str], y: Union[int, float]) -> List[Dict[str, Any]]:
    return []
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        mutant_ids = [m.id for m in mutants]
        assert len(mutant_ids) == len(set(mutant_ids))  # All IDs should be unique
    
    def test_mutant_severity_levels(self, mutator):
        """Test that mutants have appropriate severity levels."""
        source_code = '''
def critical_func(data: Optional[str]) -> Union[int, str]:
    return 0
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should have mutants with different severity levels
        severities = [m.severity for m in mutants]
        assert len(set(severities)) >= 2  # At least 2 different severity levels
        
        # Common severities should be present
        valid_severities = {"low", "medium", "high", "critical"}
        assert all(sev in valid_severities for sev in severities)
    
    def test_mutant_descriptions_are_informative(self, mutator):
        """Test that mutant descriptions are informative."""
        source_code = '''
def func(x: Optional[int]) -> List[str]:
    return []
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # All descriptions should be non-empty and informative
        for mutant in mutants:
            assert len(mutant.description) > 10  # Reasonable length
            assert mutant.original_code in mutant.description
            assert mutant.mutated_code in mutant.description
    
    def test_line_and_column_information(self, mutator):
        """Test that mutants have correct line and column information."""
        source_code = '''def func(x: int) -> str:
    return "test"
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        for mutant in mutants:
            assert mutant.line_number >= 1
            assert mutant.column_start >= 0
            assert mutant.column_end > mutant.column_start
            assert mutant.language == "python"

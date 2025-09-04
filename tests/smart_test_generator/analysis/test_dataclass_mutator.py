"""Tests for DataclassMutator - dataclass mutations for modern Python."""

import ast
import pytest
from smart_test_generator.analysis.dataclass_mutator import DataclassMutator
from smart_test_generator.models.data_models import MutationType


class TestDataclassMutator:
    """Test the DataclassMutator for dataclass mutations."""
    
    @pytest.fixture
    def mutator(self):
        """Create a DataclassMutator instance."""
        return DataclassMutator()
    
    def test_get_mutation_type(self, mutator):
        """Test that the mutator returns correct mutation type."""
        assert mutator.get_mutation_type() == MutationType.DATACLASS
    
    def test_is_applicable(self, mutator):
        """Test that mutator is applicable to correct node types."""
        # Should be applicable to class definitions (for decorators)
        class_node = ast.parse("@dataclass\nclass Test: pass").body[0]
        assert mutator.is_applicable(class_node)
        
        # Should be applicable to function definitions (for __post_init__)
        func_node = ast.parse("def __post_init__(self): pass").body[0]
        assert mutator.is_applicable(func_node)
        
        # Should be applicable to field() calls
        call_node = ast.parse("field(default=None)").body[0].value
        assert mutator.is_applicable(call_node)
        
        # Should not be applicable to simple assignments without dataclass context
        assign_node = ast.parse("x = 5").body[0]
        assert not mutator.is_applicable(assign_node)


class TestDataclassDecoratorMutations:
    """Test @dataclass decorator mutations."""
    
    @pytest.fixture
    def mutator(self):
        return DataclassMutator()
    
    def test_remove_dataclass_decorator(self, mutator):
        """Test complete removal of @dataclass decorator."""
        source_code = '''
@dataclass
class User:
    name: str
    age: int
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate mutation to remove @dataclass
        remove_mutants = [m for m in mutants if "@dataclass" in m.original_code and "removed" in m.mutated_code]
        assert len(remove_mutants) >= 1
        
        # Should be critical severity
        assert any(m.severity == "critical" for m in remove_mutants)
    
    def test_add_parameters_to_bare_dataclass(self, mutator):
        """Test adding parameters to bare @dataclass decorator."""
        source_code = '''
@dataclass
class Product:
    name: str
    price: float
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate mutations to add parameters
        add_param_mutants = [m for m in mutants if m.original_code == "@dataclass" and "=" in m.mutated_code]
        assert len(add_param_mutants) >= 3
        
        # Should add frozen, order, and init parameters
        mutated_codes = [m.mutated_code for m in add_param_mutants]
        assert any("frozen=True" in code for code in mutated_codes)
        assert any("order=True" in code for code in mutated_codes)
        assert any("init=False" in code for code in mutated_codes)
    
    def test_mutate_dataclass_parameters(self, mutator):
        """Test mutations of @dataclass() parameters."""
        source_code = '''
@dataclass(frozen=True, order=False, repr=True)
class ImmutableData:
    value: int
    name: str
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate parameter mutations
        param_mutants = [m for m in mutants if "=" in m.original_code and "=" in m.mutated_code]
        assert len(param_mutants) >= 3
        
        # Check for specific parameter flips
        frozen_mutants = [m for m in param_mutants if "frozen=" in m.original_code]
        order_mutants = [m for m in param_mutants if "order=" in m.original_code]
        repr_mutants = [m for m in param_mutants if "repr=" in m.original_code]
        
        assert len(frozen_mutants) >= 1
        assert len(order_mutants) >= 1  
        assert len(repr_mutants) >= 1
    
    def test_complex_dataclass_decorator(self, mutator):
        """Test mutations of complex @dataclass decorator."""
        source_code = '''
@dataclass(
    init=True,
    repr=True,
    eq=True,
    order=False,
    unsafe_hash=False,
    frozen=False
)
class ComplexData:
    a: int
    b: str
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate many parameter mutations
        param_mutants = [m for m in mutants if "=" in m.original_code and "=" in m.mutated_code]
        assert len(param_mutants) >= 6  # One for each parameter
        
        # Should also have decorator removal
        remove_mutants = [m for m in mutants if "@dataclass" in m.original_code and "removed" in m.mutated_code]
        assert len(remove_mutants) >= 1


class TestDataclassFieldMutations:
    """Test dataclass field mutations."""
    
    @pytest.fixture
    def mutator(self):
        return DataclassMutator()
    
    def test_simple_field_default_mutations(self, mutator):
        """Test mutations of simple field defaults."""
        source_code = '''
@dataclass
class SimpleData:
    name: str = ""
    count: int = 0
    active: bool = True
    data: Optional[dict] = None
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate default value mutations
        default_mutants = [m for m in mutants if m.original_code in ['""', "''", '0', 'True', 'None']]
        assert len(default_mutants) >= 4
        
        # Check specific mutations (handle both single and double quotes for strings)
        string_mutants = [m for m in default_mutants if m.original_code in ['""', "''"]]
        int_mutants = [m for m in default_mutants if m.original_code == '0']
        bool_mutants = [m for m in default_mutants if m.original_code == 'True']
        none_mutants = [m for m in default_mutants if m.original_code == 'None']
        
        assert len(string_mutants) >= 1
        assert len(int_mutants) >= 1  
        assert len(bool_mutants) >= 1
        assert len(none_mutants) >= 1
    
    def test_field_call_mutations(self, mutator):
        """Test mutations of field() calls."""
        source_code = '''
from dataclasses import field

@dataclass
class FieldData:
    name: str = field(default="unknown")
    items: list = field(default_factory=list)
    config: dict = field(default_factory=dict, repr=False)
    id: int = field(init=False, default=0)
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate field parameter mutations
        field_mutants = [m for m in mutants if any([
            "default=" in m.original_code,
            "default_factory=" in m.original_code,
            "repr=" in m.original_code,
            "init=" in m.original_code,
            "FIELD_" in m.id
        ])]
        assert len(field_mutants) >= 4
    
    def test_dangerous_default_factory_mutations(self, mutator):
        """Test dangerous mutations from default_factory to default."""
        source_code = '''
@dataclass  
class DangerousDefaults:
    items: list = field(default_factory=list)
    data: dict = field(default_factory=dict)
    tags: set = field(default_factory=set)
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate dangerous default_factory → default mutations
        dangerous_mutants = [m for m in mutants if "default_factory=" in m.original_code and "default=" in m.mutated_code]
        assert len(dangerous_mutants) >= 3
        
        # Should be critical severity (shared mutable defaults)
        assert all(m.severity == "critical" for m in dangerous_mutants)
        
        # Check specific dangerous conversions
        mutated_codes = [m.mutated_code for m in dangerous_mutants]
        assert any("default=[]" in code for code in mutated_codes)
        assert any("default={}" in code for code in mutated_codes)
    
    def test_field_parameter_mutations(self, mutator):
        """Test mutations of field() parameters."""
        source_code = '''
@dataclass
class FieldParams:
    name: str = field(init=True, repr=True, compare=True, hash=False)
    secret: str = field(init=False, repr=False, compare=False)
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate parameter value mutations
        param_mutants = [m for m in mutants if any(param in m.original_code for param in ["init=", "repr=", "compare=", "hash="])]
        assert len(param_mutants) >= 4
        
        # Should generate parameter removal mutations
        remove_mutants = [m for m in mutants if "removed" in m.mutated_code]
        assert len(remove_mutants) >= 2


class TestDataclassMethodMutations:
    """Test dataclass method mutations."""
    
    @pytest.fixture
    def mutator(self):
        return DataclassMutator()
    
    def test_post_init_removal(self, mutator):
        """Test removal of __post_init__ method."""
        source_code = '''
@dataclass
class WithPostInit:
    name: str
    full_name: str = field(init=False)
    
    def __post_init__(self):
        self.full_name = f"Mr. {self.name}"
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate __post_init__ removal mutation
        post_init_mutants = [m for m in mutants if "__post_init__" in m.original_code]
        assert len(post_init_mutants) >= 1
        
        # Should remove the method
        remove_mutants = [m for m in post_init_mutants if "removed" in m.mutated_code]
        assert len(remove_mutants) >= 1
        
        # Should be medium severity
        assert any(m.severity == "medium" for m in post_init_mutants)
    
    def test_complex_post_init(self, mutator):
        """Test mutations with complex __post_init__ method."""
        source_code = '''
@dataclass
class ComplexPostInit:
    value: int
    computed: float = field(init=False)
    cached: str = field(init=False)
    
    def __post_init__(self):
        self.computed = self.value * 1.5
        self.cached = f"cached_{self.value}"
        if self.value < 0:
            raise ValueError("Value must be positive")
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate __post_init__ mutations
        post_init_mutants = [m for m in mutants if "__post_init__" in m.description]
        assert len(post_init_mutants) >= 1


class TestInheritanceAndComplexScenarios:
    """Test dataclass inheritance and complex scenarios."""
    
    @pytest.fixture
    def mutator(self):
        return DataclassMutator()
    
    def test_dataclass_inheritance(self, mutator):
        """Test mutations with dataclass inheritance."""
        source_code = '''
@dataclass
class BaseData:
    id: int
    name: str

@dataclass  
class ExtendedData(BaseData):
    value: float = 0.0
    active: bool = True
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate mutations for both classes
        dataclass_mutants = [m for m in mutants if "@dataclass" in m.original_code]
        assert len(dataclass_mutants) >= 2  # Both classes should have decorator mutations
    
    def test_mixed_dataclass_and_regular_class(self, mutator):
        """Test handling of mixed dataclass and regular classes."""
        source_code = '''
class RegularClass:
    def __init__(self, value):
        self.value = value

@dataclass
class DataClass:
    name: str
    count: int = 0
    
    def __post_init__(self):
        self.processed = True

class AnotherRegular:
    pass
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should only generate mutations for the dataclass
        dataclass_mutants = [m for m in mutants if any(
            term in m.description.lower() 
            for term in ["dataclass", "field", "post_init"]
        )]
        assert len(dataclass_mutants) >= 2
        
        # Regular classes should not generate dataclass-specific mutations
        regular_mutants = [m for m in mutants if "RegularClass" in m.description or "AnotherRegular" in m.description]
        assert len(regular_mutants) == 0
    
    def test_dataclass_with_methods(self, mutator):
        """Test dataclass with additional methods."""
        source_code = '''
@dataclass
class WithMethods:
    name: str
    value: int = 0
    
    def calculate(self):
        return self.value * 2
    
    def __str__(self):
        return f"{self.name}: {self.value}"
    
    def __post_init__(self):
        if self.value < 0:
            self.value = 0
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should generate mutations for dataclass parts, not regular methods
        dataclass_mutants = [m for m in mutants if any(
            term in m.description.lower() 
            for term in ["dataclass", "post_init"]
        )]
        assert len(dataclass_mutants) >= 2
        
        # Should not mutate regular methods
        method_mutants = [m for m in mutants if "calculate" in m.description or "__str__" in m.description]
        assert len(method_mutants) == 0


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def mutator(self):
        return DataclassMutator()
    
    def test_invalid_dataclass_syntax(self, mutator):
        """Test handling of invalid dataclass syntax."""
        invalid_code = '''
@dataclass(invalid_param=True)
class BadDataclass:
    field: invalid_type = bad_default
'''
        # Should not raise an exception
        mutants = mutator.generate_mutants(invalid_code, "test.py")
        assert isinstance(mutants, list)
    
    def test_no_dataclass_code(self, mutator):
        """Test file with no dataclass code."""
        source_code = '''
class RegularClass:
    def __init__(self, value):
        self.value = value

def regular_function():
    return "no dataclass here"
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should return empty list or no dataclass-specific mutations
        assert len(mutants) == 0
    
    def test_empty_dataclass(self, mutator):
        """Test empty dataclass."""
        source_code = '''
@dataclass
class EmptyData:
    pass
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should at least generate decorator mutations
        decorator_mutants = [m for m in mutants if "@dataclass" in m.original_code]
        assert len(decorator_mutants) >= 1
    
    def test_complex_field_expressions(self, mutator):
        """Test handling of complex field expressions."""
        source_code = '''
@dataclass
class ComplexFields:
    computed: int = field(default_factory=lambda: sum(range(10)))
    metadata_field: str = field(metadata={"description": "complex"})
    complex_default: list = field(default_factory=lambda: [i for i in range(5)])
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should handle complex expressions without crashing
        assert isinstance(mutants, list)
        assert len(mutants) >= 1


class TestMutantQuality:
    """Test the quality and properties of generated mutants."""
    
    @pytest.fixture
    def mutator(self):
        return DataclassMutator()
    
    def test_mutant_ids_are_unique(self, mutator):
        """Test that generated mutants have unique IDs."""
        source_code = '''
@dataclass(frozen=True, order=True)
class TestData:
    name: str = field(default="test", repr=True)
    value: int = field(default_factory=lambda: 0)
    items: list = field(default_factory=list)
    
    def __post_init__(self):
        pass
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        mutant_ids = [m.id for m in mutants]
        assert len(mutant_ids) == len(set(mutant_ids))
    
    def test_severity_levels_appropriate(self, mutator):
        """Test that mutants have appropriate severity levels."""
        source_code = '''
@dataclass
class SeverityTest:
    items: list = field(default_factory=list)  # Critical when converted to default=[]
    name: str = field(repr=False)              # Medium/low for repr change
    
    def __post_init__(self):                   # Medium for removal
        pass
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        severities = [m.severity for m in mutants]
        assert len(set(severities)) >= 2
        
        # Should have critical severity for dangerous mutations
        assert any(sev == "critical" for sev in severities)
        
        # Should have medium/low for less dangerous mutations
        assert any(sev in ["medium", "low"] for sev in severities)
    
    def test_descriptions_are_informative(self, mutator):
        """Test that descriptions explain dataclass implications."""
        source_code = '''
@dataclass(frozen=True)
class TestDescriptions:
    items: list = field(default_factory=list)
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        for mutant in mutants:
            assert len(mutant.description) > 10
            description_lower = mutant.description.lower()
            
            # Should mention dataclass-related concepts
            assert any(term in description_lower for term in [
                "dataclass", "field", "default", "parameter", "frozen", "factory"
            ])
    
    def test_line_and_column_information(self, mutator):
        """Test that mutants have correct position information."""
        source_code = '''@dataclass
class Test:
    field: str = "value"
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        for mutant in mutants:
            assert mutant.line_number >= 1
            assert mutant.column_start >= 0
            assert mutant.column_end > mutant.column_start
            assert mutant.language == "python"
    
    def test_mutation_targeting(self, mutator):
        """Test that mutations target appropriate code sections."""
        source_code = '''
@dataclass(frozen=False)
class TargetTest:
    name: str = field(default="test")
    count: int = field(default_factory=int)
    
    def __post_init__(self):
        self.computed = self.name + str(self.count)
'''
        mutants = mutator.generate_mutants(source_code, "test.py")
        
        # Should target specific parts of dataclass code
        targets = [m.original_code for m in mutants]
        
        # Should target decorator, field parameters, and __post_init__
        assert any("@dataclass" in target or "frozen=" in target for target in targets)
        assert any("field(" in target or "default" in target for target in targets)
        assert any("__post_init__" in target for target in targets)

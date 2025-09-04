"""Dataclass mutation operator for modern Python dataclass patterns."""

import ast
import logging
from typing import Dict, List, Optional, Set, Any, Union
import sys

from smart_test_generator.models.data_models import MutationType, Mutant
from smart_test_generator.analysis.mutation_engine import MutationOperator

logger = logging.getLogger(__name__)


class DataclassMutator(MutationOperator):
    """Mutate dataclass decorators and field configurations to test dataclass edge cases."""
    
    def __init__(self):
        """Initialize dataclass mutation mappings."""
        
        # Dataclass decorator parameter mutations
        self.decorator_parameter_mutations = {
            'init': [True, False],              # init=True ↔ init=False
            'repr': [True, False],              # repr=True ↔ repr=False  
            'eq': [True, False],                # eq=True ↔ eq=False
            'order': [True, False],             # order=True ↔ order=False
            'unsafe_hash': [True, False],       # unsafe_hash=True ↔ unsafe_hash=False
            'frozen': [True, False],            # frozen=True ↔ frozen=False
        }
        
        # Field parameter mutations  
        self.field_parameter_mutations = {
            'default': ['REMOVE', 'None', '0', '""', '[]'],     # Remove default or change value
            'default_factory': ['REMOVE', 'default'],           # Remove factory or convert to default
            'init': ['REMOVE', True, False],    # Remove or toggle init parameter
            'repr': ['REMOVE', True, False],    # Remove or toggle repr parameter
            'compare': ['REMOVE', True, False], # Remove or toggle compare parameter
            'hash': ['REMOVE', True, False, None], # Remove or change hash parameter
            'metadata': ['REMOVE', '{}'],       # Remove metadata or empty dict
        }
        
        # Dangerous default_factory mutations (shared mutable defaults)
        self.dangerous_factory_mutations = {
            'list': '[]',           # field(default_factory=list) → field(default=[])
            'dict': '{}',           # field(default_factory=dict) → field(default={})
            'set': 'set()',         # field(default_factory=set) → field(default=set())
        }
        
        # Default value mutations (type-specific)
        self.default_value_mutations = {
            'None': ['0', '""', '[]', 'False'],
            '0': ['None', '1', '-1', '""'],
            '""': ['None', '0', "''", 'False'],
            "''": ['None', '0', '""', 'False'],  # Handle single quotes from ast.unparse
            '[]': ['None', '{}', 'set()', '()'],
            '{}': ['None', '[]', 'set()', '()'],
            'False': ['True', 'None', '0', '""'],
            'True': ['False', 'None', '1', '""'],
        }
        
        # Dataclass inheritance patterns
        self.inheritance_mutations = {
            'slots': [True, False],             # slots=True ↔ slots=False (Python 3.10+)
            'kw_only': [True, False],          # kw_only=True ↔ kw_only=False (Python 3.10+)
            'match_args': [True, False],       # match_args=True ↔ match_args=False (Python 3.10+)
        }
    
    def get_mutation_type(self) -> MutationType:
        """Get the mutation type for dataclass mutations."""
        return MutationType.DATACLASS
    
    def is_applicable(self, node: Any) -> bool:
        """Check if this operator can be applied to the given AST node."""
        return isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.Call, ast.Name))
    
    def generate_mutants(self, source_code: str, filepath: str) -> List[Mutant]:
        """Generate dataclass mutants for the given source code."""
        mutants = []
        
        try:
            tree = ast.parse(source_code)
            
            # Walk through all nodes to find dataclass patterns
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    mutants.extend(self._mutate_dataclass_decorator(node, source_code))
                    mutants.extend(self._mutate_dataclass_fields(node, source_code))
                    mutants.extend(self._mutate_dataclass_methods(node, source_code))
                elif isinstance(node, ast.Call):
                    mutants.extend(self._mutate_field_calls(node, source_code))
                    
        except Exception as e:
            logger.error(f"Error generating dataclass mutants for {filepath}: {e}")
        
        return mutants
    
    def _mutate_dataclass_decorator(self, node: ast.ClassDef, source_code: str) -> List[Mutant]:
        """Mutate @dataclass decorator and its parameters."""
        mutants = []
        
        for decorator in node.decorator_list:
            if self._is_dataclass_decorator(decorator):
                # Mutation 1: Remove @dataclass decorator entirely
                mutant_id = f"DATACLASS_REMOVE_{node.lineno}_{node.name}"
                mutant = Mutant(
                    id=mutant_id,
                    mutation_type=self.get_mutation_type(),
                    original_code="@dataclass",
                    mutated_code="# @dataclass removed",
                    line_number=node.lineno,
                    column_start=decorator.col_offset,
                    column_end=getattr(decorator, 'end_col_offset', 
                                     decorator.col_offset + len("@dataclass")),
                    description=f"Remove @dataclass decorator from {node.name}",
                    severity="critical",
                    language="python"
                )
                mutants.append(mutant)
                
                # Mutation 2: Mutate decorator parameters
                if isinstance(decorator, ast.Call):
                    mutants.extend(self._mutate_decorator_parameters(
                        decorator, node.name, node.lineno
                    ))
                else:
                    # Add parameters to bare @dataclass
                    mutants.extend(self._add_decorator_parameters(
                        decorator, node.name, node.lineno
                    ))
        
        return mutants
    
    def _mutate_dataclass_fields(self, node: ast.ClassDef, source_code: str) -> List[Mutant]:
        """Mutate dataclass field annotations and assignments."""
        mutants = []
        
        # Only process if class has @dataclass decorator
        if not self._has_dataclass_decorator(node):
            return mutants
        
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign):
                # Field with type annotation
                mutants.extend(self._mutate_annotated_field(stmt, node.name))
            elif isinstance(stmt, ast.Assign):
                # Simple field assignment (less common in dataclasses)
                mutants.extend(self._mutate_simple_field(stmt, node.name))
        
        return mutants
    
    def _mutate_dataclass_methods(self, node: ast.ClassDef, source_code: str) -> List[Mutant]:
        """Mutate dataclass special methods like __post_init__."""
        mutants = []
        
        # Only process if class has @dataclass decorator
        if not self._has_dataclass_decorator(node):
            return mutants
        
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                if stmt.name == '__post_init__':
                    # Mutation: Remove __post_init__ method
                    mutant_id = f"DATACLASS_POST_INIT_REMOVE_{stmt.lineno}_{node.name}"
                    mutant = Mutant(
                        id=mutant_id,
                        mutation_type=self.get_mutation_type(),
                        original_code="def __post_init__",
                        mutated_code="# def __post_init__ removed",
                        line_number=stmt.lineno,
                        column_start=stmt.col_offset,
                        column_end=stmt.col_offset + len("def __post_init__"),
                        description=f"Remove __post_init__ method from {node.name}",
                        severity="medium",
                        language="python"
                    )
                    mutants.append(mutant)
        
        return mutants
    
    def _mutate_field_calls(self, node: ast.Call, source_code: str) -> List[Mutant]:
        """Mutate field() function calls."""
        mutants = []
        
        # Check if this is a field() call
        if (isinstance(node.func, ast.Name) and node.func.id == 'field') or \
           (isinstance(node.func, ast.Attribute) and node.func.attr == 'field'):
            
            mutants.extend(self._mutate_field_parameters(node))
        
        return mutants
    
    def _mutate_decorator_parameters(self, decorator: ast.Call, class_name: str, 
                                   line_number: int) -> List[Mutant]:
        """Mutate parameters of @dataclass() decorator."""
        mutants = []
        
        # Extract current parameter values
        current_params = self._extract_decorator_params(decorator)
        
        # Mutate each parameter
        for param_name, mutations in self.decorator_parameter_mutations.items():
            current_value = current_params.get(param_name)
            
            for new_value in mutations:
                if current_value != new_value:  # Don't create identical mutations
                    mutant_id = f"DATACLASS_PARAM_{param_name}_{new_value}_{line_number}_{class_name}"
                    
                    original_param = f"{param_name}={current_value}" if current_value is not None else f"# {param_name} default"
                    mutated_param = f"{param_name}={new_value}"
                    
                    mutant = Mutant(
                        id=mutant_id,
                        mutation_type=self.get_mutation_type(),
                        original_code=original_param,
                        mutated_code=mutated_param,
                        line_number=line_number,
                        column_start=decorator.col_offset,
                        column_end=getattr(decorator, 'end_col_offset', 
                                         decorator.col_offset + 20),
                        description=f"Change dataclass parameter: {original_param} → {mutated_param}",
                        severity="medium" if param_name in ['repr', 'init'] else "low",
                        language="python"
                    )
                    mutants.append(mutant)
        
        return mutants
    
    def _add_decorator_parameters(self, decorator: ast.Name, class_name: str, 
                                line_number: int) -> List[Mutant]:
        """Add parameters to bare @dataclass decorator."""
        mutants = []
        
        # Add dangerous parameters
        dangerous_params = [
            ('frozen', 'True'),   # Make immutable
            ('order', 'True'),    # Add ordering
            ('init', 'False'),    # Disable __init__
        ]
        
        for param_name, param_value in dangerous_params:
            mutant_id = f"DATACLASS_ADD_{param_name}_{line_number}_{class_name}"
            mutant = Mutant(
                id=mutant_id,
                mutation_type=self.get_mutation_type(),
                original_code="@dataclass",
                mutated_code=f"@dataclass({param_name}={param_value})",
                line_number=line_number,
                column_start=decorator.col_offset,
                column_end=getattr(decorator, 'end_col_offset', 
                                 decorator.col_offset + len("@dataclass")),
                description=f"Add parameter to dataclass: {param_name}={param_value}",
                severity="medium",
                language="python"
            )
            mutants.append(mutant)
        
        return mutants
    
    def _mutate_annotated_field(self, node: ast.AnnAssign, class_name: str) -> List[Mutant]:
        """Mutate annotated field assignments."""
        mutants = []
        
        if not node.value:  # No default value
            return mutants
            
        field_name = ""
        if isinstance(node.target, ast.Name):
            field_name = node.target.id
        
        # Check if it's a field() call
        if isinstance(node.value, ast.Call) and \
           ((isinstance(node.value.func, ast.Name) and node.value.func.id == 'field') or
            (isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'field')):
            
            # This is handled by _mutate_field_parameters
            return mutants
        
        # Mutate simple default values
        try:
            if sys.version_info >= (3, 9):
                default_value = ast.unparse(node.value)
            else:
                default_value = self._ast_to_string(node.value)
            
            if default_value in self.default_value_mutations:
                for i, new_default in enumerate(self.default_value_mutations[default_value]):
                    # Make ID unique by including mutation index
                    mutant_id = f"DATACLASS_FIELD_DEFAULT_{field_name}_{node.lineno}_{i}"
                    mutant = Mutant(
                        id=mutant_id,
                        mutation_type=self.get_mutation_type(),
                        original_code=default_value,
                        mutated_code=new_default,
                        line_number=node.lineno,
                        column_start=node.value.col_offset,
                        column_end=getattr(node.value, 'end_col_offset',
                                         node.value.col_offset + len(default_value)),
                        description=f"Change field default: {default_value} → {new_default}",
                        severity="medium",
                        language="python"
                    )
                    mutants.append(mutant)
        except:
            pass  # Skip if unable to parse default value
        
        return mutants
    
    def _mutate_simple_field(self, node: ast.Assign, class_name: str) -> List[Mutant]:
        """Mutate simple field assignments (without type annotations)."""
        mutants = []
        
        # Simple field assignments are less common in modern dataclasses
        # but we can still mutate their values
        for target in node.targets:
            if isinstance(target, ast.Name):
                field_name = target.id
                
                try:
                    if sys.version_info >= (3, 9):
                        current_value = ast.unparse(node.value)
                    else:
                        current_value = self._ast_to_string(node.value)
                    
                    if current_value in self.default_value_mutations:
                        for new_value in self.default_value_mutations[current_value]:
                            mutant_id = f"DATACLASS_SIMPLE_FIELD_{field_name}_{node.lineno}"
                            mutant = Mutant(
                                id=mutant_id,
                                mutation_type=self.get_mutation_type(),
                                original_code=current_value,
                                mutated_code=new_value,
                                line_number=node.lineno,
                                column_start=node.value.col_offset,
                                column_end=getattr(node.value, 'end_col_offset',
                                                 node.value.col_offset + len(current_value)),
                                description=f"Change simple field value: {current_value} → {new_value}",
                                severity="low",
                                language="python"
                            )
                            mutants.append(mutant)
                except:
                    pass
        
        return mutants
    
    def _mutate_field_parameters(self, node: ast.Call) -> List[Mutant]:
        """Mutate parameters of field() calls."""
        mutants = []
        
        # Extract current field parameters
        field_params = self._extract_field_params(node)
        
        # Dangerous mutation: default_factory → default (shared mutable)
        if 'default_factory' in field_params:
            factory_value = field_params['default_factory']
            if factory_value in self.dangerous_factory_mutations:
                dangerous_default = self.dangerous_factory_mutations[factory_value]
                mutant_id = f"FIELD_FACTORY_TO_DEFAULT_{node.lineno}_{factory_value}"
                mutant = Mutant(
                    id=mutant_id,
                    mutation_type=self.get_mutation_type(),
                    original_code=f"default_factory={factory_value}",
                    mutated_code=f"default={dangerous_default}",
                    line_number=node.lineno,
                    column_start=node.col_offset,
                    column_end=getattr(node, 'end_col_offset', node.col_offset + 20),
                    description=f"Convert factory to shared mutable default: {factory_value}() → {dangerous_default}",
                    severity="critical",
                    language="python"
                )
                mutants.append(mutant)
        
        # Mutate other field parameters
        for param_name, mutations in self.field_parameter_mutations.items():
            if param_name in field_params:
                current_value = field_params[param_name]
                
                for new_value in mutations:
                    if new_value == 'REMOVE':
                        # Remove the parameter
                        mutant_id = f"FIELD_PARAM_REMOVE_{param_name}_{node.lineno}"
                        mutant = Mutant(
                            id=mutant_id,
                            mutation_type=self.get_mutation_type(),
                            original_code=f"{param_name}={current_value}",
                            mutated_code=f"# {param_name} removed",
                            line_number=node.lineno,
                            column_start=node.col_offset,
                            column_end=getattr(node, 'end_col_offset', node.col_offset + 20),
                            description=f"Remove field parameter: {param_name}",
                            severity="medium",
                            language="python"
                        )
                        mutants.append(mutant)
                    elif current_value != new_value:
                        # Change parameter value
                        mutant_id = f"FIELD_PARAM_CHANGE_{param_name}_{new_value}_{node.lineno}"
                        mutant = Mutant(
                            id=mutant_id,
                            mutation_type=self.get_mutation_type(),
                            original_code=f"{param_name}={current_value}",
                            mutated_code=f"{param_name}={new_value}",
                            line_number=node.lineno,
                            column_start=node.col_offset,
                            column_end=getattr(node, 'end_col_offset', node.col_offset + 20),
                            description=f"Change field parameter: {param_name}={current_value} → {param_name}={new_value}",
                            severity="medium" if param_name in ['init', 'repr'] else "low",
                            language="python"
                        )
                        mutants.append(mutant)
        
        return mutants
    
    def _is_dataclass_decorator(self, decorator: Union[ast.Name, ast.Call]) -> bool:
        """Check if a decorator is @dataclass."""
        if isinstance(decorator, ast.Name):
            return decorator.id == 'dataclass'
        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
            return decorator.func.id == 'dataclass'
        return False
    
    def _has_dataclass_decorator(self, node: ast.ClassDef) -> bool:
        """Check if a class has @dataclass decorator."""
        for decorator in node.decorator_list:
            if self._is_dataclass_decorator(decorator):
                return True
        return False
    
    def _extract_decorator_params(self, decorator: ast.Call) -> Dict[str, Any]:
        """Extract parameter values from @dataclass() decorator."""
        params = {}
        
        # Process keyword arguments
        for keyword in decorator.keywords:
            if keyword.arg:
                try:
                    if isinstance(keyword.value, ast.Constant):
                        params[keyword.arg] = keyword.value.value
                    elif isinstance(keyword.value, ast.NameConstant):  # Python < 3.8
                        params[keyword.arg] = keyword.value.value
                    elif isinstance(keyword.value, (ast.Name, ast.Attribute)):
                        params[keyword.arg] = self._ast_to_string(keyword.value)
                except:
                    params[keyword.arg] = 'unknown'
        
        return params
    
    def _extract_field_params(self, node: ast.Call) -> Dict[str, Any]:
        """Extract parameter values from field() call."""
        params = {}
        
        # Process keyword arguments
        for keyword in node.keywords:
            if keyword.arg:
                try:
                    if isinstance(keyword.value, ast.Constant):
                        params[keyword.arg] = keyword.value.value
                    elif isinstance(keyword.value, ast.NameConstant):  # Python < 3.8
                        params[keyword.arg] = keyword.value.value
                    elif isinstance(keyword.value, ast.Name):
                        params[keyword.arg] = keyword.value.id
                    else:
                        params[keyword.arg] = self._ast_to_string(keyword.value)
                except:
                    params[keyword.arg] = 'unknown'
        
        return params
    
    def _ast_to_string(self, node: ast.AST) -> str:
        """Convert AST node to string representation (Python < 3.9 compatibility)."""
        try:
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return f"{self._ast_to_string(node.value)}.{node.attr}"
            elif isinstance(node, ast.Call):
                func_str = self._ast_to_string(node.func)
                return f"{func_str}(...)"  # Simplified for field calls
            elif isinstance(node, ast.Constant):
                return repr(node.value)
            elif isinstance(node, ast.List):
                return "[]"
            elif isinstance(node, ast.Dict):
                return "{}"
            elif isinstance(node, ast.Set):
                return "set()"
            else:
                return f"<{type(node).__name__}>"
        except:
            return f"<unparseable:{type(node).__name__}>"

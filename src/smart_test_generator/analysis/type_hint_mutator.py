"""Type hint mutation operator for modern Python type annotations."""

import ast
import re
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
import sys

from smart_test_generator.models.data_models import MutationType, Mutant
from smart_test_generator.analysis.mutation_engine import MutationOperator

logger = logging.getLogger(__name__)


class TypeHintMutator(MutationOperator):
    """Mutate Python type hints and annotations to test type-related edge cases."""
    
    def __init__(self):
        """Initialize type hint mutation mappings."""
        
        # Optional type mutations - Optional[T] patterns
        self.optional_mutations = {
            'Optional': ['', 'None'],  # Optional[T] → T, None
            'Union': ['first', 'last', 'remove_none'],  # Union[T, None] variants
        }
        
        # Union type mutations - Union[A, B, C] patterns
        self.union_mutations = {
            'keep_first': 'first',      # Union[A, B] → A
            'keep_last': 'last',        # Union[A, B] → B
            'keep_middle': 'middle',    # Union[A, B, C] → B
            'remove_last': 'remove',    # Union[A, B, C] → Union[A, B]
        }
        
        # Generic type mutations - List[T], Dict[K, V] patterns
        self.generic_mutations = {
            'List': ['list', '[]'],                    # List[T] → list, []
            'Dict': ['dict', '{}'],                    # Dict[K, V] → dict, {}
            'Set': ['set', 'frozenset'],               # Set[T] → set, frozenset
            'Tuple': ['tuple', '()'],                  # Tuple[T, ...] → tuple, ()
            'Iterable': ['list', 'Iterator'],          # Iterable[T] → list, Iterator
            'Iterator': ['Iterable', 'list'],          # Iterator[T] → Iterable, list
            'Sequence': ['list', 'tuple'],             # Sequence[T] → list, tuple
            'Mapping': ['dict', 'Dict'],               # Mapping[K, V] → dict, Dict
            'MutableMapping': ['dict', 'Dict'],        # MutableMapping[K, V] → dict
            'Callable': ['function', 'Any'],           # Callable[[T], R] → function, Any
        }
        
        # Protocol and TypedDict mutations
        self.protocol_mutations = {
            'Protocol': ['object', 'Any'],             # Protocol → object, Any
            'TypedDict': ['dict', 'Dict[str, Any]'],   # TypedDict → dict
            'final': '',                               # @final → removed
            'runtime_checkable': '',                   # @runtime_checkable → removed
        }
        
        # Type variable mutations
        self.typevar_mutations = {
            'TypeVar': ['Any', 'object'],              # TypeVar('T') → Any
            'Generic': ['object'],                     # Generic[T] → object
        }
        
        # Complex type patterns for parsing
        self.typing_imports = {
            'Optional', 'Union', 'List', 'Dict', 'Set', 'Tuple', 'Any', 'Type',
            'Callable', 'Iterable', 'Iterator', 'Sequence', 'Mapping', 'MutableMapping',
            'Protocol', 'TypedDict', 'Generic', 'TypeVar', 'final', 'runtime_checkable',
            'ClassVar', 'NoReturn', 'Literal', 'Final'
        }
    
    def get_mutation_type(self) -> MutationType:
        """Get the mutation type for type hint mutations."""
        return MutationType.TYPE_HINT
    
    def is_applicable(self, node: Any) -> bool:
        """Check if this operator can be applied to the given AST node."""
        return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.AnnAssign, 
                                ast.arg, ast.ClassDef, ast.Subscript, ast.Name))
    
    def generate_mutants(self, source_code: str, filepath: str) -> List[Mutant]:
        """Generate type hint mutants for the given source code."""
        mutants = []
        
        try:
            tree = ast.parse(source_code)
            
            # Walk through all nodes to find type annotations
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    mutants.extend(self._mutate_function_annotations(node, source_code))
                elif isinstance(node, ast.AnnAssign):
                    mutants.extend(self._mutate_variable_annotation(node, source_code))
                elif isinstance(node, ast.ClassDef):
                    mutants.extend(self._mutate_class_annotations(node, source_code))
                    
        except Exception as e:
            logger.error(f"Error generating type hint mutants for {filepath}: {e}")
        
        return mutants
    
    def _mutate_function_annotations(self, node: ast.FunctionDef, source_code: str) -> List[Mutant]:
        """Mutate function parameter and return type annotations."""
        mutants = []
        
        # Mutate parameter annotations
        for arg in node.args.args + node.args.kwonlyargs:
            if arg.annotation:
                mutants.extend(self._mutate_annotation(
                    arg.annotation, source_code, f"param_{arg.arg}", node.lineno
                ))
        
        # Mutate return type annotation
        if node.returns:
            mutants.extend(self._mutate_annotation(
                node.returns, source_code, "return", node.lineno
            ))
        
        return mutants
    
    def _mutate_variable_annotation(self, node: ast.AnnAssign, source_code: str) -> List[Mutant]:
        """Mutate variable type annotations."""
        mutants = []
        
        if node.annotation:
            target_name = "variable"
            if isinstance(node.target, ast.Name):
                target_name = node.target.id
            
            mutants.extend(self._mutate_annotation(
                node.annotation, source_code, target_name, node.lineno
            ))
        
        return mutants
    
    def _mutate_class_annotations(self, node: ast.ClassDef, source_code: str) -> List[Mutant]:
        """Mutate class-level type annotations and decorators."""
        mutants = []
        
        # Mutate @dataclass, @final, and other type-related decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id in self.protocol_mutations:
                    mutants.extend(self._mutate_decorator(
                        decorator, source_code, node.lineno
                    ))
        
        # Mutate class variables with type annotations
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign):
                mutants.extend(self._mutate_variable_annotation(stmt, source_code))
        
        return mutants
    
    def _mutate_annotation(self, annotation: ast.AST, source_code: str, 
                          context: str, line_number: int) -> List[Mutant]:
        """Core method to mutate a type annotation AST node."""
        mutants = []
        
        try:
            # Get the string representation of the annotation
            if sys.version_info >= (3, 9):
                original_type = ast.unparse(annotation)
            else:
                original_type = self._ast_to_string(annotation)
            
            # Generate mutations based on annotation type
            if isinstance(annotation, ast.Subscript):
                mutants.extend(self._mutate_subscript_annotation(
                    annotation, original_type, context, line_number
                ))
            elif isinstance(annotation, ast.Name):
                mutants.extend(self._mutate_name_annotation(
                    annotation, original_type, context, line_number
                ))
            elif isinstance(annotation, ast.Attribute):
                mutants.extend(self._mutate_attribute_annotation(
                    annotation, original_type, context, line_number
                ))
            elif isinstance(annotation, ast.Constant):
                mutants.extend(self._mutate_constant_annotation(
                    annotation, original_type, context, line_number
                ))
        
        except Exception as e:
            logger.warning(f"Could not mutate annotation at line {line_number}: {e}")
        
        return mutants
    
    def _mutate_subscript_annotation(self, node: ast.Subscript, original_type: str,
                                   context: str, line_number: int) -> List[Mutant]:
        """Mutate subscripted type annotations like Optional[T], List[T], Union[A, B]."""
        mutants = []
        
        # Get the base type name (Optional, List, Dict, etc.)
        base_type = None
        if isinstance(node.value, ast.Name):
            base_type = node.value.id
        elif isinstance(node.value, ast.Attribute):
            base_type = node.value.attr
        
        if not base_type:
            return mutants
        
        # Handle Optional[T] mutations
        if base_type == 'Optional':
            mutants.extend(self._mutate_optional_type(
                node, original_type, context, line_number
            ))
        
        # Handle Union[A, B, ...] mutations  
        elif base_type == 'Union':
            mutants.extend(self._mutate_union_type(
                node, original_type, context, line_number
            ))
        
        # Handle generic types like List[T], Dict[K, V]
        elif base_type in self.generic_mutations:
            mutants.extend(self._mutate_generic_type(
                node, original_type, base_type, context, line_number
            ))
        
        return mutants
    
    def _mutate_optional_type(self, node: ast.Subscript, original_type: str,
                            context: str, line_number: int) -> List[Mutant]:
        """Generate mutations for Optional[T] types."""
        mutants = []
        
        # Extract the inner type T from Optional[T]
        if isinstance(node.slice, ast.Name):
            inner_type = node.slice.id
        elif sys.version_info >= (3, 9) and hasattr(node.slice, 'value'):
            inner_type = ast.unparse(node.slice)
        else:
            inner_type = self._ast_to_string(node.slice)
        
        # Mutation 1: Optional[T] → T (remove None possibility)
        mutant_id = f"TYPE_OPT_REMOVE_{line_number}_{context}"
        mutant = Mutant(
            id=mutant_id,
            mutation_type=self.get_mutation_type(),
            original_code=original_type,
            mutated_code=inner_type,
            line_number=line_number,
            column_start=node.col_offset,
            column_end=getattr(node, 'end_col_offset', node.col_offset + len(original_type)),
            description=f"Remove Optional wrapper: {original_type} → {inner_type}",
            severity="high",
            language="python"
        )
        mutants.append(mutant)
        
        # Mutation 2: Optional[T] → None (force None-only)
        mutant_id = f"TYPE_OPT_NONE_{line_number}_{context}"
        mutant = Mutant(
            id=mutant_id,
            mutation_type=self.get_mutation_type(),
            original_code=original_type,
            mutated_code="None",
            line_number=line_number,
            column_start=node.col_offset,
            column_end=getattr(node, 'end_col_offset', node.col_offset + len(original_type)),
            description=f"Force None type: {original_type} → None",
            severity="critical",
            language="python"
        )
        mutants.append(mutant)
        
        return mutants
    
    def _mutate_union_type(self, node: ast.Subscript, original_type: str,
                         context: str, line_number: int) -> List[Mutant]:
        """Generate mutations for Union[A, B, ...] types."""
        mutants = []
        
        # Extract union member types
        union_types = []
        if isinstance(node.slice, ast.Tuple):
            for elt in node.slice.elts:
                if isinstance(elt, ast.Name):
                    union_types.append(elt.id)
                else:
                    if sys.version_info >= (3, 9):
                        union_types.append(ast.unparse(elt))
                    else:
                        union_types.append(self._ast_to_string(elt))
        
        if len(union_types) < 2:
            return mutants
        
        # Mutation 1: Union[A, B, C] → A (keep first type)
        mutant_id = f"TYPE_UNION_FIRST_{line_number}_{context}"
        mutant = Mutant(
            id=mutant_id,
            mutation_type=self.get_mutation_type(),
            original_code=original_type,
            mutated_code=union_types[0],
            line_number=line_number,
            column_start=node.col_offset,
            column_end=getattr(node, 'end_col_offset', node.col_offset + len(original_type)),
            description=f"Keep first Union type: {original_type} → {union_types[0]}",
            severity="high",
            language="python"
        )
        mutants.append(mutant)
        
        # Mutation 2: Union[A, B, C] → C (keep last type)
        mutant_id = f"TYPE_UNION_LAST_{line_number}_{context}"
        mutant = Mutant(
            id=mutant_id,
            mutation_type=self.get_mutation_type(),
            original_code=original_type,
            mutated_code=union_types[-1],
            line_number=line_number,
            column_start=node.col_offset,
            column_end=getattr(node, 'end_col_offset', node.col_offset + len(original_type)),
            description=f"Keep last Union type: {original_type} → {union_types[-1]}",
            severity="high",
            language="python"
        )
        mutants.append(mutant)
        
        # Mutation 3: Union[A, B, C] → Union[A, B] (remove last type)
        if len(union_types) > 2:
            reduced_union = f"Union[{', '.join(union_types[:-1])}]"
            mutant_id = f"TYPE_UNION_REDUCE_{line_number}_{context}"
            mutant = Mutant(
                id=mutant_id,
                mutation_type=self.get_mutation_type(),
                original_code=original_type,
                mutated_code=reduced_union,
                line_number=line_number,
                column_start=node.col_offset,
                column_end=getattr(node, 'end_col_offset', node.col_offset + len(original_type)),
                description=f"Reduce Union types: {original_type} → {reduced_union}",
                severity="medium",
                language="python"
            )
            mutants.append(mutant)
        
        return mutants
    
    def _mutate_generic_type(self, node: ast.Subscript, original_type: str, base_type: str,
                           context: str, line_number: int) -> List[Mutant]:
        """Generate mutations for generic types like List[T], Dict[K, V]."""
        mutants = []
        
        if base_type in self.generic_mutations:
            for substitute in self.generic_mutations[base_type]:
                mutant_id = f"TYPE_GENERIC_{base_type}_{line_number}_{context}_{substitute}"
                mutant = Mutant(
                    id=mutant_id,
                    mutation_type=self.get_mutation_type(),
                    original_code=original_type,
                    mutated_code=substitute,
                    line_number=line_number,
                    column_start=node.col_offset,
                    column_end=getattr(node, 'end_col_offset', node.col_offset + len(original_type)),
                    description=f"Substitute generic type: {original_type} → {substitute}",
                    severity="medium",
                    language="python"
                )
                mutants.append(mutant)
        
        return mutants
    
    def _mutate_name_annotation(self, node: ast.Name, original_type: str,
                              context: str, line_number: int) -> List[Mutant]:
        """Mutate simple name annotations like 'int', 'str', 'Any'."""
        mutants = []
        
        # Common type substitutions
        type_substitutions = {
            'int': ['float', 'str', 'Any'],
            'str': ['bytes', 'int', 'Any'],
            'float': ['int', 'str', 'Any'],
            'bool': ['int', 'str', 'Any'],
            'bytes': ['str', 'bytearray', 'Any'],
            'Any': ['object', 'None'],
            'object': ['Any', 'None'],
            'None': ['Any', 'object']
        }
        
        if node.id in type_substitutions:
            for substitute in type_substitutions[node.id]:
                mutant_id = f"TYPE_NAME_{node.id}_{line_number}_{context}_{substitute}"
                mutant = Mutant(
                    id=mutant_id,
                    mutation_type=self.get_mutation_type(),
                    original_code=original_type,
                    mutated_code=substitute,
                    line_number=line_number,
                    column_start=node.col_offset,
                    column_end=getattr(node, 'end_col_offset', node.col_offset + len(original_type)),
                    description=f"Substitute type: {original_type} → {substitute}",
                    severity="medium",
                    language="python"
                )
                mutants.append(mutant)
        
        return mutants
    
    def _mutate_constant_annotation(self, node: ast.Constant, original_type: str,
                                   context: str, line_number: int) -> List[Mutant]:
        """Mutate constant annotations like None."""
        mutants = []
        
        # Handle None constants specially
        if node.value is None and original_type == 'None':
            # Use the same substitutions as for Name nodes
            substitutions = ['Any', 'object']
            
            for substitute in substitutions:
                mutant_id = f"TYPE_CONST_None_{line_number}_{context}_{substitute}"
                mutant = Mutant(
                    id=mutant_id,
                    mutation_type=self.get_mutation_type(),
                    original_code=original_type,
                    mutated_code=substitute,
                    line_number=line_number,
                    column_start=node.col_offset,
                    column_end=getattr(node, 'end_col_offset', node.col_offset + len(original_type)),
                    description=f"Substitute None type: {original_type} → {substitute}",
                    severity="medium",
                    language="python"
                )
                mutants.append(mutant)
        
        return mutants
    
    def _mutate_attribute_annotation(self, node: ast.Attribute, original_type: str,
                                   context: str, line_number: int) -> List[Mutant]:
        """Mutate attribute annotations like 'typing.Optional', 'collections.abc.Sequence'."""
        mutants = []
        
        # Handle typing module attributes
        if (isinstance(node.value, ast.Name) and node.value.id == 'typing' and 
            node.attr in self.typing_imports):
            
            # Convert typing.Optional to Optional for further mutation
            simple_type = node.attr
            if simple_type in ['Optional', 'Union', 'List', 'Dict', 'Set', 'Tuple']:
                mutant_id = f"TYPE_ATTR_{simple_type}_{line_number}_{context}"
                mutant = Mutant(
                    id=mutant_id,
                    mutation_type=self.get_mutation_type(),
                    original_code=original_type,
                    mutated_code=simple_type,
                    line_number=line_number,
                    column_start=node.col_offset,
                    column_end=getattr(node, 'end_col_offset', node.col_offset + len(original_type)),
                    description=f"Simplify typing import: {original_type} → {simple_type}",
                    severity="low",
                    language="python"
                )
                mutants.append(mutant)
        
        return mutants
    
    def _mutate_decorator(self, node: ast.Name, source_code: str, line_number: int) -> List[Mutant]:
        """Mutate type-related decorators like @final, @runtime_checkable."""
        mutants = []
        
        if node.id in self.protocol_mutations:
            substitute = self.protocol_mutations[node.id]
            
            if substitute == '':  # Remove decorator
                mutant_id = f"TYPE_DECORATOR_REMOVE_{node.id}_{line_number}"
                mutant = Mutant(
                    id=mutant_id,
                    mutation_type=self.get_mutation_type(),
                    original_code=f"@{node.id}",
                    mutated_code=f"# @{node.id} removed",
                    line_number=line_number,
                    column_start=node.col_offset,
                    column_end=getattr(node, 'end_col_offset', node.col_offset + len(node.id)),
                    description=f"Remove type decorator: @{node.id}",
                    severity="medium",
                    language="python"
                )
                mutants.append(mutant)
        
        return mutants
    
    def _ast_to_string(self, node: ast.AST) -> str:
        """Convert AST node to string representation (Python < 3.9 compatibility)."""
        try:
            # Fallback implementation for Python < 3.9
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return f"{self._ast_to_string(node.value)}.{node.attr}"
            elif isinstance(node, ast.Subscript):
                value_str = self._ast_to_string(node.value)
                slice_str = self._ast_to_string(node.slice)
                return f"{value_str}[{slice_str}]"
            elif isinstance(node, ast.Tuple):
                elements = [self._ast_to_string(elt) for elt in node.elts]
                return ", ".join(elements)
            elif isinstance(node, ast.Constant):
                return repr(node.value)
            else:
                return f"<{type(node).__name__}>"
        except:
            return f"<unparseable:{type(node).__name__}>"

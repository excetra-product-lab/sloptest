"""Map source files to existing tests."""

import os
import ast
import logging
from pathlib import Path
from typing import List, Dict, Set

from smart_test_generator.config import Config

logger = logging.getLogger(__name__)


class TestMapper:
    """Map source files to existing tests."""

    def __init__(self, project_root: Path, config: Config):
        self.project_root = project_root
        self.config = config
        self.test_patterns = self.config.get('test_generation.test_patterns', [])

    def find_test_files(self, source_file: str) -> List[str]:
        """Find all test files for a given source file."""
        test_files = []
        source_path = Path(source_file)
        module_name = source_path.stem

        # Common test file patterns
        test_names = [
            f"test_{module_name}.py",
            f"{module_name}_test.py",
            f"test_{module_name}_*.py",
        ]

        # Search in common test directories
        test_dirs = [
            self.project_root / "tests",
            self.project_root / "test",
            source_path.parent / "tests",
            source_path.parent,
        ]

        for test_dir in test_dirs:
            if test_dir.exists():
                for test_name_pattern in test_names:
                    # Search recursively in test directories
                    if "*" in test_name_pattern:
                        # Use rglob for recursive pattern matching
                        for test_file in test_dir.rglob(test_name_pattern):
                            if test_file.is_file() and self._imports_module_ast(test_file, source_file):
                                test_files.append(str(test_file))
                    else:
                        # Use rglob for recursive exact matching
                        for test_file in test_dir.rglob(test_name_pattern):
                            if test_file.is_file() and self._imports_module_ast(test_file, source_file):
                                test_files.append(str(test_file))

        return list(set(test_files))  # Remove duplicates

    def _imports_module_ast(self, test_file: Path, source_file: str) -> bool:
        """Check if test file imports the source module using AST analysis."""
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the file into an AST
            tree = ast.parse(content, filename=str(test_file))
            
            # Get the module name and possible import paths
            source_path = Path(source_file)
            module_name = source_path.stem
            
            # Calculate possible module paths relative to project root
            possible_import_paths = self._calculate_import_paths(source_path, module_name)
            
            # Walk through all nodes in the AST
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    # Handle 'import module' statements
                    for alias in node.names:
                        if self._matches_any_import_path(alias.name, possible_import_paths):
                            return True
                            
                elif isinstance(node, ast.ImportFrom):
                    # Handle 'from module import ...' statements
                    if node.module:
                        # Direct module import
                        if self._matches_any_import_path(node.module, possible_import_paths):
                            return True
                        
                        # Check if importing from a package that contains our module
                        for name in node.names:
                            full_import = f"{node.module}.{name.name}" if name.name != '*' else node.module
                            if self._matches_any_import_path(full_import, possible_import_paths):
                                return True
                    else:
                        # Relative imports (from . import ...)
                        for name in node.names:
                            if name.name == module_name or name.name in possible_import_paths:
                                return True

        except Exception as e:
            logger.debug(f"Failed to analyze imports in {test_file}: {e}")

        return False

    def _calculate_import_paths(self, source_path: Path, module_name: str) -> Set[str]:
        """Calculate possible import paths for a source module."""
        import_paths = {module_name}
        
        try:
            # Get relative path from project root
            relative_path = source_path.relative_to(self.project_root)
            
            # Convert file path to module path
            # e.g., src/package/module.py -> package.module
            parts = list(relative_path.parts[:-1])  # Remove filename
            if parts and parts[0] == 'src':
                parts = parts[1:]  # Remove 'src' prefix
            
            if parts:
                module_path = '.'.join(parts + [module_name])
                import_paths.add(module_path)
                
                # Also add intermediate paths
                # e.g., for package.subpackage.module, also check package.subpackage
                for i in range(1, len(parts) + 1):
                    partial_path = '.'.join(parts[:i])
                    import_paths.add(partial_path)
                    
        except ValueError:
            # source_path is not relative to project_root
            pass
            
        return import_paths

    def _matches_any_import_path(self, import_name: str, possible_paths: Set[str]) -> bool:
        """Check if an import name matches any of the possible import paths."""
        if import_name in possible_paths:
            return True
            
        # Check if the import name ends with any of our possible paths
        for path in possible_paths:
            if import_name.endswith(f".{path}") or import_name.endswith(path):
                return True
                
        return False

    def analyze_test_completeness(self, source_file: str, test_files: List[str]) -> Dict[str, Set[str]]:
        """Analyze what's already tested using comprehensive AST analysis."""
        tested_elements = {
            'functions': set(),
            'classes': set(),
            'methods': set()
        }

        # Get the source module elements to compare against
        source_elements = self._extract_source_elements(source_file)

        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read(), filename=test_file)

                # Analyze each test function/method
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if self._is_test_function(node):
                            tested_in_function = self._analyze_test_function(node, source_elements)
                            
                            # Merge results
                            for category, elements in tested_in_function.items():
                                tested_elements[category].update(elements)

            except Exception as e:
                logger.error(f"Failed to analyze test file {test_file}: {e}")

        return tested_elements

    def _extract_source_elements(self, source_file: str) -> Dict[str, Set[str]]:
        """Extract all testable elements from the source file."""
        elements = {
            'functions': set(),
            'classes': set(),
            'methods': set()
        }
        
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=source_file)
            
            # Process top-level nodes only
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    elements['classes'].add(node.name)
                    
                    # Extract methods from the class
                    for class_node in node.body:
                        if isinstance(class_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if not class_node.name.startswith('_') or class_node.name in ['__init__', '__str__', '__repr__']:
                                elements['methods'].add(f"{node.name}.{class_node.name}")
                                
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Top-level functions only
                    if not node.name.startswith('_') or node.name.startswith('__'):
                        elements['functions'].add(node.name)
                            
        except Exception as e:
            logger.error(f"Failed to extract source elements from {source_file}: {e}")
            
        return elements

    def _is_test_function(self, node: ast.FunctionDef) -> bool:
        """Determine if a function is a test function."""
        # Check function name patterns
        if node.name.startswith('test_') or node.name.endswith('_test'):
            return True
        
        # Check for pytest decorators
        for decorator in node.decorator_list:
            if self._is_pytest_decorator(decorator):
                return True
                
        return False

    def _is_pytest_decorator(self, decorator) -> bool:
        """Check if a decorator is a pytest decorator."""
        if isinstance(decorator, ast.Name):
            # Simple decorator like @parametrize
            return decorator.id in ['parametrize', 'fixture', 'mark']
        
        elif isinstance(decorator, ast.Attribute):
            # Attribute decorator like @pytest.mark.parametrize or @mark.parametrize
            if isinstance(decorator.value, ast.Name):
                # @mark.parametrize
                if decorator.value.id == 'mark':
                    return True
                # @pytest.fixture, @pytest.parametrize, etc.
                if decorator.value.id == 'pytest':
                    return True
                    
            elif isinstance(decorator.value, ast.Attribute):
                # @pytest.mark.parametrize
                if (isinstance(decorator.value.value, ast.Name) and 
                    decorator.value.value.id == 'pytest' and 
                    decorator.value.attr == 'mark'):
                    return True
        
        elif isinstance(decorator, ast.Call):
            # Decorator with arguments like @pytest.mark.parametrize(...)
            return self._is_pytest_decorator(decorator.func)
            
        return False

    def _analyze_test_function(self, test_node: ast.FunctionDef, source_elements: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Analyze a single test function to determine what it tests."""
        tested_elements = {
            'functions': set(),
            'classes': set(),
            'methods': set()
        }
        
        # Extract tested elements from test name
        test_name = test_node.name
        if test_name.startswith('test_'):
            potential_target = test_name[5:]  # Remove 'test_' prefix
            self._match_test_name_to_elements(potential_target, source_elements, tested_elements)
        
        # Analyze function calls within the test
        for node in ast.walk(test_node):
            if isinstance(node, ast.Call):
                self._analyze_function_call(node, source_elements, tested_elements)
            elif isinstance(node, ast.Attribute):
                self._analyze_attribute_access(node, source_elements, tested_elements)
        
        return tested_elements

    def _match_test_name_to_elements(self, test_target: str, source_elements: Dict[str, Set[str]], tested_elements: Dict[str, Set[str]]):
        """Match test name components to source elements."""
        test_target_lower = test_target.lower()
        
        # Direct name matches
        for func_name in source_elements['functions']:
            if func_name.lower() == test_target_lower or test_target_lower in func_name.lower():
                tested_elements['functions'].add(func_name)
        
        for class_name in source_elements['classes']:
            if class_name.lower() == test_target_lower or test_target_lower in class_name.lower():
                tested_elements['classes'].add(class_name)
        
        for method_name in source_elements['methods']:
            method_simple = method_name.split('.')[-1]  # Get just the method name without class
            if method_simple.lower() == test_target_lower or test_target_lower in method_simple.lower():
                tested_elements['methods'].add(method_name)

    def _analyze_function_call(self, call_node: ast.Call, source_elements: Dict[str, Set[str]], tested_elements: Dict[str, Set[str]]):
        """Analyze a function call to see if it calls source elements."""
        if isinstance(call_node.func, ast.Name):
            # Direct function call: func() or Class()
            func_name = call_node.func.id
            
            # Check if it's a function call
            if func_name in source_elements['functions']:
                tested_elements['functions'].add(func_name)
            
            # Check if it's a class instantiation (constructor call)
            elif func_name in source_elements['classes']:
                tested_elements['classes'].add(func_name)
                
        elif isinstance(call_node.func, ast.Attribute):
            # Method call: obj.method() or module.func()
            if isinstance(call_node.func.value, ast.Name):
                # obj.method()
                obj_name = call_node.func.value.id
                method_name = call_node.func.attr
                
                # Check if this matches a class method
                full_method = f"{obj_name}.{method_name}"
                for source_method in source_elements['methods']:
                    if source_method.endswith(f".{method_name}"):
                        tested_elements['methods'].add(source_method)
                        
                # Check if the object name is a class being tested
                if obj_name in source_elements['classes']:
                    tested_elements['classes'].add(obj_name)

    def _analyze_attribute_access(self, attr_node: ast.Attribute, source_elements: Dict[str, Set[str]], tested_elements: Dict[str, Set[str]]):
        """Analyze attribute access that might indicate testing."""
        if isinstance(attr_node.value, ast.Name):
            obj_name = attr_node.value.id
            attr_name = attr_node.attr
            
            # Check if accessing a method/attribute of a source class
            if obj_name in source_elements['classes']:
                tested_elements['classes'].add(obj_name)
                
                # Check if the attribute is a known method
                full_method = f"{obj_name}.{attr_name}"
                if full_method in source_elements['methods']:
                    tested_elements['methods'].add(full_method)

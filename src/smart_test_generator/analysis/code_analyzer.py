"""AST-based code analysis."""

import ast
import logging
from pathlib import Path
from typing import List

from smart_test_generator.models.data_models import TestableElement

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """Deep analysis of source code structure."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def extract_testable_elements(self, filepath: str) -> List[TestableElement]:
        """Extract all testable elements from a file."""
        elements = []

        try:
            with open(filepath, 'r') as f:
                content = f.read()
                tree = ast.parse(content, filename=filepath)

            # Extract module-level functions
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith('_') or node.name.startswith('__'):
                        elements.append(self._analyze_function(node, filepath))

                elif isinstance(node, ast.ClassDef):
                    # Extract class and its methods
                    for method in node.body:
                        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if not method.name.startswith('_') or method.name in ['__init__', '__str__', '__repr__']:
                                elements.append(self._analyze_method(method, node.name, filepath))

        except Exception as e:
            logger.error(f"Failed to extract testable elements from {filepath}: {e}")

        return elements

    def _analyze_function(self, node: ast.FunctionDef, filepath: str) -> TestableElement:
        """Analyze a function node."""
        return TestableElement(
            name=node.name,
            type='function',
            filepath=filepath,
            line_number=node.lineno,
            signature=self._get_signature(node),
            docstring=ast.get_docstring(node),
            complexity=self._calculate_complexity(node),
            dependencies=self._extract_dependencies(node),
            decorators=[d.id for d in node.decorator_list if isinstance(d, ast.Name)],
            raises=self._extract_raises(node)
        )

    def _analyze_method(self, node: ast.FunctionDef, class_name: str, filepath: str) -> TestableElement:
        """Analyze a method node."""
        return TestableElement(
            name=f"{class_name}.{node.name}",
            type='method',
            filepath=filepath,
            line_number=node.lineno,
            signature=self._get_signature(node),
            docstring=ast.get_docstring(node),
            complexity=self._calculate_complexity(node),
            dependencies=self._extract_dependencies(node),
            decorators=[d.id for d in node.decorator_list if isinstance(d, ast.Name)],
            raises=self._extract_raises(node)
        )

    def _get_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature."""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        return f"{node.name}({', '.join(args)})"

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
        return complexity

    def _extract_dependencies(self, node: ast.FunctionDef) -> List[str]:
        """Extract external dependencies called in function."""
        deps = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    deps.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    deps.append(f"{child.func.value.id if isinstance(child.func.value, ast.Name) else 'unknown'}.{child.func.attr}")
        return list(set(deps))

    def _extract_raises(self, node: ast.FunctionDef) -> List[str]:
        """Extract exceptions raised in function."""
        raises = []
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                if isinstance(child.exc, ast.Call) and isinstance(child.exc.func, ast.Name):
                    raises.append(child.exc.func.id)
        return list(set(raises))

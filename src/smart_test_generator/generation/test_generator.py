"""Incremental test generation logic."""

import logging
from pathlib import Path
from typing import List, Optional

from smart_test_generator.models.data_models import TestGenerationPlan, TestCoverage, TestableElement
from smart_test_generator.config import Config
from smart_test_generator.analysis.code_analyzer import CodeAnalyzer
from smart_test_generator.analysis.test_mapper import TestMapper

logger = logging.getLogger(__name__)


class IncrementalTestGenerator:
    """Generate only missing tests."""

    def __init__(self, project_root: Path, config: Config):
        self.project_root = project_root
        self.config = config
        self.code_analyzer = CodeAnalyzer(project_root)
        self.test_mapper = TestMapper(project_root, config)

    def generate_test_plan(self, source_file: str, coverage: Optional[TestCoverage]) -> TestGenerationPlan:
        """Create a plan for what tests to generate."""
        # Find existing test files
        existing_tests = self.test_mapper.find_test_files(source_file)

        # Extract all testable elements
        all_elements = self.code_analyzer.extract_testable_elements(source_file)

        # Analyze what's already tested
        tested_elements = self.test_mapper.analyze_test_completeness(source_file, existing_tests)

        # Identify elements that need tests
        elements_to_test = []
        for element in all_elements:
            element_tested = False

            # Check if element is already tested
            if element.type == 'function' and element.name in tested_elements['functions']:
                element_tested = True
            elif element.type == 'method':
                method_name = element.name.split('.')[-1]
                if method_name in tested_elements['methods']:
                    element_tested = True

            # Add elements that need tests
            if not element_tested:
                # If no existing test files, we need tests for all untested elements
                if not existing_tests:
                    elements_to_test.append(element)
                # If existing test files exist, use coverage data to determine priority
                elif coverage and element.name in coverage.uncovered_functions:
                    elements_to_test.append(element)
                # If no coverage data but existing tests, be more conservative
                elif not coverage:
                    # Only add elements if we're confident they're not tested
                    # Check if the element name suggests it might already be tested
                    if not self._might_be_tested(element, existing_tests):
                        elements_to_test.append(element)

        # Estimate coverage improvement
        estimated_coverage = self._estimate_coverage_improvement(
            len(all_elements), len(elements_to_test), coverage
        )

        return TestGenerationPlan(
            source_file=source_file,
            existing_test_files=existing_tests,
            elements_to_test=elements_to_test,
            coverage_before=coverage,
            estimated_coverage_after=estimated_coverage
        )

    def _estimate_coverage_improvement(self, total_elements: int, new_elements: int,
                                       current_coverage: Optional[TestCoverage]) -> float:
        """Estimate coverage after generating new tests."""
        if not current_coverage:
            return (new_elements / total_elements * 100) if total_elements > 0 else 0

        current = current_coverage.line_coverage
        improvement = (new_elements / total_elements * 100) if total_elements > 0 else 0
        return min(100, current + improvement * 0.8)  # Conservative estimate

    def _might_be_tested(self, element: TestableElement, existing_tests: List[str]) -> bool:
        """Conservative check if an element might already be tested."""
        if not existing_tests:
            return False
            
        element_name = element.name
        if '.' in element_name:
            element_name = element_name.split('.')[-1]  # Get just the method name
            
        # Check if any test file contains test functions that might test this element
        for test_file in existing_tests:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for test functions that might test this element
                    if f"test_{element_name.lower()}" in content.lower():
                        return True
                    if f"test{element_name}" in content:
                        return True
                    # Check if element name appears in test file (might be called/tested)
                    if element_name in content:
                        return True
            except Exception:
                continue
                
        return False

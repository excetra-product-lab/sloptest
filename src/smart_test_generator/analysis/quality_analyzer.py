"""Test quality analysis framework."""

import ast
import re
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple

from smart_test_generator.models.data_models import (
    QualityDimension, QualityScore, TestQualityReport
)

logger = logging.getLogger(__name__)


class QualityAnalyzer(ABC):
    """Abstract base class for quality analyzers."""
    
    @abstractmethod
    def analyze(self, test_code: str, source_code: str = "") -> QualityScore:
        """Analyze test quality for a specific dimension."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> QualityDimension:
        """Get the quality dimension this analyzer handles."""
        pass


class EdgeCaseAnalyzer(QualityAnalyzer):
    """Analyze edge case coverage in tests."""
    
    def __init__(self):
        self.edge_case_patterns = {
            'null_checks': [
                r'is\s+None', r'==\s*None', r'!=\s*None',
                r'assert.*None', r'assertIsNone', r'assertIsNotNone'
            ],
            'empty_collections': [
                r'len\([^)]+\)\s*==\s*0', r'not\s+\w+', r'==\s*\[\]',
                r'==\s*""', r'==\s*\'\'', r'assertEmpty', r'assertEqual.*\[\]'
            ],
            'boundary_values': [
                r'==\s*0', r'==\s*1', r'==\s*-1',
                r'>\s*0', r'<\s*0', r'>=\s*1', r'<=\s*0',
                r'sys\.maxsize', r'float\(\'inf\'\)', r'float\(\'-inf\'\)'
            ],
            'type_validation': [
                r'isinstance\(', r'type\(', r'hasattr\(',
                r'assertIsInstance', r'assertRaises.*TypeError'
            ],
            'exception_handling': [
                r'pytest\.raises', r'assertRaises', r'with\s+self\.assertRaises',
                r'try:', r'except:', r'raise'
            ],
            'floating_point': [
                r'pytest\.approx', r'assertAlmostEqual', r'math\.isclose',
                r'abs\([^)]+\)\s*<\s*\d*\.?\d*e?-?\d*'
            ]
        }
    
    def get_dimension(self) -> QualityDimension:
        return QualityDimension.EDGE_CASE_COVERAGE
    
    def analyze(self, test_code: str, source_code: str = "") -> QualityScore:
        """Analyze edge case coverage."""
        found_patterns = {}
        total_score = 0
        max_possible = 0
        
        for category, patterns in self.edge_case_patterns.items():
            found_in_category = []
            for pattern in patterns:
                matches = re.findall(pattern, test_code, re.IGNORECASE | re.MULTILINE)
                if matches:
                    found_in_category.extend(matches)
            
            # Score based on category coverage
            category_score = min(len(found_in_category) * 10, 20)  # Max 20 per category
            total_score += category_score
            max_possible += 20
            found_patterns[category] = found_in_category
        
        # Bonus for comprehensive edge case coverage
        covered_categories = sum(1 for patterns in found_patterns.values() if patterns)
        if covered_categories >= 4:
            total_score += 20  # Bonus for covering multiple categories
            max_possible += 20
        
        final_score = min((total_score / max_possible) * 100, 100) if max_possible > 0 else 0
        
        # Generate suggestions
        suggestions = []
        missing_categories = [cat for cat, patterns in found_patterns.items() if not patterns]
        for category in missing_categories[:3]:  # Top 3 missing
            suggestions.append(self._get_suggestion_for_category(category))
        
        return QualityScore(
            dimension=self.get_dimension(),
            score=final_score,
            details={
                'found_patterns': found_patterns,
                'covered_categories': covered_categories,
                'total_categories': len(self.edge_case_patterns)
            },
            suggestions=suggestions
        )
    
    def _get_suggestion_for_category(self, category: str) -> str:
        """Get improvement suggestion for a missing category."""
        suggestions = {
            'null_checks': "Add tests for None/null inputs using 'assert value is None' or similar",
            'empty_collections': "Test empty lists, strings, and collections with 'assert len(result) == 0'",
            'boundary_values': "Test boundary conditions like 0, 1, -1, min/max values",
            'type_validation': "Add type checking tests using 'isinstance()' or 'assertIsInstance()'",
            'exception_handling': "Test error conditions with 'pytest.raises()' or 'assertRaises()'",
            'floating_point': "Use 'pytest.approx()' for floating-point comparisons"
        }
        return suggestions.get(category, f"Consider adding tests for {category}")


class AssertionStrengthAnalyzer(QualityAnalyzer):
    """Analyze the strength and specificity of test assertions."""
    
    def __init__(self):
        self.weak_patterns = [
            r'assert\s+\w+\s*$',  # assert something (no comparison)
            r'assert\s+\w+\s+is\s+not\s+None',
            r'assert\s+\w+\s*!=\s*None',
            r'assert\s+len\([^)]+\)\s*>\s*0',
            r'assert\s+\w+',  # Basic assert without comparison
        ]
        
        self.strong_patterns = [
            r'assert\s+\w+\s*==\s*[^N][^o][^n][^e]',  # Specific equality (not None)
            r'assert\s+\w+\s+in\s+',
            r'assert\s+\w+\s*<\s*\d+',
            r'assert\s+\w+\s*>\s*\d+',
            r'assertEqual\s*\(',
            r'assertIn\s*\(',
            r'assertGreater\s*\(',
            r'assertLess\s*\(',
        ]
    
    def get_dimension(self) -> QualityDimension:
        return QualityDimension.ASSERTION_STRENGTH
    
    def analyze(self, test_code: str, source_code: str = "") -> QualityScore:
        """Analyze assertion strength."""
        # Count all assertions
        all_assertions = re.findall(r'assert\w*\s*\(|assert\s+', test_code, re.IGNORECASE)
        total_assertions = len(all_assertions)
        
        if total_assertions == 0:
            return QualityScore(
                dimension=self.get_dimension(),
                score=0.0,
                details={'total_assertions': 0},
                suggestions=["Add assertions to verify test expectations"]
            )
        
        # Count weak assertions
        weak_count = 0
        for pattern in self.weak_patterns:
            weak_count += len(re.findall(pattern, test_code, re.IGNORECASE))
        
        # Count strong assertions
        strong_count = 0
        for pattern in self.strong_patterns:
            strong_count += len(re.findall(pattern, test_code, re.IGNORECASE))
        
        # Calculate score (higher strong/total ratio = better score)
        if total_assertions > 0:
            strong_ratio = strong_count / total_assertions
            weak_ratio = weak_count / total_assertions
            
            # Score based on strong assertions and penalty for weak ones
            score = (strong_ratio * 80) + max(0, (1 - weak_ratio) * 20)
        else:
            score = 0
        
        # Check for assertion messages
        assertion_messages = re.findall(r'assert.*,\s*["\'][^"\']+["\']', test_code)
        if assertion_messages:
            score += min(len(assertion_messages) * 5, 15)  # Bonus for error messages
        
        suggestions = []
        if weak_ratio > 0.3:
            suggestions.append("Replace generic assertions like 'assert result' with specific comparisons")
        if strong_ratio < 0.5:
            suggestions.append("Use more specific assertions like 'assertEqual()' instead of generic 'assert'")
        if not assertion_messages:
            suggestions.append("Add descriptive messages to assertions for better test failure diagnosis")
        
        return QualityScore(
            dimension=self.get_dimension(),
            score=min(score, 100),
            details={
                'total_assertions': total_assertions,
                'strong_assertions': strong_count,
                'weak_assertions': weak_count,
                'assertion_messages': len(assertion_messages)
            },
            suggestions=suggestions
        )


class MaintainabilityAnalyzer(QualityAnalyzer):
    """Analyze test maintainability and readability."""
    
    def get_dimension(self) -> QualityDimension:
        return QualityDimension.MAINTAINABILITY
    
    def analyze(self, test_code: str, source_code: str = "") -> QualityScore:
        """Analyze test maintainability."""
        score = 0
        max_score = 100
        details = {}
        suggestions = []
        
        # Check for docstrings
        docstring_count = len(re.findall(r'"""[^"]*"""', test_code))
        if docstring_count > 0:
            score += 15
            details['has_docstrings'] = True
        else:
            suggestions.append("Add docstrings to test functions explaining what they test")
            details['has_docstrings'] = False
        
        # Check for Arrange-Act-Assert pattern
        aaa_patterns = [
            r'#\s*[Aa]rrange', r'#\s*[Aa]ct', r'#\s*[Aa]ssert',
            r'#\s*[Gg]iven', r'#\s*[Ww]hen', r'#\s*[Tt]hen'
        ]
        aaa_found = sum(1 for pattern in aaa_patterns if re.search(pattern, test_code))
        if aaa_found >= 2:
            score += 15
            details['follows_aaa_pattern'] = True
        else:
            suggestions.append("Use Arrange-Act-Assert pattern with comments for better structure")
            details['follows_aaa_pattern'] = False
        
        # Check for magic numbers
        magic_numbers = re.findall(r'\b\d{2,}\b', test_code)  # Numbers with 2+ digits
        magic_number_penalty = min(len(magic_numbers) * 2, 15)
        score += max(0, 15 - magic_number_penalty)
        if len(magic_numbers) > 3:
            suggestions.append("Replace magic numbers with named constants for better readability")
        details['magic_numbers'] = len(magic_numbers)
        
        # Check for descriptive variable names
        variables = re.findall(r'(\w+)\s*=', test_code)
        descriptive_vars = [v for v in variables if len(v) > 3 and not v.startswith('_')]
        if variables:
            descriptive_ratio = len(descriptive_vars) / len(variables)
            score += descriptive_ratio * 15
            if descriptive_ratio < 0.7:
                suggestions.append("Use more descriptive variable names instead of short abbreviations")
        details['descriptive_variable_ratio'] = len(descriptive_vars) / len(variables) if variables else 0
        
        # Check test function naming
        test_functions = re.findall(r'def\s+(test_\w+)', test_code)
        well_named_tests = [f for f in test_functions if len(f) > 10]  # Reasonable length
        if test_functions:
            naming_ratio = len(well_named_tests) / len(test_functions)
            score += naming_ratio * 15
            if naming_ratio < 0.8:
                suggestions.append("Use descriptive test names that explain what behavior is being tested")
        details['well_named_tests_ratio'] = len(well_named_tests) / len(test_functions) if test_functions else 0
        
        # Check for code duplication (simplified)
        lines = test_code.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
        unique_lines = set(non_empty_lines)
        if non_empty_lines:
            duplication_ratio = 1 - (len(unique_lines) / len(non_empty_lines))
            score += max(0, (1 - duplication_ratio) * 15)
            if duplication_ratio > 0.3:
                suggestions.append("Reduce code duplication by extracting common setup into fixtures or helper methods")
        details['code_duplication_ratio'] = duplication_ratio if non_empty_lines else 0
        
        # Check for appropriate test size
        line_count = len([line for line in lines if line.strip()])
        if line_count > 0:
            if line_count < 200:  # Reasonable test file size
                score += 10
            else:
                suggestions.append("Consider splitting large test files into smaller, focused test modules")
        details['line_count'] = line_count
        
        return QualityScore(
            dimension=self.get_dimension(),
            score=min(score, max_score),
            details=details,
            suggestions=suggestions
        )


class BugDetectionAnalyzer(QualityAnalyzer):
    """Analyze potential for detecting real bugs."""
    
    def get_dimension(self) -> QualityDimension:
        return QualityDimension.BUG_DETECTION_POTENTIAL
    
    def analyze(self, test_code: str, source_code: str = "") -> QualityScore:
        """Analyze bug detection potential."""
        score = 0
        details = {}
        suggestions = []
        
        # Check for error condition testing
        error_patterns = [
            r'pytest\.raises', r'assertRaises', r'with\s+self\.assertRaises',
            r'try:', r'except:', r'raise'
        ]
        error_tests = sum(len(re.findall(pattern, test_code, re.IGNORECASE)) for pattern in error_patterns)
        if error_tests > 0:
            score += min(error_tests * 10, 25)
            details['tests_error_conditions'] = True
        else:
            suggestions.append("Add tests for error conditions and exception handling")
            details['tests_error_conditions'] = False
        
        # Check for multiple test scenarios per function
        test_functions = re.findall(r'def\s+test_\w+.*?(?=def|\Z)', test_code, re.DOTALL)
        multi_scenario_tests = 0
        for test_func in test_functions:
            assertions = len(re.findall(r'assert', test_func))
            if assertions > 2:  # Multiple assertions suggest multiple scenarios
                multi_scenario_tests += 1
        
        if test_functions:
            scenario_ratio = multi_scenario_tests / len(test_functions)
            score += scenario_ratio * 20
            if scenario_ratio < 0.5:
                suggestions.append("Test multiple scenarios within each test function")
        details['multi_scenario_test_ratio'] = scenario_ratio if test_functions else 0
        
        # Check for state verification (not just return value testing)
        state_patterns = [
            r'\.called', r'\.call_count', r'\.call_args',  # Mock verification
            r'\.assert_called', r'\.assert_called_with',
            r'len\(', r'\.count\(', r'\.index\(',  # Collection state
        ]
        state_verifications = sum(len(re.findall(pattern, test_code)) for pattern in state_patterns)
        if state_verifications > 0:
            score += min(state_verifications * 5, 20)
            details['verifies_state_changes'] = True
        else:
            suggestions.append("Verify state changes and side effects, not just return values")
            details['verifies_state_changes'] = False
        
        # Check for boundary and edge case testing
        boundary_patterns = [
            r'==\s*0', r'==\s*1', r'==\s*-1',
            r'empty', r'None', r'null',
            r'min', r'max', r'inf'
        ]
        boundary_tests = sum(len(re.findall(pattern, test_code, re.IGNORECASE)) for pattern in boundary_patterns)
        if boundary_tests > 0:
            score += min(boundary_tests * 3, 20)
            details['tests_boundaries'] = True
        else:
            suggestions.append("Add boundary value testing (0, 1, -1, min, max, empty, None)")
            details['tests_boundaries'] = False
        
        # Check for integration-like testing (testing interactions)
        integration_patterns = [
            r'mock', r'patch', r'Mock\(',
            r'setUp', r'tearDown', r'fixture',
            r'database', r'db', r'session'
        ]
        integration_indicators = sum(len(re.findall(pattern, test_code, re.IGNORECASE)) for pattern in integration_patterns)
        if integration_indicators > 0:
            score += min(integration_indicators * 2, 15)
            details['tests_interactions'] = True
        else:
            suggestions.append("Consider testing component interactions and dependencies")
            details['tests_interactions'] = False
        
        return QualityScore(
            dimension=self.get_dimension(),
            score=min(score, 100),
            details=details,
            suggestions=suggestions
        )


class TestQualityEngine:
    """Main engine for comprehensive test quality analysis."""
    
    def __init__(self, custom_analyzers: Optional[List[QualityAnalyzer]] = None):
        """Initialize with default or custom analyzers."""
        self.analyzers = custom_analyzers or [
            EdgeCaseAnalyzer(),
            AssertionStrengthAnalyzer(),
            MaintainabilityAnalyzer(),
            BugDetectionAnalyzer()
        ]
        
        # Weights for different quality dimensions
        self.dimension_weights = {
            QualityDimension.EDGE_CASE_COVERAGE: 0.25,
            QualityDimension.ASSERTION_STRENGTH: 0.20,
            QualityDimension.MAINTAINABILITY: 0.20,
            QualityDimension.BUG_DETECTION_POTENTIAL: 0.30,
            QualityDimension.READABILITY: 0.05,
            QualityDimension.INDEPENDENCE: 0.00  # Not implemented yet
        }
    
    def analyze_test_quality(self, test_file: str, source_file: str = "") -> TestQualityReport:
        """Perform comprehensive quality analysis of a test file."""
        try:
            # Read test file
            with open(test_file, 'r', encoding='utf-8') as f:
                test_code = f.read()
            
            # Read source file if provided
            source_code = ""
            if source_file and Path(source_file).exists():
                with open(source_file, 'r', encoding='utf-8') as f:
                    source_code = f.read()
            
            return self._analyze_code_quality(test_file, test_code, source_code)
            
        except Exception as e:
            logger.error(f"Failed to analyze test quality for {test_file}: {e}")
            return self._create_empty_report(test_file)
    
    def analyze_test_code_quality(self, test_code: str, test_file: str = "unknown", source_code: str = "") -> TestQualityReport:
        """Analyze quality of test code directly."""
        return self._analyze_code_quality(test_file, test_code, source_code)
    
    def _analyze_code_quality(self, test_file: str, test_code: str, source_code: str) -> TestQualityReport:
        """Internal method to analyze test code quality."""
        dimension_scores = {}
        all_suggestions = []
        all_priority_fixes = []
        
        # Run each analyzer
        for analyzer in self.analyzers:
            try:
                score = analyzer.analyze(test_code, source_code)
                dimension_scores[score.dimension] = score
                all_suggestions.extend(score.suggestions)
                
                # Add priority fixes for low scores
                if score.score < 50:
                    all_priority_fixes.extend(score.suggestions[:2])
                    
            except Exception as e:
                logger.warning(f"Analyzer {analyzer.__class__.__name__} failed: {e}")
                # Create a zero score for failed analysis
                dimension_scores[analyzer.get_dimension()] = QualityScore(
                    dimension=analyzer.get_dimension(),
                    score=0.0,
                    suggestions=[f"Analysis failed: {str(e)}"]
                )
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(dimension_scores)
        
        # Analyze specific quality aspects
        edge_cases_found, edge_cases_missing = self._analyze_edge_cases(test_code)
        weak_assertions = self._analyze_weak_assertions(test_code)
        maintainability_issues = self._analyze_maintainability_issues(test_code)
        
        return TestQualityReport(
            test_file=test_file,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            edge_cases_found=edge_cases_found,
            edge_cases_missing=edge_cases_missing,
            weak_assertions=weak_assertions,
            maintainability_issues=maintainability_issues,
            improvement_suggestions=list(set(all_suggestions)),  # Remove duplicates
            priority_fixes=list(set(all_priority_fixes))
        )
    
    def _calculate_overall_score(self, dimension_scores: Dict[QualityDimension, QualityScore]) -> float:
        """Calculate weighted overall quality score."""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for dimension, weight in self.dimension_weights.items():
            if dimension in dimension_scores and weight > 0:
                total_weighted_score += dimension_scores[dimension].score * weight
                total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _analyze_edge_cases(self, test_code: str) -> Tuple[List[str], List[str]]:
        """Analyze edge cases found and missing."""
        found = []
        missing = []
        
        # Common edge cases to look for
        edge_case_checks = {
            'None/null values': r'is\s+None|==\s*None|assertIsNone',
            'Empty collections': r'==\s*\[\]|==\s*""|\blen\([^)]+\)\s*==\s*0',
            'Zero values': r'==\s*0\b',
            'Negative values': r'==\s*-\d+|<\s*0',
            'Boundary values': r'==\s*1\b|==\s*-1\b',
            'Exception handling': r'pytest\.raises|assertRaises'
        }
        
        for case_name, pattern in edge_case_checks.items():
            if re.search(pattern, test_code, re.IGNORECASE):
                found.append(case_name)
            else:
                missing.append(case_name)
        
        return found, missing
    
    def _analyze_weak_assertions(self, test_code: str) -> List[Dict[str, Any]]:
        """Find weak assertions that could be improved."""
        weak_assertions = []
        
        # Find basic assert statements
        basic_asserts = re.finditer(r'assert\s+(\w+)\s*$', test_code, re.MULTILINE)
        for match in basic_asserts:
            weak_assertions.append({
                'type': 'basic_assert',
                'line': test_code[:match.start()].count('\n') + 1,
                'code': match.group(0),
                'suggestion': f"Replace 'assert {match.group(1)}' with specific comparison"
            })
        
        # Find is not None assertions
        not_none_asserts = re.finditer(r'assert\s+\w+\s+is\s+not\s+None', test_code, re.IGNORECASE)
        for match in not_none_asserts:
            weak_assertions.append({
                'type': 'not_none_assert',
                'line': test_code[:match.start()].count('\n') + 1,
                'code': match.group(0),
                'suggestion': "Consider testing the actual expected value instead of just 'not None'"
            })
        
        return weak_assertions
    
    def _analyze_maintainability_issues(self, test_code: str) -> List[str]:
        """Identify maintainability issues."""
        issues = []
        
        # Check for long test functions
        test_functions = re.findall(r'def\s+test_\w+.*?(?=def|\Z)', test_code, re.DOTALL)
        for i, func in enumerate(test_functions):
            lines = len([line for line in func.split('\n') if line.strip()])
            if lines > 30:
                issues.append(f"Test function {i+1} is too long ({lines} lines) - consider splitting")
        
        # Check for missing docstrings
        functions_without_docs = re.findall(r'def\s+test_\w+[^:]*:\s*\n(?!\s*""")', test_code)
        if functions_without_docs:
            issues.append(f"{len(functions_without_docs)} test functions missing docstrings")
        
        # Check for magic numbers
        magic_numbers = re.findall(r'\b\d{3,}\b', test_code)  # Numbers with 3+ digits
        if len(magic_numbers) > 5:
            issues.append(f"Too many magic numbers ({len(magic_numbers)}) - consider using named constants")
        
        return issues
    
    def _create_empty_report(self, test_file: str) -> TestQualityReport:
        """Create an empty report for failed analysis."""
        return TestQualityReport(
            test_file=test_file,
            overall_score=0.0,
            improvement_suggestions=["Analysis failed - check file format and syntax"],
            priority_fixes=["Fix syntax errors or file accessibility issues"]
        ) 
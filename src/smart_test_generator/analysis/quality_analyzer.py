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
            BugDetectionAnalyzer(),
            IndependenceAnalyzer()
        ]
        
        # Weights for different quality dimensions
        self.dimension_weights = {
            QualityDimension.EDGE_CASE_COVERAGE: 0.22,
            QualityDimension.ASSERTION_STRENGTH: 0.18,
            QualityDimension.MAINTAINABILITY: 0.18,
            QualityDimension.BUG_DETECTION_POTENTIAL: 0.27,
            QualityDimension.READABILITY: 0.05,
            QualityDimension.INDEPENDENCE: 0.10  # Newly implemented
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
    
    def _create_empty_report(self, test_file: str) -> TestQualityReport:
        """Create an empty quality report for failed analysis."""
        return TestQualityReport(
            test_file=test_file,
            overall_score=0.0,
            dimension_scores={},
            edge_cases_found=[],
            edge_cases_missing=[],
            weak_assertions=[],
            maintainability_issues=["Analysis failed - unable to process test file"],
            improvement_suggestions=["Analysis failed - fix test file errors and re-run analysis"],
            priority_fixes=["Fix test file accessibility and syntax errors"]
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


class IndependenceAnalyzer(QualityAnalyzer):
    """Analyze test independence and isolation."""
    
    def __init__(self):
        # Patterns for detecting shared state and dependencies
        self.shared_state_patterns = {
            'class_variables': [
                r'self\.__class__\.\w+\s*=',  # Class attribute modification
                r'class\s+\w+.*:\s*\n\s+\w+\s*=.*\[\]',  # Shared class collections
                r'class\s+\w+.*:\s*\n\s+\w+\s*=.*\{\}',  # Shared class dicts
            ],
            'global_variables': [
                r'global\s+\w+',  # Global declarations
                r'import.*as\s+\w+.*\n.*\w+\.\w+\s*=',  # Module attribute modification
            ],
            'module_state': [
                r'sys\.modules\[',  # Module manipulation
                r'importlib\.reload',  # Module reloading
                r'os\.environ\[',  # Environment modification
                r'os\.chdir\(',  # Directory changes
            ],
            'singleton_patterns': [
                r'\.instance\s*=',  # Singleton instance modification
                r'\.getInstance\(\)',  # Singleton access
                r'@.*singleton',  # Singleton decorators
            ]
        }
        
        self.dependency_patterns = {
            'execution_order': [
                r'#.*previous.*test',  # References to previous tests in comments
                r'#.*called.*before',  # Order dependencies in comments
                r'#.*depends.*on.*test',  # Explicit dependencies in comments
            ],
            'test_data_sharing': [
                r'self\.\w+.*=.*test_\w+',  # Storing test results in instance vars
                r'self\.\w+\s*=\s*["\'].*["\']',  # Storing data in instance variables
                r'class.*:\s*\n\s*\w+\s*=.*\[\]',  # Shared class-level collections
                r'class.*:\s*\n\s*\w+\s*=.*\{\}',  # Shared class-level dicts
                r'assert\s+hasattr\(self,\s*["\'].*["\']',  # Checking for attributes set by other tests
            ],
            'file_system_deps': [
                r'open\(["\'][^"\']*["\'].*["\']w["\']',  # File writing without cleanup
                r'mkdir\(|makedirs\(',  # Directory creation
                r'remove\(|rmdir\(',  # File/dir removal (potential cleanup)
            ],
            'database_deps': [
                r'\.commit\(\)',  # Database commits
                r'\.save\(\)',  # ORM saves
                r'INSERT\s+INTO|UPDATE\s+|DELETE\s+FROM',  # SQL operations
            ]
        }
        
        self.isolation_patterns = {
            'proper_cleanup': [
                r'tearDown\(|teardown\(',  # Cleanup methods
                r'@.*fixture.*scope',  # Scoped fixtures
                r'yield.*\n.*finally:|yield.*\n.*except:',  # Fixture cleanup
                r'with\s+.*:',  # Context managers
            ],
            'mocking_isolation': [
                r'@patch|@mock\.patch',  # Proper mocking
                r'with\s+patch\(',  # Context manager mocking
                r'Mock\(\)|MagicMock\(\)',  # Mock objects
            ],
            'temporary_resources': [
                r'tempfile\.|NamedTemporaryFile|TemporaryDirectory',  # Temp files
                r'mkdtemp\(|mkstemp\(',  # Temp directory/file creation
            ]
        }
    
    def get_dimension(self) -> QualityDimension:
        return QualityDimension.INDEPENDENCE
    
    def analyze(self, test_code: str, source_code: str = "") -> QualityScore:
        """Analyze test independence and isolation."""
        score = 100.0  # Start with perfect score
        details = {}
        suggestions = []
        violations = []
        
        # Analyze shared state issues
        shared_state_score, shared_violations = self._analyze_shared_state(test_code)
        score -= (100 - shared_state_score) * 0.4  # 40% weight
        violations.extend(shared_violations)
        details['shared_state_score'] = shared_state_score
        
        # Analyze execution order dependencies
        order_score, order_violations = self._analyze_execution_order(test_code)
        score -= (100 - order_score) * 0.3  # 30% weight
        violations.extend(order_violations)
        details['execution_order_score'] = order_score
        
        # Analyze proper isolation and cleanup
        isolation_score, isolation_violations = self._analyze_isolation(test_code)
        score -= (100 - isolation_score) * 0.3  # 30% weight
        violations.extend(isolation_violations)
        details['isolation_score'] = isolation_score
        
        # Generate suggestions based on violations
        suggestions = self._generate_suggestions(violations)
        
        # Additional analysis
        test_functions = re.findall(r'def\s+test_\w+', test_code)
        details['total_test_functions'] = len(test_functions)
        details['violations_found'] = len(violations)
        details['violation_types'] = list(set(v['type'] for v in violations))
        
        # Ensure score doesn't go below 0
        final_score = max(score, 0.0)
        
        return QualityScore(
            dimension=self.get_dimension(),
            score=final_score,
            details=details,
            suggestions=suggestions
        )
    
    def _analyze_shared_state(self, test_code: str) -> Tuple[float, List[Dict[str, Any]]]:
        """Analyze shared state violations."""
        violations = []
        score = 100.0
        
        for category, patterns in self.shared_state_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, test_code, re.MULTILINE | re.IGNORECASE))
                for match in matches:
                    line_num = test_code[:match.start()].count('\n') + 1
                    violations.append({
                        'type': f'shared_state_{category}',
                        'line': line_num,
                        'code': match.group(0).strip(),
                        'severity': self._get_severity(category)
                    })
                    
                    # Deduct points based on severity
                    if category == 'global_variables':
                        score -= 20  # High penalty for globals
                    elif category == 'class_variables':
                        score -= 15  # Medium penalty for class vars
                    elif category == 'module_state':
                        score -= 25  # Very high penalty for module state
                    else:
                        score -= 10  # Lower penalty for other patterns
        
        return max(score, 0.0), violations
    
    def _analyze_execution_order(self, test_code: str) -> Tuple[float, List[Dict[str, Any]]]:
        """Analyze execution order dependencies."""
        violations = []
        score = 100.0
        
        for category, patterns in self.dependency_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, test_code, re.MULTILINE | re.IGNORECASE))
                for match in matches:
                    line_num = test_code[:match.start()].count('\n') + 1
                    violations.append({
                        'type': f'order_dependency_{category}',
                        'line': line_num,
                        'code': match.group(0).strip(),
                        'severity': self._get_severity(category)
                    })
                    
                    # Deduct points based on category
                    if category == 'execution_order':
                        score -= 30  # Very high penalty for order deps
                    elif category == 'test_data_sharing':
                        score -= 20  # High penalty for data sharing
                    elif category == 'database_deps':
                        score -= 25  # High penalty for DB deps
                    else:
                        score -= 15  # Medium penalty for file system deps
        
        return max(score, 0.0), violations
    
    def _analyze_isolation(self, test_code: str) -> Tuple[float, List[Dict[str, Any]]]:
        """Analyze proper isolation and cleanup."""
        score = 100.0
        violations = []
        isolation_found = {}
        
        # Check for proper isolation patterns
        for category, patterns in self.isolation_patterns.items():
            found_count = 0
            for pattern in patterns:
                matches = re.findall(pattern, test_code, re.MULTILINE | re.IGNORECASE)
                found_count += len(matches)
            isolation_found[category] = found_count
        
        # Analyze test classes for proper setup/teardown
        test_classes = re.findall(r'class\s+(Test\w+).*?(?=class|\Z)', test_code, re.DOTALL)
        for i, test_class in enumerate(test_classes):
            # Check for setUp/tearDown methods
            has_setup = bool(re.search(r'def\s+setUp\(', test_class))
            has_teardown = bool(re.search(r'def\s+tearDown\(', test_class))
            
            if has_setup and not has_teardown:
                violations.append({
                    'type': 'missing_teardown',
                    'line': test_code.find(test_class) // len(test_code.split('\n')) + 1,
                    'code': f'Test class {i+1} has setUp but no tearDown',
                    'severity': 'medium'
                })
                score -= 15
        
        # Check for resource cleanup patterns
        resource_patterns = [
            r'(?<!with\s)open\(["\'][^"\']+["\'].*["\']w["\']',  # Files opened for writing without context manager
            r'(?<!with\s)connect\([^)]+\)',  # Connections without context manager
        ]
        
        for pattern in resource_patterns:
            matches = list(re.finditer(pattern, test_code, re.MULTILINE))
            for match in matches:
                line_num = test_code[:match.start()].count('\n') + 1
                violations.append({
                    'type': 'resource_leak_risk',
                    'line': line_num,
                    'code': match.group(0).strip(),
                    'severity': 'medium'
                })
                score -= 10
        
        # Bonus for good isolation practices
        if isolation_found['proper_cleanup'] > 0:
            score += min(isolation_found['proper_cleanup'] * 2, 10)
        if isolation_found['mocking_isolation'] > 0:
            score += min(isolation_found['mocking_isolation'] * 2, 10)
        if isolation_found['temporary_resources'] > 0:
            score += min(isolation_found['temporary_resources'] * 2, 10)
        
        return min(max(score, 0.0), 100.0), violations
    
    def _get_severity(self, category: str) -> str:
        """Get severity level for a violation category."""
        high_severity = ['global_variables', 'module_state', 'execution_order', 'database_deps']
        medium_severity = ['class_variables', 'test_data_sharing', 'file_system_deps']
        
        if category in high_severity:
            return 'high'
        elif category in medium_severity:
            return 'medium'
        else:
            return 'low'
    
    def _generate_suggestions(self, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate improvement suggestions based on violations."""
        suggestions = []
        violation_types = set(v['type'] for v in violations)
        
        # Suggestions for specific violation types
        suggestion_map = {
            'shared_state_global_variables': "Avoid global variables in tests. Use fixtures or dependency injection instead.",
            'shared_state_class_variables': "Avoid modifying class variables in tests. Use instance variables or mocks.",
            'shared_state_module_state': "Avoid modifying module state. Use mocks or temporary environments.",
            'order_dependency_execution_order': "Tests should not depend on execution order. Make each test independent.",
            'order_dependency_test_data_sharing': "Avoid sharing data between tests. Use fresh data in each test.",
            'order_dependency_database_deps': "Use database transactions or separate test databases to isolate tests.",
            'order_dependency_file_system_deps': "Use temporary files/directories that are cleaned up after each test.",
            'missing_teardown': "Add tearDown methods to clean up after setUp methods.",
            'resource_leak_risk': "Use context managers (with statements) for resource management."
        }
        
        # Add specific suggestions for found violations
        for violation_type in violation_types:
            if violation_type in suggestion_map:
                suggestions.append(suggestion_map[violation_type])
        
        # Add general suggestions based on violation patterns
        if any('shared_state' in vt for vt in violation_types):
            suggestions.append("Consider using pytest fixtures with appropriate scopes for test data.")
        
        if any('order_dependency' in vt for vt in violation_types):
            suggestions.append("Each test should set up its own required state and clean up afterwards.")
        
        if len(violations) > 5:
            suggestions.append("Consider refactoring tests to improve isolation and reduce dependencies.")
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(suggestions))

 
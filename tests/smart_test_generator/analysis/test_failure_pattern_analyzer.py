"""Tests for FailurePatternAnalyzer functionality."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from smart_test_generator.analysis.failure_pattern_analyzer import (
    FailurePatternAnalyzer,
    FailurePattern,
    FixSuggestion,
    FailureAnalysis,
    FailureHistory,
    FailureCategory,
    create_failure_analyzer
)
from smart_test_generator.analysis.coverage.failure_parser import (
    FailureRecord,
    ParsedFailures
)


@pytest.fixture
def temp_project_root():
    """Create temporary project root directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_failures():
    """Create sample failure records for testing."""
    return ParsedFailures(
        total=3,
        failures=[
            FailureRecord(
                nodeid="test_module.py::test_assert_failure",
                file="test_module.py",
                line=10,
                message="AssertionError: Expected 5, got 3",
                assertion_diff="assert 5 == 3",
                captured_stdout=None,
                captured_stderr=None,
                duration=0.1
            ),
            FailureRecord(
                nodeid="test_imports.py::test_import_error",
                file="test_imports.py", 
                line=5,
                message="ImportError: No module named 'missing_module'",
                assertion_diff=None,
                captured_stdout=None,
                captured_stderr=None,
                duration=0.05
            ),
            FailureRecord(
                nodeid="test_attributes.py::test_none_error",
                file="test_attributes.py",
                line=15,
                message="AttributeError: 'NoneType' object has no attribute 'method'",
                assertion_diff=None,
                captured_stdout=None,
                captured_stderr=None,
                duration=0.02
            )
        ]
    )


@pytest.fixture
def analyzer(temp_project_root):
    """Create FailurePatternAnalyzer instance."""
    return FailurePatternAnalyzer(temp_project_root)


class TestFailurePatternAnalyzer:
    """Test cases for FailurePatternAnalyzer."""
    
    def test_initialization(self, temp_project_root):
        """Test analyzer initialization."""
        analyzer = FailurePatternAnalyzer(temp_project_root)
        
        assert analyzer.project_root == temp_project_root
        assert analyzer.history_file == temp_project_root / ".smart_test_generator" / "failure_history.json"
        assert len(analyzer.failure_patterns) > 0
        assert len(analyzer.fix_suggestions_db) > 0
        assert isinstance(analyzer.history, FailureHistory)
    
    def test_factory_function(self, temp_project_root):
        """Test create_failure_analyzer factory function."""
        analyzer = create_failure_analyzer(temp_project_root)
        assert isinstance(analyzer, FailurePatternAnalyzer)
        assert analyzer.project_root == temp_project_root
    
    def test_categorize_assertion_error(self, analyzer):
        """Test categorization of assertion errors."""
        failure = FailureRecord(
            nodeid="test.py::test_func",
            file="test.py",
            line=10,
            message="AssertionError: Expected 5, got 3",
            assertion_diff="assert 5 == 3",
            captured_stdout=None,
            captured_stderr=None,
            duration=0.1
        )
        
        category, confidence = analyzer._categorize_failure(failure)
        assert category == FailureCategory.ASSERTION_ERROR
        assert confidence > 0.3
    
    def test_categorize_import_error(self, analyzer):
        """Test categorization of import errors."""
        failure = FailureRecord(
            nodeid="test.py::test_func",
            file="test.py",
            line=5,
            message="ImportError: No module named 'missing'",
            assertion_diff=None,
            captured_stdout=None,
            captured_stderr=None,
            duration=0.05
        )
        
        category, confidence = analyzer._categorize_failure(failure)
        assert category == FailureCategory.IMPORT_ERROR
        assert confidence > 0.3
    
    def test_categorize_attribute_error(self, analyzer):
        """Test categorization of attribute errors."""
        failure = FailureRecord(
            nodeid="test.py::test_func",
            file="test.py",
            line=15,
            message="AttributeError: 'NoneType' object has no attribute 'method'",
            assertion_diff=None,
            captured_stdout=None,
            captured_stderr=None,
            duration=0.02
        )
        
        category, confidence = analyzer._categorize_failure(failure)
        assert category == FailureCategory.ATTRIBUTE_ERROR
        assert confidence > 0.3
    
    def test_categorize_unknown_error(self, analyzer):
        """Test categorization of unknown errors."""
        failure = FailureRecord(
            nodeid="test.py::test_func",
            file="test.py",
            line=20,
            message="SomeCustomError: This is a custom error message",
            assertion_diff=None,
            captured_stdout=None,
            captured_stderr=None,
            duration=0.01
        )
        
        category, confidence = analyzer._categorize_failure(failure)
        assert category == FailureCategory.UNKNOWN
        assert confidence == 0.0
    
    def test_analyze_failures_basic(self, analyzer, sample_failures):
        """Test basic failure analysis."""
        analysis = analyzer.analyze_failures(sample_failures)
        
        assert isinstance(analysis, FailureAnalysis)
        assert analysis.total_failures == 3
        assert len(analysis.categorized_failures) > 0
        assert len(analysis.pattern_frequencies) > 0
        assert len(analysis.fix_suggestions) > 0
        assert len(analysis.confidence_scores) == 3
    
    def test_analyze_failures_categories(self, analyzer, sample_failures):
        """Test failure analysis produces correct categories."""
        analysis = analyzer.analyze_failures(sample_failures)
        
        # Should have assertion, import, and attribute errors
        categories = analysis.categorized_failures.keys()
        assert FailureCategory.ASSERTION_ERROR in categories
        assert FailureCategory.IMPORT_ERROR in categories
        assert FailureCategory.ATTRIBUTE_ERROR in categories
    
    def test_analyze_failures_fix_suggestions(self, analyzer, sample_failures):
        """Test that fix suggestions are generated."""
        analysis = analyzer.analyze_failures(sample_failures)
        
        assert len(analysis.fix_suggestions) > 0
        
        for suggestion in analysis.fix_suggestions:
            assert isinstance(suggestion, FixSuggestion)
            assert suggestion.title
            assert suggestion.description
            assert suggestion.category in FailureCategory
            assert isinstance(suggestion.priority, int)
            assert isinstance(suggestion.automated, bool)
    
    def test_customize_suggestion(self, analyzer):
        """Test suggestion customization based on failures."""
        base_suggestion = FixSuggestion(
            FailureCategory.ASSERTION_ERROR,
            "Review assertion logic",
            "Check assertion conditions",
            priority=1
        )
        
        failures = [
            FailureRecord(
                nodeid="test_common.py::test_func1",
                file="test_common.py",
                line=10,
                message="AssertionError: Failed",
                assertion_diff=None,
                captured_stdout=None,
                captured_stderr=None,
                duration=0.1
            ),
            FailureRecord(
                nodeid="test_common.py::test_func2",
                file="test_common.py",
                line=20,
                message="AssertionError: Failed again",
                assertion_diff=None,
                captured_stdout=None,
                captured_stderr=None,
                duration=0.1
            )
        ]
        
        customized = analyzer._customize_suggestion(base_suggestion, failures)
        
        assert customized is not None
        assert "test_common.py" in customized.description
        assert customized.title == base_suggestion.title
        assert customized.category == base_suggestion.category
    
    def test_extract_function_names(self, analyzer):
        """Test extraction of function names from failure nodeids."""
        failures = [
            FailureRecord(
                nodeid="test_file.py::TestClass::test_method",
                file="test_file.py",
                line=10,
                message="Error",
                assertion_diff=None,
                captured_stdout=None,
                captured_stderr=None,
                duration=0.1
            ),
            FailureRecord(
                nodeid="test_file.py::test_function",
                file="test_file.py",
                line=20,
                message="Error",
                assertion_diff=None,
                captured_stdout=None,
                captured_stderr=None,
                duration=0.1
            )
        ]
        
        functions = analyzer._extract_function_names(failures)
        
        assert "test_method" in functions
        assert "test_function" in functions
        assert functions["test_method"] == 1
        assert functions["test_function"] == 1
    
    def test_update_failure_history(self, analyzer):
        """Test failure history updating."""
        failure = FailureRecord(
            nodeid="test.py::test_func",
            file="test.py",
            line=10,
            message="Error",
            assertion_diff=None,
            captured_stdout=None,
            captured_stderr=None,
            duration=0.1
        )
        
        initial_count = analyzer.history.failure_counts[failure.nodeid]
        analyzer._update_failure_history(failure, FailureCategory.ASSERTION_ERROR)
        
        assert analyzer.history.failure_counts[failure.nodeid] == initial_count + 1
        assert FailureCategory.ASSERTION_ERROR in analyzer.history.category_trends
        assert failure.nodeid in analyzer.history.last_seen
    
    def test_mark_resolution_success(self, analyzer):
        """Test marking resolution success."""
        category = FailureCategory.ASSERTION_ERROR
        
        # Mark success
        analyzer.mark_resolution_success("test.py::test_func", category, True)
        assert category in analyzer.history.resolution_success
        
        # Mark failure
        initial_rate = analyzer.history.resolution_success[category]
        analyzer.mark_resolution_success("test.py::test_func", category, False)
        
        # Rate should decrease
        assert analyzer.history.resolution_success[category] < initial_rate
    
    def test_get_failure_trends(self, analyzer):
        """Test getting failure trends."""
        # Add some trend data
        analyzer.history.category_trends[FailureCategory.ASSERTION_ERROR] = [1, 2, 3, 4, 5]
        analyzer.history.category_trends[FailureCategory.IMPORT_ERROR] = [2, 1, 3]
        
        trends = analyzer.get_failure_trends()
        
        assert FailureCategory.ASSERTION_ERROR in trends
        assert FailureCategory.IMPORT_ERROR in trends
        assert trends[FailureCategory.ASSERTION_ERROR] == [1, 2, 3, 4, 5]
    
    def test_get_success_rates(self, analyzer):
        """Test getting success rates."""
        analyzer.history.resolution_success[FailureCategory.ASSERTION_ERROR] = 0.8
        analyzer.history.resolution_success[FailureCategory.IMPORT_ERROR] = 0.6
        
        rates = analyzer.get_success_rates()
        
        assert rates[FailureCategory.ASSERTION_ERROR] == 0.8
        assert rates[FailureCategory.IMPORT_ERROR] == 0.6


class TestFailureHistory:
    """Test cases for FailureHistory management."""
    
    def test_save_and_load_history(self, temp_project_root):
        """Test saving and loading failure history."""
        analyzer = FailurePatternAnalyzer(temp_project_root)
        
        # Add some history data
        analyzer.history.failure_counts["test.py::test_func"] = 5
        analyzer.history.category_trends[FailureCategory.ASSERTION_ERROR] = [1, 2, 3]
        analyzer.history.resolution_success[FailureCategory.IMPORT_ERROR] = 0.7
        analyzer.history.last_seen["test.py::test_func"] = "1234567890"
        
        # Save history
        analyzer._save_history()
        
        # Create new analyzer and load
        new_analyzer = FailurePatternAnalyzer(temp_project_root)
        
        assert new_analyzer.history.failure_counts["test.py::test_func"] == 5
        assert FailureCategory.ASSERTION_ERROR in new_analyzer.history.category_trends
        assert new_analyzer.history.category_trends[FailureCategory.ASSERTION_ERROR] == [1, 2, 3]
        assert new_analyzer.history.resolution_success[FailureCategory.IMPORT_ERROR] == 0.7
        assert new_analyzer.history.last_seen["test.py::test_func"] == "1234567890"
    
    def test_load_history_file_not_exists(self, temp_project_root):
        """Test loading history when file doesn't exist."""
        analyzer = FailurePatternAnalyzer(temp_project_root)
        
        # Should create empty history
        assert len(analyzer.history.failure_counts) == 0
        assert len(analyzer.history.category_trends) == 0
        assert len(analyzer.history.resolution_success) == 0
    
    @patch("builtins.open", mock_open(read_data='{"invalid": "json"'))
    def test_load_history_invalid_json(self, temp_project_root):
        """Test loading history with invalid JSON."""
        # Create history file
        history_file = temp_project_root / ".smart_test_generator" / "failure_history.json"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        history_file.write_text('{"invalid": json}')  # Invalid JSON
        
        analyzer = FailurePatternAnalyzer(temp_project_root)
        
        # Should handle gracefully and create empty history
        assert isinstance(analyzer.history, FailureHistory)
    
    def test_save_history_permission_error(self, analyzer):
        """Test saving history with permission errors."""
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            # Should handle gracefully without raising
            analyzer._save_history()


class TestFailurePatterns:
    """Test cases for failure pattern definitions."""
    
    def test_pattern_definitions_exist(self, analyzer):
        """Test that pattern definitions are properly initialized."""
        patterns = analyzer.failure_patterns
        
        # Should have patterns for major categories
        assert FailureCategory.ASSERTION_ERROR in patterns
        assert FailureCategory.IMPORT_ERROR in patterns
        assert FailureCategory.ATTRIBUTE_ERROR in patterns
        assert FailureCategory.TYPE_ERROR in patterns
        assert FailureCategory.FIXTURE_ERROR in patterns
        
        # Each category should have at least one pattern
        for category, pattern_list in patterns.items():
            assert len(pattern_list) > 0
            for pattern in pattern_list:
                assert isinstance(pattern, FailurePattern)
                assert pattern.category == category
                assert pattern.pattern  # Non-empty pattern
                assert pattern.description  # Non-empty description
    
    def test_fix_suggestions_exist(self, analyzer):
        """Test that fix suggestions are properly initialized."""
        suggestions = analyzer.fix_suggestions_db
        
        # Should have suggestions for major categories
        assert FailureCategory.ASSERTION_ERROR in suggestions
        assert FailureCategory.IMPORT_ERROR in suggestions
        assert FailureCategory.ATTRIBUTE_ERROR in suggestions
        assert FailureCategory.FIXTURE_ERROR in suggestions
        
        # Each category should have at least one suggestion
        for category, suggestion_list in suggestions.items():
            assert len(suggestion_list) > 0
            for suggestion in suggestion_list:
                assert isinstance(suggestion, FixSuggestion)
                assert suggestion.category == category
                assert suggestion.title  # Non-empty title
                assert suggestion.description  # Non-empty description
                assert isinstance(suggestion.priority, int)
                assert suggestion.priority >= 1
    
    def test_pattern_matching(self, analyzer):
        """Test that patterns match expected error messages."""
        test_cases = [
            ("AssertionError: Expected 5, got 3", FailureCategory.ASSERTION_ERROR),
            ("ImportError: No module named 'test'", FailureCategory.IMPORT_ERROR),
            ("AttributeError: 'NoneType' object has no attribute 'method'", FailureCategory.ATTRIBUTE_ERROR),
            ("TypeError: unsupported operand type(s)", FailureCategory.TYPE_ERROR),
        ]
        
        for message, expected_category in test_cases:
            failure = FailureRecord(
                nodeid="test.py::test_func",
                file="test.py",
                line=10,
                message=message,
                assertion_diff=None,
                captured_stdout=None,
                captured_stderr=None,
                duration=0.1
            )
            
            category, confidence = analyzer._categorize_failure(failure)
            assert category == expected_category
            assert confidence > 0.0


class TestTrendingPatterns:
    """Test cases for trending pattern calculation."""
    
    def test_calculate_trending_patterns(self, analyzer, sample_failures):
        """Test trending pattern calculation."""
        # Add historical trend data
        analyzer.history.category_trends[FailureCategory.ASSERTION_ERROR] = [1, 1, 2, 3, 4]  # Increasing
        analyzer.history.category_trends[FailureCategory.IMPORT_ERROR] = [4, 3, 2, 1, 1]      # Decreasing
        
        analysis = analyzer.analyze_failures(sample_failures)
        trending = analysis.trending_patterns
        
        assert len(trending) > 0
        for category, trend_score in trending:
            assert isinstance(category, FailureCategory)
            assert isinstance(trend_score, float)
            assert trend_score >= 0
    
    def test_trending_new_patterns(self, analyzer):
        """Test trending calculation for new patterns."""
        # Analyze failures with no historical data
        failures = ParsedFailures(
            total=1,
            failures=[
                FailureRecord(
                    nodeid="test.py::test_func",
                    file="test.py",
                    line=10,
                    message="AssertionError: New failure type",
                    assertion_diff=None,
                    captured_stdout=None,
                    captured_stderr=None,
                    duration=0.1
                )
            ]
        )
        
        analysis = analyzer.analyze_failures(failures)
        
        # New patterns should have trend score of 1.0
        for category, trend_score in analysis.trending_patterns:
            if category == FailureCategory.ASSERTION_ERROR:
                assert trend_score == 1.0


class TestIntegration:
    """Integration tests for FailurePatternAnalyzer."""
    
    def test_full_analysis_workflow(self, analyzer, sample_failures):
        """Test complete analysis workflow."""
        # Run analysis
        analysis = analyzer.analyze_failures(sample_failures)
        
        # Verify comprehensive results
        assert analysis.total_failures == 3
        assert len(analysis.categorized_failures) >= 3
        assert len(analysis.pattern_frequencies) >= 3
        assert len(analysis.fix_suggestions) > 0
        assert len(analysis.confidence_scores) == 3
        assert len(analysis.trending_patterns) >= 0
        
        # Verify history was updated
        assert len(analyzer.history.failure_counts) >= 3
        assert len(analyzer.history.category_trends) >= 3
        
        # Mark some resolutions and verify
        for category in analysis.pattern_frequencies.keys():
            analyzer.mark_resolution_success("test_resolution", category, True)
            
        success_rates = analyzer.get_success_rates()
        assert len(success_rates) >= 3
    
    def test_repeated_analysis_consistency(self, analyzer, sample_failures):
        """Test that repeated analysis produces consistent results."""
        analysis1 = analyzer.analyze_failures(sample_failures)
        analysis2 = analyzer.analyze_failures(sample_failures)
        
        # Core results should be consistent
        assert analysis1.total_failures == analysis2.total_failures
        assert len(analysis1.categorized_failures) == len(analysis2.categorized_failures)
        
        # History should accumulate
        assert all(
            analyzer.history.failure_counts[nodeid] >= 2
            for nodeid in ["test_module.py::test_assert_failure", 
                          "test_imports.py::test_import_error",
                          "test_attributes.py::test_none_error"]
        )

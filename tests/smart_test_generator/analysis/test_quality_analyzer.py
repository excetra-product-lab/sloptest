import pytest
import re
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from smart_test_generator.analysis.quality_analyzer import (
    QualityAnalyzer,
    EdgeCaseAnalyzer,
    AssertionStrengthAnalyzer,
    MaintainabilityAnalyzer,
    BugDetectionAnalyzer,
    TestQualityEngine
)
from smart_test_generator.models.data_models import (
    QualityDimension,
    QualityScore,
    TestQualityReport
)


class TestQualityAnalyzer:
    """Test the abstract QualityAnalyzer base class."""
    
    def test_analyze_is_abstract_method(self):
        """Test that analyze method is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            QualityAnalyzer()
    
    def test_get_dimension_is_abstract_method(self):
        """Test that get_dimension method is abstract."""
        # Create a concrete subclass missing get_dimension
        class IncompleteAnalyzer(QualityAnalyzer):
            def analyze(self, test_code: str, source_code: str = "") -> QualityScore:
                return QualityScore(dimension=QualityDimension.EDGE_CASE_COVERAGE, score=0.0)
        
        with pytest.raises(TypeError):
            IncompleteAnalyzer()


class TestEdgeCaseAnalyzer:
    """Test the EdgeCaseAnalyzer class."""
    
    def test_init_sets_up_edge_case_patterns(self):
        """Test that __init__ properly initializes edge case patterns."""
        analyzer = EdgeCaseAnalyzer()
        
        assert hasattr(analyzer, 'edge_case_patterns')
        assert isinstance(analyzer.edge_case_patterns, dict)
        assert 'null_checks' in analyzer.edge_case_patterns
        assert 'empty_collections' in analyzer.edge_case_patterns
        assert 'boundary_values' in analyzer.edge_case_patterns
        assert 'type_validation' in analyzer.edge_case_patterns
        assert 'exception_handling' in analyzer.edge_case_patterns
        assert 'floating_point' in analyzer.edge_case_patterns
    
    def test_get_dimension_returns_edge_case_coverage(self):
        """Test that get_dimension returns EDGE_CASE_COVERAGE."""
        analyzer = EdgeCaseAnalyzer()
        
        result = analyzer.get_dimension()
        
        assert result == QualityDimension.EDGE_CASE_COVERAGE
    
    def test_analyze_with_comprehensive_edge_cases(self):
        """Test analyze method with code containing comprehensive edge cases."""
        analyzer = EdgeCaseAnalyzer()
        test_code = """
        def test_function():
            assert value is None
            assert len(items) == 0
            assert count == 0
            assert isinstance(obj, str)
            with pytest.raises(ValueError):
                func()
            assert math.isclose(result, 3.14)
        """
        
        result = analyzer.analyze(test_code)
        
        assert isinstance(result, QualityScore)
        assert result.dimension == QualityDimension.EDGE_CASE_COVERAGE
        assert result.score > 80  # Should score high with comprehensive coverage
        assert 'found_patterns' in result.details
        assert 'covered_categories' in result.details
        assert result.details['covered_categories'] >= 4
    
    def test_analyze_with_no_edge_cases(self):
        """Test analyze method with code containing no edge cases."""
        analyzer = EdgeCaseAnalyzer()
        test_code = """
        def test_function():
            result = add(2, 3)
            assert result == 5
        """
        
        result = analyzer.analyze(test_code)
        
        assert isinstance(result, QualityScore)
        assert result.score < 20  # Should score low with no edge cases
        assert result.details['covered_categories'] == 0
        assert len(result.suggestions) > 0
    
    def test_analyze_with_partial_edge_cases(self):
        """Test analyze method with code containing some edge cases."""
        analyzer = EdgeCaseAnalyzer()
        test_code = """
        def test_function():
            assert value is None
            assert len(items) == 0
        """
        
        result = analyzer.analyze(test_code)
        
        assert isinstance(result, QualityScore)
        assert 20 <= result.score <= 80  # Should score moderately
        assert result.details['covered_categories'] == 2
        assert len(result.suggestions) > 0
    
    def test_analyze_with_empty_code(self):
        """Test analyze method with empty test code."""
        analyzer = EdgeCaseAnalyzer()
        
        result = analyzer.analyze("")
        
        assert result.score == 0.0
        assert result.details['covered_categories'] == 0
        assert len(result.suggestions) > 0


class TestAssertionStrengthAnalyzer:
    """Test the AssertionStrengthAnalyzer class."""
    
    def test_init_sets_up_assertion_patterns(self):
        """Test that __init__ properly initializes assertion patterns."""
        analyzer = AssertionStrengthAnalyzer()
        
        assert hasattr(analyzer, 'weak_patterns')
        assert hasattr(analyzer, 'strong_patterns')
        assert isinstance(analyzer.weak_patterns, list)
        assert isinstance(analyzer.strong_patterns, list)
        assert len(analyzer.weak_patterns) > 0
        assert len(analyzer.strong_patterns) > 0
    
    def test_get_dimension_returns_assertion_strength(self):
        """Test that get_dimension returns ASSERTION_STRENGTH."""
        analyzer = AssertionStrengthAnalyzer()
        
        result = analyzer.get_dimension()
        
        assert result == QualityDimension.ASSERTION_STRENGTH
    
    def test_analyze_with_strong_assertions(self):
        """Test analyze method with strong assertions."""
        analyzer = AssertionStrengthAnalyzer()
        test_code = """
        def test_function():
            assert result == 42
            assert item in collection
            assertEqual(actual, expected)
            assertGreater(value, 10)
        """
        
        result = analyzer.analyze(test_code)
        
        assert isinstance(result, QualityScore)
        assert result.dimension == QualityDimension.ASSERTION_STRENGTH
        assert result.score > 70  # Should score high with strong assertions
        assert result.details['strong_assertions'] > 0
        assert result.details['total_assertions'] > 0
    
    def test_analyze_with_weak_assertions(self):
        """Test analyze method with weak assertions."""
        analyzer = AssertionStrengthAnalyzer()
        test_code = """
        def test_function():
            assert result
            assert value is not None
            assert len(items) > 0
        """
        
        result = analyzer.analyze(test_code)
        
        assert isinstance(result, QualityScore)
        assert result.score < 50  # Should score low with weak assertions
        assert result.details['weak_assertions'] > 0
        assert len(result.suggestions) > 0
    
    def test_analyze_with_no_assertions(self):
        """Test analyze method with no assertions."""
        analyzer = AssertionStrengthAnalyzer()
        test_code = """
        def test_function():
            result = add(2, 3)
            print(result)
        """
        
        result = analyzer.analyze(test_code)
        
        assert result.score == 0.0
        assert result.details['total_assertions'] == 0
        assert "Add assertions" in result.suggestions[0]
    
    def test_analyze_with_assertion_messages(self):
        """Test analyze method with assertion messages for bonus points."""
        analyzer = AssertionStrengthAnalyzer()
        test_code = """
        def test_function():
            assert result == 42, "Result should be 42"
            assert item in collection, "Item not found in collection"
        """
        
        result = analyzer.analyze(test_code)
        
        assert result.score > 80  # Should get bonus for messages
        assert result.details['assertion_messages'] == 2


class TestMaintainabilityAnalyzer:
    """Test the MaintainabilityAnalyzer class."""
    
    def test_get_dimension_returns_maintainability(self):
        """Test that get_dimension returns MAINTAINABILITY."""
        analyzer = MaintainabilityAnalyzer()
        
        result = analyzer.get_dimension()
        
        assert result == QualityDimension.MAINTAINABILITY
    
    def test_analyze_with_high_maintainability_code(self):
        """Test analyze method with highly maintainable code."""
        analyzer = MaintainabilityAnalyzer()
        test_code = """
        def test_user_registration_with_valid_email():
            \"\"\"Test that user registration works with valid email.\"\"\"
            # Arrange
            valid_email = "user@example.com"
            user_service = UserService()
            
            # Act
            result = user_service.register_user(valid_email)
            
            # Assert
            assert result.success is True
            assert result.user_id is not None
        """
        
        result = analyzer.analyze(test_code)
        
        assert isinstance(result, QualityScore)
        assert result.dimension == QualityDimension.MAINTAINABILITY
        assert result.score > 70  # Should score high
        assert result.details['has_docstrings'] is True
        assert result.details['follows_aaa_pattern'] is True
    
    def test_analyze_with_low_maintainability_code(self):
        """Test analyze method with poorly maintainable code."""
        analyzer = MaintainabilityAnalyzer()
        test_code = """
        def test_func():
            x = func(12345, 67890, 99999)
            y = x + 11111
            assert y == 22222
            z = another_func(x, y, 33333)
            assert z
        """
        
        result = analyzer.analyze(test_code)
        
        assert result.score < 50  # Should score low
        assert result.details['has_docstrings'] is False
        assert result.details['follows_aaa_pattern'] is False
        assert result.details['magic_numbers'] > 3
        assert len(result.suggestions) > 0
    
    def test_analyze_with_empty_code(self):
        """Test analyze method with empty code."""
        analyzer = MaintainabilityAnalyzer()
        
        result = analyzer.analyze("")
        
        assert result.score >= 0
        assert result.details['line_count'] == 0
    
    def test_analyze_detects_code_duplication(self):
        """Test that analyze method detects code duplication."""
        analyzer = MaintainabilityAnalyzer()
        test_code = """
        def test_one():
            result = calculate(10, 20)
            assert result == 30
        
        def test_two():
            result = calculate(10, 20)
            assert result == 30
        """
        
        result = analyzer.analyze(test_code)
        
        assert 'code_duplication_ratio' in result.details
        assert result.details['code_duplication_ratio'] > 0


class TestBugDetectionAnalyzer:
    """Test the BugDetectionAnalyzer class."""
    
    def test_get_dimension_returns_bug_detection_potential(self):
        """Test that get_dimension returns BUG_DETECTION_POTENTIAL."""
        analyzer = BugDetectionAnalyzer()
        
        result = analyzer.get_dimension()
        
        assert result == QualityDimension.BUG_DETECTION_POTENTIAL
    
    def test_analyze_with_comprehensive_bug_detection(self):
        """Test analyze method with comprehensive bug detection patterns."""
        analyzer = BugDetectionAnalyzer()
        test_code = """
        def test_function():
            with pytest.raises(ValueError):
                func(invalid_input)
            
            mock_service.assert_called_with(expected_arg)
            assert len(result) == 0
            assert value is None
            
            # Multiple assertions for different scenarios
            assert result.success is True
            assert result.error_code == 404
            assert result.message == "Not found"
        """
        
        result = analyzer.analyze(test_code)
        
        assert isinstance(result, QualityScore)
        assert result.dimension == QualityDimension.BUG_DETECTION_POTENTIAL
        assert result.score > 70  # Should score high
        assert result.details['tests_error_conditions'] is True
        assert result.details['verifies_state_changes'] is True
        assert result.details['tests_boundaries'] is True
    
    def test_analyze_with_minimal_bug_detection(self):
        """Test analyze method with minimal bug detection capability."""
        analyzer = BugDetectionAnalyzer()
        test_code = """
        def test_function():
            result = add(2, 3)
            assert result == 5
        """
        
        result = analyzer.analyze(test_code)
        
        assert result.score < 30  # Should score low
        assert result.details['tests_error_conditions'] is False
        assert result.details['verifies_state_changes'] is False
        assert len(result.suggestions) > 0
    
    def test_analyze_detects_integration_testing(self):
        """Test that analyze method detects integration testing patterns."""
        analyzer = BugDetectionAnalyzer()
        test_code = """
        @patch('service.database')
        def test_with_mock(mock_db):
            mock_db.get.return_value = test_data
            result = service.process_data()
            mock_db.save.assert_called_once()
        """
        
        result = analyzer.analyze(test_code)
        
        assert result.details['tests_interactions'] is True
        assert result.score > 40  # Should get points for integration testing


class TestTestQualityEngine:
    """Test the TestQualityEngine class."""
    
    def test_init_with_default_analyzers(self):
        """Test that __init__ sets up default analyzers when none provided."""
        engine = TestQualityEngine()
        
        assert len(engine.analyzers) == 4
        assert isinstance(engine.analyzers[0], EdgeCaseAnalyzer)
        assert isinstance(engine.analyzers[1], AssertionStrengthAnalyzer)
        assert isinstance(engine.analyzers[2], MaintainabilityAnalyzer)
        assert isinstance(engine.analyzers[3], BugDetectionAnalyzer)
        assert hasattr(engine, 'dimension_weights')
        assert isinstance(engine.dimension_weights, dict)
    
    def test_init_with_custom_analyzers(self):
        """Test that __init__ accepts custom analyzers."""
        custom_analyzers = [EdgeCaseAnalyzer(), MaintainabilityAnalyzer()]
        
        engine = TestQualityEngine(custom_analyzers=custom_analyzers)
        
        assert len(engine.analyzers) == 2
        assert engine.analyzers == custom_analyzers
    
    @patch('builtins.open', new_callable=mock_open, read_data="test code")
    @patch('pathlib.Path.exists')
    def test_analyze_test_quality_with_valid_files(self, mock_exists, mock_file):
        """Test analyze_test_quality with valid test and source files."""
        mock_exists.return_value = True
        engine = TestQualityEngine()
        
        result = engine.analyze_test_quality("test_file.py", "source_file.py")
        
        assert isinstance(result, TestQualityReport)
        assert result.test_file == "test_file.py"
        assert result.overall_score >= 0
        assert isinstance(result.dimension_scores, dict)
        mock_file.assert_called()
    
    @patch('builtins.open', side_effect=FileNotFoundError())
    def test_analyze_test_quality_with_missing_file(self, mock_file):
        """Test analyze_test_quality handles missing files gracefully."""
        engine = TestQualityEngine()
        
        result = engine.analyze_test_quality("missing_file.py")
        
        assert isinstance(result, TestQualityReport)
        assert result.overall_score == 0.0
        assert "Analysis failed" in result.improvement_suggestions[0]
    
    def test_analyze_test_code_quality_with_good_code(self):
        """Test analyze_test_code_quality with high-quality test code."""
        engine = TestQualityEngine()
        test_code = """
        def test_user_registration_with_valid_email():
            \"\"\"Test user registration with valid email address.\"\"\"
            # Arrange
            valid_email = "user@example.com"
            user_service = UserService()
            
            # Act
            with pytest.raises(ValidationError):
                user_service.register_user(None)
            
            result = user_service.register_user(valid_email)
            
            # Assert
            assert result.success is True
            assert result.user_id is not None
            assert isinstance(result.user_id, int)
        """
        
        result = engine.analyze_test_code_quality(test_code)
        
        assert isinstance(result, TestQualityReport)
        assert result.overall_score > 60  # Should score well
        assert len(result.dimension_scores) == 4
        assert all(isinstance(score, QualityScore) for score in result.dimension_scores.values())
    
    def test_analyze_test_code_quality_with_poor_code(self):
        """Test analyze_test_code_quality with poor-quality test code."""
        engine = TestQualityEngine()
        test_code = """
        def test_func():
            x = func()
            assert x
        """
        
        result = engine.analyze_test_code_quality(test_code)
        
        assert result.overall_score < 40  # Should score poorly
        assert len(result.improvement_suggestions) > 0
        assert len(result.priority_fixes) > 0
    
    def test_analyze_test_code_quality_handles_analyzer_failure(self):
        """Test that analyzer failures are handled gracefully."""
        # Create a mock analyzer that raises an exception
        mock_analyzer = Mock()
        mock_analyzer.analyze.side_effect = Exception("Test error")
        mock_analyzer.get_dimension.return_value = QualityDimension.EDGE_CASE_COVERAGE
        
        engine = TestQualityEngine(custom_analyzers=[mock_analyzer])
        
        result = engine.analyze_test_code_quality("test code")
        
        assert isinstance(result, TestQualityReport)
        assert QualityDimension.EDGE_CASE_COVERAGE in result.dimension_scores
        assert result.dimension_scores[QualityDimension.EDGE_CASE_COVERAGE].score == 0.0
    
    def test_analyze_test_code_quality_generates_priority_fixes(self):
        """Test that priority fixes are generated for low-scoring dimensions."""
        engine = TestQualityEngine()
        # Code that will score poorly on most dimensions
        test_code = "def test(): pass"
        
        result = engine.analyze_test_code_quality(test_code)
        
        assert len(result.priority_fixes) > 0
        # Priority fixes should come from low-scoring dimensions
        low_scoring_dimensions = [dim for dim, score in result.dimension_scores.items() if score.score < 50]
        assert len(low_scoring_dimensions) > 0
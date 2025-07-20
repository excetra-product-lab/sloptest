import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

from smart_test_generator.services.coverage_service import CoverageService
from smart_test_generator.models.data_models import TestCoverage
from smart_test_generator.utils.user_feedback import UserFeedback
from smart_test_generator.config import Config


class TestCoverageService:
    """Test suite for CoverageService class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config object."""
        config = Mock(spec=Config)
        config.get.return_value = []
        return config
    
    @pytest.fixture
    def mock_feedback(self):
        """Create a mock feedback object."""
        return Mock(spec=UserFeedback)
    
    @pytest.fixture
    def project_root(self, tmp_path):
        """Create a temporary project root directory."""
        return tmp_path
    
    @pytest.fixture
    def sample_coverage_data(self):
        """Create sample coverage data for testing."""
        return {
            "src/module1.py": TestCoverage(
                filepath="src/module1.py",
                line_coverage=85.0,
                branch_coverage=80.0,
                missing_lines=[10, 15, 20],
                covered_functions=["func1", "func2"],
                uncovered_functions=["func3"]
            ),
            "src/module2.py": TestCoverage(
                filepath="src/module2.py",
                line_coverage=75.0,
                branch_coverage=70.0,
                missing_lines=[5, 25],
                covered_functions=["func_a"],
                uncovered_functions=["func_b", "func_c"]
            )
        }
    
    def test_init_creates_coverage_service_with_required_dependencies(self, project_root, mock_config):
        """Test that CoverageService initializes with required dependencies."""
        # Arrange & Act
        with patch('smart_test_generator.services.coverage_service.CoverageAnalyzer') as mock_analyzer, \
             patch('smart_test_generator.services.coverage_service.TestGenerationTracker') as mock_tracker:
            
            service = CoverageService(project_root, mock_config)
            
            # Assert
            assert service.project_root == project_root
            assert service.config == mock_config
            assert service.feedback is None
            mock_analyzer.assert_called_once_with(project_root, mock_config)
            mock_tracker.assert_called_once()
    
    def test_init_creates_coverage_service_with_feedback(self, project_root, mock_config, mock_feedback):
        """Test that CoverageService initializes with optional feedback."""
        # Arrange & Act
        with patch('smart_test_generator.services.coverage_service.CoverageAnalyzer'), \
             patch('smart_test_generator.services.coverage_service.TestGenerationTracker'):
            
            service = CoverageService(project_root, mock_config, mock_feedback)
            
            # Assert
            assert service.feedback == mock_feedback
    
    def test_analyze_coverage_returns_coverage_data_for_valid_files(self, project_root, mock_config, sample_coverage_data):
        """Test that analyze_coverage returns coverage data for valid files."""
        # Arrange
        files = ["src/module1.py", "src/module2.py"]
        
        with patch('smart_test_generator.services.coverage_service.CoverageAnalyzer') as mock_analyzer_class, \
             patch('smart_test_generator.services.coverage_service.TestGenerationTracker') as mock_tracker_class:
            
            mock_analyzer = Mock()
            mock_analyzer.run_coverage_analysis.return_value = sample_coverage_data
            mock_analyzer_class.return_value = mock_analyzer
            
            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker
            
            service = CoverageService(project_root, mock_config)
            
            # Act
            result = service.analyze_coverage(files)
            
            # Assert
            assert result == sample_coverage_data
            mock_analyzer.run_coverage_analysis.assert_called_once_with(files)
            assert mock_tracker.update_coverage.call_count == 2
            mock_tracker.update_coverage.assert_any_call("src/module1.py", 85.0)
            mock_tracker.update_coverage.assert_any_call("src/module2.py", 75.0)
            mock_tracker.save_state.assert_called_once()
    
    def test_analyze_coverage_returns_empty_dict_when_no_coverage_data(self, project_root, mock_config):
        """Test that analyze_coverage returns empty dict when no coverage data available."""
        # Arrange
        files = ["src/module1.py"]
        
        with patch('smart_test_generator.services.coverage_service.CoverageAnalyzer') as mock_analyzer_class, \
             patch('smart_test_generator.services.coverage_service.TestGenerationTracker') as mock_tracker_class:
            
            mock_analyzer = Mock()
            mock_analyzer.run_coverage_analysis.return_value = {}
            mock_analyzer_class.return_value = mock_analyzer
            
            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker
            
            service = CoverageService(project_root, mock_config)
            
            # Act
            result = service.analyze_coverage(files)
            
            # Assert
            assert result == {}
            mock_tracker.update_coverage.assert_not_called()
            mock_tracker.save_state.assert_not_called()
    
    def test_analyze_coverage_raises_exception_when_analyzer_fails(self, project_root, mock_config):
        """Test that analyze_coverage raises exception when coverage analyzer fails."""
        # Arrange
        files = ["src/module1.py"]
        error_message = "Coverage analysis failed"
        
        with patch('smart_test_generator.services.coverage_service.CoverageAnalyzer') as mock_analyzer_class, \
             patch('smart_test_generator.services.coverage_service.TestGenerationTracker') as mock_tracker_class:
            
            mock_analyzer = Mock()
            mock_analyzer.run_coverage_analysis.side_effect = Exception(error_message)
            mock_analyzer_class.return_value = mock_analyzer
            
            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker
            
            service = CoverageService(project_root, mock_config)
            
            # Act & Assert
            with pytest.raises(Exception, match=error_message):
                service.analyze_coverage(files)
    
    def test_analyze_coverage_handles_empty_file_list(self, project_root, mock_config):
        """Test that analyze_coverage handles empty file list gracefully."""
        # Arrange
        files = []
        
        with patch('smart_test_generator.services.coverage_service.CoverageAnalyzer') as mock_analyzer_class, \
             patch('smart_test_generator.services.coverage_service.TestGenerationTracker') as mock_tracker_class:
            
            mock_analyzer = Mock()
            mock_analyzer.run_coverage_analysis.return_value = {}
            mock_analyzer_class.return_value = mock_analyzer
            
            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker
            
            service = CoverageService(project_root, mock_config)
            
            # Act
            result = service.analyze_coverage(files)
            
            # Assert
            assert result == {}
            mock_analyzer.run_coverage_analysis.assert_called_once_with(files)
    
    def test_generate_coverage_report_returns_no_data_message_for_empty_coverage(self, project_root, mock_config):
        """Test that generate_coverage_report returns appropriate message for empty coverage data."""
        # Arrange
        with patch('smart_test_generator.services.coverage_service.CoverageAnalyzer'), \
             patch('smart_test_generator.services.coverage_service.TestGenerationTracker'):
            
            service = CoverageService(project_root, mock_config)
            
            # Act
            result = service.generate_coverage_report({})
            
            # Assert
            assert result == "No coverage data available."
    
    def test_generate_coverage_report_creates_summary_for_single_file(self, project_root, mock_config):
        """Test that generate_coverage_report creates proper summary for single file."""
        # Arrange
        coverage_data = {
            "src/module1.py": TestCoverage(
                filepath="src/module1.py",
                line_coverage=85.0,
                branch_coverage=80.0,
                missing_lines=[10, 15],
                covered_functions=["func1"],
                uncovered_functions=["func2"]
            )
        }
        
        with patch('smart_test_generator.services.coverage_service.CoverageAnalyzer'), \
             patch('smart_test_generator.services.coverage_service.TestGenerationTracker'):
            
            service = CoverageService(project_root, mock_config)
            
            # Act
            result = service.generate_coverage_report(coverage_data)
            
            # Assert
            expected_lines = [
                "Overall coverage: 85.0%",
                "",
                "src/module1.py: 85.0%"
            ]
            assert result == "\n".join(expected_lines)
    
    def test_generate_coverage_report_calculates_average_for_multiple_files(self, project_root, mock_config, sample_coverage_data):
        """Test that generate_coverage_report calculates correct average for multiple files."""
        # Arrange
        with patch('smart_test_generator.services.coverage_service.CoverageAnalyzer'), \
             patch('smart_test_generator.services.coverage_service.TestGenerationTracker'):
            
            service = CoverageService(project_root, mock_config)
            
            # Act
            result = service.generate_coverage_report(sample_coverage_data)
            
            # Assert
            # Average of 85.0 and 75.0 should be 80.0
            assert "Overall coverage: 80.0%" in result
            assert "src/module1.py: 85.0%" in result
            assert "src/module2.py: 75.0%" in result
    
    def test_generate_coverage_report_handles_relative_paths_correctly(self, project_root, mock_config):
        """Test that generate_coverage_report handles relative paths correctly."""
        # Arrange
        absolute_path = str(project_root / "src" / "module1.py")
        coverage_data = {
            absolute_path: TestCoverage(
                filepath=absolute_path,
                line_coverage=90.0,
                branch_coverage=85.0,
                missing_lines=[],
                covered_functions=["func1"],
                uncovered_functions=[]
            )
        }
        
        with patch('smart_test_generator.services.coverage_service.CoverageAnalyzer'), \
             patch('smart_test_generator.services.coverage_service.TestGenerationTracker'):
            
            service = CoverageService(project_root, mock_config)
            
            # Act
            result = service.generate_coverage_report(coverage_data)
            
            # Assert
            assert "src/module1.py: 90.0%" in result
    
    def test_get_coverage_history_returns_copy_of_tracker_history(self, project_root, mock_config):
        """Test that get_coverage_history returns a copy of the tracker's coverage history."""
        # Arrange
        expected_history = {
            "src/module1.py": [80.0, 85.0, 90.0],
            "src/module2.py": [70.0, 75.0]
        }
        
        with patch('smart_test_generator.services.coverage_service.CoverageAnalyzer'), \
             patch('smart_test_generator.services.coverage_service.TestGenerationTracker') as mock_tracker_class:
            
            mock_tracker = Mock()
            mock_state = Mock()
            mock_state.coverage_history = expected_history
            mock_tracker.state = mock_state
            mock_tracker_class.return_value = mock_tracker
            
            service = CoverageService(project_root, mock_config)
            
            # Act
            result = service.get_coverage_history()
            
            # Assert
            assert result == expected_history
            # Verify it's a copy by checking the copy method was called
            mock_state.coverage_history.copy.assert_called_once()
    
    def test_get_coverage_history_returns_empty_dict_when_no_history(self, project_root, mock_config):
        """Test that get_coverage_history returns empty dict when no history exists."""
        # Arrange
        with patch('smart_test_generator.services.coverage_service.CoverageAnalyzer'), \
             patch('smart_test_generator.services.coverage_service.TestGenerationTracker') as mock_tracker_class:
            
            mock_tracker = Mock()
            mock_state = Mock()
            mock_state.coverage_history = {}
            mock_tracker.state = mock_state
            mock_tracker_class.return_value = mock_tracker
            
            service = CoverageService(project_root, mock_config)
            
            # Act
            result = service.get_coverage_history()
            
            # Assert
            assert result == {}
    
    def test_analyze_coverage_logs_info_message_on_start(self, project_root, mock_config, mock_feedback):
        """Test that analyze_coverage logs info message when starting analysis."""
        # Arrange
        files = ["src/module1.py"]
        
        with patch('smart_test_generator.services.coverage_service.CoverageAnalyzer') as mock_analyzer_class, \
             patch('smart_test_generator.services.coverage_service.TestGenerationTracker') as mock_tracker_class:
            
            mock_analyzer = Mock()
            mock_analyzer.run_coverage_analysis.return_value = {}
            mock_analyzer_class.return_value = mock_analyzer
            
            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker
            
            service = CoverageService(project_root, mock_config, mock_feedback)
            
            # Act
            service.analyze_coverage(files)
            
            # Assert
            mock_feedback.info.assert_called_with("Running coverage analysis...")
    
    def test_analyze_coverage_logs_warning_when_no_coverage_data(self, project_root, mock_config, mock_feedback):
        """Test that analyze_coverage logs warning when no coverage data is available."""
        # Arrange
        files = ["src/module1.py"]
        
        with patch('smart_test_generator.services.coverage_service.CoverageAnalyzer') as mock_analyzer_class, \
             patch('smart_test_generator.services.coverage_service.TestGenerationTracker') as mock_tracker_class:
            
            mock_analyzer = Mock()
            mock_analyzer.run_coverage_analysis.return_value = {}
            mock_analyzer_class.return_value = mock_analyzer
            
            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker
            
            service = CoverageService(project_root, mock_config, mock_feedback)
            
            # Act
            service.analyze_coverage(files)
            
            # Assert
            mock_feedback.warning.assert_called_with("No coverage data available. Make sure you have tests and pytest-cov installed.")
    
    def test_analyze_coverage_logs_error_when_exception_occurs(self, project_root, mock_config, mock_feedback):
        """Test that analyze_coverage logs error when exception occurs during analysis."""
        # Arrange
        files = ["src/module1.py"]
        error_message = "Analysis failed"
        
        with patch('smart_test_generator.services.coverage_service.CoverageAnalyzer') as mock_analyzer_class, \
             patch('smart_test_generator.services.coverage_service.TestGenerationTracker') as mock_tracker_class:
            
            mock_analyzer = Mock()
            mock_analyzer.run_coverage_analysis.side_effect = Exception(error_message)
            mock_analyzer_class.return_value = mock_analyzer
            
            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker
            
            service = CoverageService(project_root, mock_config, mock_feedback)
            
            # Act & Assert
            with pytest.raises(Exception):
                service.analyze_coverage(files)
            
            mock_feedback.error.assert_called_with(f"Coverage analysis failed: {error_message}")
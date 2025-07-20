import pytest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
from typing import Dict, List

from smart_test_generator.services.test_generation_service import TestGenerationService, TestFileResult
from smart_test_generator.models.data_models import TestGenerationPlan, TestCoverage, TestableElement
from smart_test_generator.config import Config
from smart_test_generator.utils.user_feedback import UserFeedback
from smart_test_generator.exceptions import TestGenerationError


class TestTestGenerationService:
    """Test suite for TestGenerationService."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock config object."""
        config = Mock(spec=Config)
        config.get.return_value = []
        return config
    
    @pytest.fixture
    def mock_feedback(self):
        """Create a mock feedback object."""
        feedback = Mock(spec=UserFeedback)
        feedback.console = Mock()
        feedback.console.print = Mock()
        return feedback
    
    @pytest.fixture
    def project_root(self, tmp_path):
        """Create a temporary project root."""
        return tmp_path
    
    @pytest.fixture
    def sample_test_plan(self):
        """Create a sample test generation plan."""
        element = TestableElement(
            name="test_function",
            type="function",
            line_number=10,
            complexity=2
        )
        coverage = TestCoverage(
            filepath="src/module.py",
            line_coverage=50.0,
            branch_coverage=40.0,
            missing_lines=[5, 10, 15],
            covered_functions={"test_function"},
            uncovered_functions={"another_function"}
        )
        return TestGenerationPlan(
            source_file="src/module.py",
            elements_to_test=[element],
            coverage_before=coverage,
            estimated_coverage_after=75.0
        )
    
    @patch('smart_test_generator.services.test_generation_service.TestGenerationReporter')
    @patch('smart_test_generator.services.test_generation_service.TestFileWriter')
    @patch('smart_test_generator.services.test_generation_service.TestGenerationTracker')
    def test_init_creates_required_components(self, mock_tracker_class, mock_writer_class, mock_reporter_class, project_root, mock_config, mock_feedback):
        """Test that __init__ creates all required components."""
        # Arrange
        mock_tracker = Mock()
        mock_writer = Mock()
        mock_reporter = Mock()
        mock_tracker_class.return_value = mock_tracker
        mock_writer_class.return_value = mock_writer
        mock_reporter_class.return_value = mock_reporter
        
        # Act
        service = TestGenerationService(project_root, mock_config, mock_feedback)
        
        # Assert
        assert service.project_root == project_root
        assert service.config == mock_config
        assert service.feedback == mock_feedback
        assert service.tracker == mock_tracker
        assert service.writer == mock_writer
        assert service.reporter == mock_reporter
        
        mock_tracker_class.assert_called_once()
        mock_writer_class.assert_called_once_with(str(project_root), mock_config)
        mock_reporter_class.assert_called_once_with(project_root)
    
    @patch('smart_test_generator.services.test_generation_service.TestGenerationReporter')
    @patch('smart_test_generator.services.test_generation_service.TestFileWriter')
    @patch('smart_test_generator.services.test_generation_service.TestGenerationTracker')
    def test_init_without_feedback_uses_none(self, mock_tracker_class, mock_writer_class, mock_reporter_class, project_root, mock_config):
        """Test that __init__ works without feedback parameter."""
        # Act
        service = TestGenerationService(project_root, mock_config)
        
        # Assert
        assert service.feedback is None
    
    @patch('smart_test_generator.services.test_generation_service.IncrementalLLMClient')
    @patch('smart_test_generator.services.test_generation_service.ProgressTracker')
    def test_generate_tests_with_empty_plans_returns_empty_dict(self, mock_progress_class, mock_incremental_class, project_root, mock_config, mock_feedback):
        """Test that generate_tests returns empty dict when no test plans provided."""
        # Arrange
        service = TestGenerationService(project_root, mock_config, mock_feedback)
        mock_llm_client = Mock()
        
        # Act
        result = service.generate_tests(mock_llm_client, [], "directory_structure")
        
        # Assert
        assert result == {}
        mock_incremental_class.assert_not_called()
    
    @patch('smart_test_generator.services.test_generation_service.IncrementalLLMClient')
    @patch('smart_test_generator.services.test_generation_service.ProgressTracker')
    def test_generate_tests_processes_single_batch_successfully(self, mock_progress_class, mock_incremental_class, project_root, mock_config, mock_feedback, sample_test_plan):
        """Test that generate_tests processes a single batch successfully."""
        # Arrange
        service = TestGenerationService(project_root, mock_config, mock_feedback)
        mock_llm_client = Mock()
        mock_incremental_client = Mock()
        mock_incremental_class.return_value = mock_incremental_client
        mock_incremental_client.generate_contextual_tests.return_value = {
            "src/module.py": "test content"
        }
        
        mock_progress = Mock()
        mock_progress_class.return_value = mock_progress
        
        # Mock the write method to succeed
        with patch.object(service, '_write_batch_immediately') as mock_write:
            mock_write.return_value = [TestFileResult("src/module.py", True)]
            
            # Act
            result = service.generate_tests(mock_llm_client, [sample_test_plan], "directory_structure")
        
        # Assert
        assert result == {"src/module.py": "Generated and written successfully"}
        mock_incremental_class.assert_called_once_with(mock_llm_client, mock_config)
        mock_incremental_client.generate_contextual_tests.assert_called_once()
        mock_write.assert_called_once()
    
    @patch('smart_test_generator.services.test_generation_service.IncrementalLLMClient')
    @patch('smart_test_generator.services.test_generation_service.ProgressTracker')
    def test_generate_tests_handles_batch_failure_gracefully(self, mock_progress_class, mock_incremental_class, project_root, mock_config, mock_feedback, sample_test_plan):
        """Test that generate_tests continues processing when a batch fails."""
        # Arrange
        service = TestGenerationService(project_root, mock_config, mock_feedback)
        mock_llm_client = Mock()
        mock_incremental_client = Mock()
        mock_incremental_class.return_value = mock_incremental_client
        
        # First batch fails, second succeeds
        sample_test_plan2 = TestGenerationPlan(
            source_file="src/module2.py",
            elements_to_test=[],
            coverage_before=None,
            estimated_coverage_after=60.0
        )
        
        mock_incremental_client.generate_contextual_tests.side_effect = [
            Exception("API Error"),
            {"src/module2.py": "test content"}
        ]
        
        mock_progress = Mock()
        mock_progress_class.return_value = mock_progress
        
        with patch.object(service, '_write_batch_immediately') as mock_write:
            mock_write.return_value = [TestFileResult("src/module2.py", True)]
            
            # Act
            result = service.generate_tests(mock_llm_client, [sample_test_plan, sample_test_plan2], "directory_structure", batch_size=1)
        
        # Assert
        assert result == {"src/module2.py": "Generated and written successfully"}
        assert mock_incremental_client.generate_contextual_tests.call_count == 2
    
    @patch('smart_test_generator.services.test_generation_service.IncrementalLLMClient')
    @patch('smart_test_generator.services.test_generation_service.ProgressTracker')
    def test_generate_tests_raises_error_when_no_files_written(self, mock_progress_class, mock_incremental_class, project_root, mock_config, mock_feedback, sample_test_plan):
        """Test that generate_tests raises TestGenerationError when no files are written."""
        # Arrange
        service = TestGenerationService(project_root, mock_config, mock_feedback)
        mock_llm_client = Mock()
        mock_incremental_client = Mock()
        mock_incremental_class.return_value = mock_incremental_client
        mock_incremental_client.generate_contextual_tests.side_effect = Exception("API Error")
        
        mock_progress = Mock()
        mock_progress_class.return_value = mock_progress
        
        # Act & Assert
        with pytest.raises(TestGenerationError) as exc_info:
            service.generate_tests(mock_llm_client, [sample_test_plan], "directory_structure")
        
        assert "Failed to generate any tests" in str(exc_info.value)
    
    @patch('smart_test_generator.services.test_generation_service.IncrementalLLMClient')
    @patch('smart_test_generator.services.test_generation_service.ProgressTracker')
    def test_generate_tests_streaming_with_empty_plans_returns_empty_dict(self, mock_progress_class, mock_incremental_class, project_root, mock_config, mock_feedback):
        """Test that generate_tests_streaming returns empty dict when no test plans provided."""
        # Arrange
        service = TestGenerationService(project_root, mock_config, mock_feedback)
        mock_llm_client = Mock()
        
        # Act
        result = service.generate_tests_streaming(mock_llm_client, [], "directory_structure")
        
        # Assert
        assert result == {}
        mock_incremental_class.assert_not_called()
    
    @patch('smart_test_generator.services.test_generation_service.IncrementalLLMClient')
    @patch('smart_test_generator.services.test_generation_service.ProgressTracker')
    def test_generate_tests_streaming_processes_files_individually(self, mock_progress_class, mock_incremental_class, project_root, mock_config, mock_feedback, sample_test_plan):
        """Test that generate_tests_streaming processes each file individually."""
        # Arrange
        service = TestGenerationService(project_root, mock_config, mock_feedback)
        mock_llm_client = Mock()
        mock_incremental_client = Mock()
        mock_incremental_class.return_value = mock_incremental_client
        mock_incremental_client.generate_single_file_test.return_value = "test content"
        
        mock_progress = Mock()
        mock_progress_class.return_value = mock_progress
        
        with patch.object(service, '_write_single_test_file') as mock_write:
            mock_write.return_value = TestFileResult("src/module.py", True)
            with patch.object(service, '_update_tracking_incremental') as mock_track:
                
                # Act
                result = service.generate_tests_streaming(mock_llm_client, [sample_test_plan], "directory_structure")
        
        # Assert
        assert result == {"src/module.py": "Generated and written successfully"}
        mock_incremental_client.generate_single_file_test.assert_called_once()
        mock_write.assert_called_once_with("src/module.py", "test content")
        mock_track.assert_called_once()
    
    @patch('smart_test_generator.services.test_generation_service.IncrementalLLMClient')
    @patch('smart_test_generator.services.test_generation_service.ProgressTracker')
    def test_generate_tests_streaming_handles_individual_file_failure(self, mock_progress_class, mock_incremental_class, project_root, mock_config, mock_feedback, sample_test_plan):
        """Test that generate_tests_streaming continues when individual files fail."""
        # Arrange
        service = TestGenerationService(project_root, mock_config, mock_feedback)
        mock_llm_client = Mock()
        mock_incremental_client = Mock()
        mock_incremental_class.return_value = mock_incremental_client
        
        # First file fails, second succeeds
        sample_test_plan2 = TestGenerationPlan(
            source_file="src/module2.py",
            elements_to_test=[],
            coverage_before=None,
            estimated_coverage_after=60.0
        )
        
        mock_incremental_client.generate_single_file_test.side_effect = [
            Exception("API Error"),
            "test content"
        ]
        
        mock_progress = Mock()
        mock_progress_class.return_value = mock_progress
        
        with patch.object(service, '_write_single_test_file') as mock_write:
            mock_write.return_value = TestFileResult("src/module2.py", True)
            with patch.object(service, '_update_tracking_incremental') as mock_track:
                
                # Act
                result = service.generate_tests_streaming(mock_llm_client, [sample_test_plan, sample_test_plan2], "directory_structure")
        
        # Assert
        assert result == {"src/module2.py": "Generated and written successfully"}
        assert mock_incremental_client.generate_single_file_test.call_count == 2
    
    @patch('smart_test_generator.services.test_generation_service.IncrementalLLMClient')
    @patch('smart_test_generator.services.test_generation_service.ProgressTracker')
    def test_generate_tests_streaming_handles_no_test_content_generated(self, mock_progress_class, mock_incremental_class, project_root, mock_config, mock_feedback, sample_test_plan):
        """Test that generate_tests_streaming handles when no test content is generated."""
        # Arrange
        service = TestGenerationService(project_root, mock_config, mock_feedback)
        mock_llm_client = Mock()
        mock_incremental_client = Mock()
        mock_incremental_class.return_value = mock_incremental_client
        mock_incremental_client.generate_single_file_test.return_value = None
        
        mock_progress = Mock()
        mock_progress_class.return_value = mock_progress
        
        # Act & Assert
        with pytest.raises(TestGenerationError) as exc_info:
            service.generate_tests_streaming(mock_llm_client, [sample_test_plan], "directory_structure")
        
        assert "Failed to generate any tests" in str(exc_info.value)
    
    def test_measure_coverage_improvement_calculates_correctly(self, project_root, mock_config, mock_feedback):
        """Test that measure_coverage_improvement calculates improvement correctly."""
        # Arrange
        service = TestGenerationService(project_root, mock_config, mock_feedback)
        
        files_processed = ["src/module1.py", "src/module2.py"]
        old_coverage_data = {
            "src/module1.py": TestCoverage(filepath="src/module1.py", line_coverage=50.0, branch_coverage=40.0, missing_lines=[10], covered_functions={"func1"}, uncovered_functions={"func2"}),
            "src/module2.py": TestCoverage(filepath="src/module2.py", line_coverage=60.0, branch_coverage=50.0, missing_lines=[12], covered_functions={"func3"}, uncovered_functions={"func4"})
        }
        
        mock_coverage_service = Mock()
        new_coverage_data = {
            "src/module1.py": TestCoverage(filepath="src/module1.py", line_coverage=75.0, branch_coverage=65.0, missing_lines=[15], covered_functions={"func1", "func2"}, uncovered_functions=set()),
            "src/module2.py": TestCoverage(filepath="src/module2.py", line_coverage=80.0, branch_coverage=70.0, missing_lines=[16], covered_functions={"func3", "func4"}, uncovered_functions=set())
        }
        mock_coverage_service.analyze_coverage.return_value = new_coverage_data
        
        # Act
        result = service.measure_coverage_improvement(files_processed, old_coverage_data, mock_coverage_service)
        
        # Assert
        expected_before = (50.0 + 60.0) / 2  # 55.0
        expected_after = (75.0 + 80.0) / 2   # 77.5
        expected_improvement = expected_after - expected_before  # 22.5
        
        assert result['before'] == expected_before
        assert result['after'] == expected_after
        assert result['improvement'] == expected_improvement
        mock_coverage_service.analyze_coverage.assert_called_once_with(files_processed)
    
    def test_measure_coverage_improvement_handles_missing_coverage_data(self, project_root, mock_config, mock_feedback):
        """Test that measure_coverage_improvement handles missing coverage data gracefully."""
        # Arrange
        service = TestGenerationService(project_root, mock_config, mock_feedback)
        
        files_processed = ["src/module1.py"]
        old_coverage_data = {}  # Missing data
        
        mock_coverage_service = Mock()
        mock_coverage_service.analyze_coverage.return_value = {}  # Missing new data too
        
        # Act
        result = service.measure_coverage_improvement(files_processed, old_coverage_data, mock_coverage_service)
        
        # Assert
        assert result['before'] == 0
        assert result['after'] == 0
        assert result['improvement'] == 0
    
    def test_measure_coverage_improvement_handles_empty_files_list(self, project_root, mock_config, mock_feedback):
        """Test that measure_coverage_improvement handles empty files list."""
        # Arrange
        service = TestGenerationService(project_root, mock_config, mock_feedback)
        
        files_processed = []
        old_coverage_data = {}
        
        mock_coverage_service = Mock()
        mock_coverage_service.analyze_coverage.return_value = {}
        
        # Act
        result = service.measure_coverage_improvement(files_processed, old_coverage_data, mock_coverage_service)
        
        # Assert
        assert result['before'] == 0
        assert result['after'] == 0
        assert result['improvement'] == 0
    
    def test_measure_coverage_improvement_handles_exception(self, project_root, mock_config, mock_feedback):
        """Test that measure_coverage_improvement handles exceptions gracefully."""
        # Arrange
        service = TestGenerationService(project_root, mock_config, mock_feedback)
        
        files_processed = ["src/module1.py"]
        old_coverage_data = {}
        
        mock_coverage_service = Mock()
        mock_coverage_service.analyze_coverage.side_effect = Exception("Coverage analysis failed")
        
        # Act
        result = service.measure_coverage_improvement(files_processed, old_coverage_data, mock_coverage_service)
        
        # Assert
        assert result['before'] == 0
        assert result['after'] == 0
        assert result['improvement'] == 0
    
    def test_generate_final_report_creates_correct_results_structure(self, project_root, mock_config, mock_feedback):
        """Test that generate_final_report creates the correct results structure."""
        # Arrange
        service = TestGenerationService(project_root, mock_config, mock_feedback)
        service.reporter = Mock()
        service.reporter.generate_report.return_value = "Generated report"
        
        generated_tests = {
            "src/module1.py": "test content 1",
            "src/module2.py": "test content 2"
        }
        coverage_improvement = {
            'before': 50.0,
            'after': 75.0,
            'improvement': 25.0
        }
        
        # Act
        result = service.generate_final_report(generated_tests, coverage_improvement)
        
        # Assert
        expected_results = {
            "files_processed": 2,
            "tests_generated": 2,
            "coverage_before": 50.0,
            "coverage_after": 75.0,
            "details": ["src/module1.py", "src/module2.py"]
        }
        
        assert result == "Generated report"
        service.reporter.generate_report.assert_called_once_with(expected_results)
    
    def test_generate_final_report_handles_missing_coverage_data(self, project_root, mock_config, mock_feedback):
        """Test that generate_final_report handles missing coverage improvement data."""
        # Arrange
        service = TestGenerationService(project_root, mock_config, mock_feedback)
        service.reporter = Mock()
        service.reporter.generate_report.return_value = "Generated report"
        
        generated_tests = {"src/module1.py": "test content"}
        coverage_improvement = {}  # Missing data
        
        # Act
        result = service.generate_final_report(generated_tests, coverage_improvement)
        
        # Assert
        expected_results = {
            "files_processed": 1,
            "tests_generated": 1,
            "coverage_before": 0,
            "coverage_after": 0,
            "details": ["src/module1.py"]
        }
        
        assert result == "Generated report"
        service.reporter.generate_report.assert_called_once_with(expected_results)
    
    def test_generate_final_report_handles_empty_generated_tests(self, project_root, mock_config, mock_feedback):
        """Test that generate_final_report handles empty generated tests."""
        # Arrange
        service = TestGenerationService(project_root, mock_config, mock_feedback)
        service.reporter = Mock()
        service.reporter.generate_report.return_value = "Generated report"
        
        generated_tests = {}
        coverage_improvement = {'before': 0, 'after': 0, 'improvement': 0}
        
        # Act
        result = service.generate_final_report(generated_tests, coverage_improvement)
        
        # Assert
        expected_results = {
            "files_processed": 0,
            "tests_generated": 0,
            "coverage_before": 0,
            "coverage_after": 0,
            "details": []
        }
        
        assert result == "Generated report"
        service.reporter.generate_report.assert_called_once_with(expected_results)
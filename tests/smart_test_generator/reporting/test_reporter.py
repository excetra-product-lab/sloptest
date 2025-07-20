import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, mock_open

from smart_test_generator.reporting.reporter import TestGenerationReporter


class TestTestGenerationReporter:
    """Test TestGenerationReporter class."""

    def test_init_sets_project_root_and_report_file_path(self):
        """Test that __init__ correctly sets project_root and report_file attributes."""
        # Arrange
        project_root = Path("/test/project")
        
        # Act
        reporter = TestGenerationReporter(project_root)
        
        # Assert
        assert reporter.project_root == project_root
        assert reporter.report_file == project_root / ".testgen_report.json"

    def test_init_with_pathlib_path_object(self):
        """Test that __init__ works with Path objects."""
        # Arrange
        project_root = Path("/home/user/project")
        
        # Act
        reporter = TestGenerationReporter(project_root)
        
        # Assert
        assert isinstance(reporter.project_root, Path)
        assert isinstance(reporter.report_file, Path)
        assert str(reporter.report_file).endswith(".testgen_report.json")

    def test_init_with_string_path(self):
        """Test that __init__ works with string paths."""
        # Arrange
        project_root = "/test/string/path"
        
        # Act
        reporter = TestGenerationReporter(project_root)
        
        # Assert
        assert reporter.project_root == project_root
        assert reporter.report_file == Path(project_root) / ".testgen_report.json"

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('smart_test_generator.reporting.reporter.datetime')
    def test_generate_report_creates_complete_report_structure(self, mock_datetime, mock_json_dump, mock_file):
        """Test that generate_report creates a complete report with all expected fields."""
        # Arrange
        project_root = Path("/test/project")
        reporter = TestGenerationReporter(project_root)
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"
        
        results = {
            "files_processed": 5,
            "tests_generated": 15,
            "coverage_before": 60.5,
            "coverage_after": 85.2,
            "details": ["test1.py", "test2.py"]
        }
        
        # Act
        summary = reporter.generate_report(results)
        
        # Assert
        expected_report = {
            "timestamp": "2024-01-01T12:00:00",
            "summary": {
                "files_processed": 5,
                "tests_generated": 15,
                "coverage_before": 60.5,
                "coverage_after": 85.2,
                "coverage_improvement": 24.7
            },
            "details": ["test1.py", "test2.py"]
        }
        
        mock_file.assert_called_once_with(reporter.report_file, 'w')
        mock_json_dump.assert_called_once_with(expected_report, mock_file.return_value.__enter__.return_value, indent=2)
        assert "Test Generation Report" in summary
        assert "Files processed: 5" in summary
        assert "Tests generated: 15" in summary
        assert "Coverage before: 60.5%" in summary
        assert "Coverage after: 85.2%" in summary
        assert "Improvement: +24.7%" in summary

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('smart_test_generator.reporting.reporter.datetime')
    def test_generate_report_handles_missing_results_fields(self, mock_datetime, mock_json_dump, mock_file):
        """Test that generate_report handles missing fields in results with default values."""
        # Arrange
        project_root = Path("/test/project")
        reporter = TestGenerationReporter(project_root)
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"
        
        results = {}  # Empty results
        
        # Act
        summary = reporter.generate_report(results)
        
        # Assert
        expected_report = {
            "timestamp": "2024-01-01T12:00:00",
            "summary": {
                "files_processed": 0,
                "tests_generated": 0,
                "coverage_before": 0,
                "coverage_after": 0,
                "coverage_improvement": 0
            },
            "details": []
        }
        
        mock_json_dump.assert_called_once_with(expected_report, mock_file.return_value.__enter__.return_value, indent=2)
        assert "Files processed: 0" in summary
        assert "Tests generated: 0" in summary
        assert "Coverage before: 0.0%" in summary
        assert "Coverage after: 0.0%" in summary
        assert "Improvement: +0.0%" in summary

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('smart_test_generator.reporting.reporter.datetime')
    def test_generate_report_handles_partial_results(self, mock_datetime, mock_json_dump, mock_file):
        """Test that generate_report handles partially populated results."""
        # Arrange
        project_root = Path("/test/project")
        reporter = TestGenerationReporter(project_root)
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"
        
        results = {
            "files_processed": 3,
            "coverage_before": 45.0,
            # Missing tests_generated, coverage_after, details
        }
        
        # Act
        summary = reporter.generate_report(results)
        
        # Assert
        expected_report = {
            "timestamp": "2024-01-01T12:00:00",
            "summary": {
                "files_processed": 3,
                "tests_generated": 0,
                "coverage_before": 45.0,
                "coverage_after": 0,
                "coverage_improvement": -45.0
            },
            "details": []
        }
        
        mock_json_dump.assert_called_once_with(expected_report, mock_file.return_value.__enter__.return_value, indent=2)
        assert "Files processed: 3" in summary
        assert "Coverage before: 45.0%" in summary
        assert "Improvement: +-45.0%" in summary

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('smart_test_generator.reporting.reporter.datetime')
    def test_generate_report_calculates_coverage_improvement_correctly(self, mock_datetime, mock_json_dump, mock_file):
        """Test that generate_report correctly calculates coverage improvement."""
        # Arrange
        project_root = Path("/test/project")
        reporter = TestGenerationReporter(project_root)
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"
        
        results = {
            "coverage_before": 30.5,
            "coverage_after": 75.8
        }
        
        # Act
        reporter.generate_report(results)
        
        # Assert
        call_args = mock_json_dump.call_args[0][0]
        assert call_args["summary"]["coverage_improvement"] == 45.3

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('smart_test_generator.reporting.reporter.datetime')
    def test_generate_report_handles_negative_coverage_improvement(self, mock_datetime, mock_json_dump, mock_file):
        """Test that generate_report handles cases where coverage decreases."""
        # Arrange
        project_root = Path("/test/project")
        reporter = TestGenerationReporter(project_root)
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"
        
        results = {
            "coverage_before": 80.0,
            "coverage_after": 70.0
        }
        
        # Act
        summary = reporter.generate_report(results)
        
        # Assert
        call_args = mock_json_dump.call_args[0][0]
        assert call_args["summary"]["coverage_improvement"] == -10.0
        assert "Improvement: +-10.0%" in summary

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('smart_test_generator.reporting.reporter.datetime')
    def test_generate_report_includes_details_in_report(self, mock_datetime, mock_json_dump, mock_file):
        """Test that generate_report includes details from results in the report."""
        # Arrange
        project_root = Path("/test/project")
        reporter = TestGenerationReporter(project_root)
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"
        
        details = ["Generated test_module1.py", "Generated test_module2.py", "Skipped module3.py"]
        results = {"details": details}
        
        # Act
        reporter.generate_report(results)
        
        # Assert
        call_args = mock_json_dump.call_args[0][0]
        assert call_args["details"] == details

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('smart_test_generator.reporting.reporter.datetime')
    def test_generate_report_returns_formatted_summary_string(self, mock_datetime, mock_json_dump, mock_file):
        """Test that generate_report returns a properly formatted summary string."""
        # Arrange
        project_root = Path("/test/project")
        reporter = TestGenerationReporter(project_root)
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"
        
        results = {
            "files_processed": 10,
            "tests_generated": 25,
            "coverage_before": 40.0,
            "coverage_after": 80.0
        }
        
        # Act
        summary = reporter.generate_report(results)
        
        # Assert
        assert "Test Generation Report" in summary
        assert "======================" in summary
        assert "Generated: 2024-01-01T12:00:00" in summary
        assert "Summary:" in summary
        assert "- Files processed: 10" in summary
        assert "- Tests generated: 25" in summary
        assert "- Coverage before: 40.0%" in summary
        assert "- Coverage after: 80.0%" in summary
        assert "- Improvement: +40.0%" in summary
        assert f"Report saved to: {reporter.report_file}" in summary

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('smart_test_generator.reporting.reporter.datetime')
    def test_generate_report_uses_current_timestamp(self, mock_datetime, mock_json_dump, mock_file):
        """Test that generate_report uses the current timestamp."""
        # Arrange
        project_root = Path("/test/project")
        reporter = TestGenerationReporter(project_root)
        mock_datetime.now.return_value.isoformat.return_value = "2024-12-25T15:30:45"
        
        results = {}
        
        # Act
        summary = reporter.generate_report(results)
        
        # Assert
        mock_datetime.now.assert_called_once()
        call_args = mock_json_dump.call_args[0][0]
        assert call_args["timestamp"] == "2024-12-25T15:30:45"
        assert "Generated: 2024-12-25T15:30:45" in summary

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('smart_test_generator.reporting.reporter.datetime')
    def test_generate_report_saves_to_correct_file_path(self, mock_datetime, mock_json_dump, mock_file):
        """Test that generate_report saves the report to the correct file path."""
        # Arrange
        project_root = Path("/custom/project/path")
        reporter = TestGenerationReporter(project_root)
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"
        
        results = {}
        expected_file_path = project_root / ".testgen_report.json"
        
        # Act
        reporter.generate_report(results)
        
        # Assert
        mock_file.assert_called_once_with(expected_file_path, 'w')

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('smart_test_generator.reporting.reporter.datetime')
    def test_generate_report_formats_json_with_proper_indentation(self, mock_datetime, mock_json_dump, mock_file):
        """Test that generate_report formats JSON with proper indentation."""
        # Arrange
        project_root = Path("/test/project")
        reporter = TestGenerationReporter(project_root)
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"
        
        results = {}
        
        # Act
        reporter.generate_report(results)
        
        # Assert
        mock_json_dump.assert_called_once()
        call_args = mock_json_dump.call_args
        assert call_args[1]['indent'] == 2
import pytest
from unittest.mock import Mock, patch, MagicMock
import time
from pathlib import Path
from smart_test_generator.utils.ui_demo import demo_world_class_ui


class TestDemoBeautifulCli:
    """Test suite for the demo_world_class_ui function."""
    
    @patch('smart_test_generator.utils.ui_demo.UserFeedback')
    @patch('smart_test_generator.utils.ui_demo.ProgressTracker')
    @patch('smart_test_generator.utils.ui_demo.time.sleep')
    def test_demo_world_class_ui_executes_all_sections(self, mock_sleep, mock_progress_tracker, mock_user_feedback):
        """Test that demo_world_class_ui executes all demonstration sections without errors."""
        # Arrange
        mock_feedback = Mock()
        mock_user_feedback.return_value = mock_feedback
        mock_tracker = Mock()
        mock_progress_tracker.return_value = mock_tracker
        
        # Act
        demo_world_class_ui()
        
        # Assert - Verify UserFeedback was initialized with verbose=True
        mock_user_feedback.assert_called_once_with(verbose=True)
        
        # Verify section header was called
        mock_feedback.section_header.assert_called_with("CLI Interface Showcase")
        
        # Verify feature showcase was called with expected features
        expected_features = [
            "Professional status indicators",
            "Beautiful progress tracking", 
            "Rich table displays",
            "File tree visualization",
            "Elegant panels and summaries"
        ]
        mock_feedback.feature_showcase.assert_called_with(expected_features)
        
        # Verify divider calls
        assert mock_feedback.divider.call_count >= 5
        
        # Verify ProgressTracker was initialized
        mock_progress_tracker.assert_called_once_with(mock_feedback)
    
    @patch('smart_test_generator.utils.ui_demo.UserFeedback')
    @patch('smart_test_generator.utils.ui_demo.ProgressTracker')
    @patch('smart_test_generator.utils.ui_demo.time.sleep')
    def test_demo_world_class_ui_calls_status_messages(self, mock_sleep, mock_progress_tracker, mock_user_feedback):
        """Test that demo_world_class_ui demonstrates all status message types."""
        # Arrange
        mock_feedback = Mock()
        mock_user_feedback.return_value = mock_feedback
        mock_tracker = Mock()
        mock_progress_tracker.return_value = mock_tracker
        
        # Act
        demo_world_class_ui()
        
        # Assert - Verify all status message types are called
        mock_feedback.success.assert_any_call("This is a success message")
        mock_feedback.info.assert_any_call("This is an informational message")
        mock_feedback.warning.assert_any_call("This is a warning message", "Consider taking this action")
        mock_feedback.debug.assert_any_call("This is a debug message (only shown in verbose mode)")
        
        # Verify subsection header
        mock_feedback.subsection_header.assert_any_call("Status Messages")
    
    @patch('smart_test_generator.utils.ui_demo.UserFeedback')
    @patch('smart_test_generator.utils.ui_demo.ProgressTracker')
    @patch('smart_test_generator.utils.ui_demo.time.sleep')
    def test_demo_world_class_ui_displays_status_table(self, mock_sleep, mock_progress_tracker, mock_user_feedback):
        """Test that demo_world_class_ui displays status table with correct validation items."""
        # Arrange
        mock_feedback = Mock()
        mock_user_feedback.return_value = mock_feedback
        mock_tracker = Mock()
        mock_progress_tracker.return_value = mock_tracker
        
        # Act
        demo_world_class_ui()
        
        # Assert - Verify status table is called with expected items
        expected_validation_items = [
            ("success", "System Check", "All requirements met"),
            ("success", "Dependencies", "All packages available"),
            ("warning", "Configuration", "Using default settings"),
            ("running", "File Analysis", "Scanning project files"),
            ("pending", "Test Generation", "Waiting for analysis")
        ]
        mock_feedback.status_table.assert_called_with("Project Status", expected_validation_items)
    
    @patch('smart_test_generator.utils.ui_demo.UserFeedback')
    @patch('smart_test_generator.utils.ui_demo.ProgressTracker')
    @patch('smart_test_generator.utils.ui_demo.time.sleep')
    def test_demo_world_class_ui_shows_configuration_panel(self, mock_sleep, mock_progress_tracker, mock_user_feedback):
        """Test that demo_world_class_ui displays configuration panel with correct settings."""
        # Arrange
        mock_feedback = Mock()
        mock_user_feedback.return_value = mock_feedback
        mock_tracker = Mock()
        mock_progress_tracker.return_value = mock_tracker
        
        # Act
        demo_world_class_ui()
        
        # Assert - Verify summary panel is called with configuration info
        expected_config = {
            "Project Root": "/path/to/project",
            "Test Directory": "tests/",
            "Coverage Threshold": "80%",
            "Model": "claude-sonnet-4",
            "Batch Size": "10 files",
            "Force Regeneration": "No"
        }
        mock_feedback.summary_panel.assert_any_call("Current Configuration", expected_config, "blue")
    
    @patch('smart_test_generator.utils.ui_demo.UserFeedback')
    @patch('smart_test_generator.utils.ui_demo.ProgressTracker')
    @patch('smart_test_generator.utils.ui_demo.time.sleep')
    def test_demo_world_class_ui_demonstrates_progress_tracking(self, mock_sleep, mock_progress_tracker, mock_user_feedback):
        """Test that demo_world_class_ui properly demonstrates progress tracking functionality."""
        # Arrange
        mock_feedback = Mock()
        mock_user_feedback.return_value = mock_feedback
        mock_tracker = Mock()
        mock_progress_tracker.return_value = mock_tracker
        
        # Act
        demo_world_class_ui()
        
        # Assert - Verify progress tracker setup
        mock_tracker.set_total_steps.assert_called_once_with(6, "Setup Progress")
        
        # Verify step calls (should be called 6 times for each step)
        assert mock_tracker.step.call_count == 6
        
        # Verify specific step calls
        expected_steps = [
            "Initializing system",
            "Scanning Python files", 
            "Analyzing code structure",
            "Calculating coverage",
            "Planning test generation",
            "Finalizing setup"
        ]
        
        for step in expected_steps:
            mock_tracker.step.assert_any_call(step)
        
        # Verify completion
        mock_tracker.complete.assert_called_once_with("Setup completed successfully!")
        
        # Verify sleep was called for each step (6 times)
        assert mock_sleep.call_count >= 6
    
    @patch('smart_test_generator.utils.ui_demo.UserFeedback')
    @patch('smart_test_generator.utils.ui_demo.ProgressTracker')
    @patch('smart_test_generator.utils.ui_demo.time.sleep')
    def test_demo_world_class_ui_displays_file_tree(self, mock_sleep, mock_progress_tracker, mock_user_feedback):
        """Test that demo_world_class_ui displays file tree with expected demo files."""
        # Arrange
        mock_feedback = Mock()
        mock_user_feedback.return_value = mock_feedback
        mock_tracker = Mock()
        mock_progress_tracker.return_value = mock_tracker
        
        # Act
        demo_world_class_ui()
        
        # Assert - Verify file_tree was called
        mock_feedback.file_tree.assert_called_once()
        
        # Get the call arguments
        call_args = mock_feedback.file_tree.call_args
        title, files, base_path = call_args[0]
        
        # Verify title and base path
        assert title == "Project Structure"
        assert base_path == Path("src")
        
        # Verify expected files are included
        file_paths = [str(f) for f in files]
        expected_files = [
            "src/smart_test_generator/__init__.py",
            "src/smart_test_generator/cli.py",
            "src/smart_test_generator/config.py",
            "src/smart_test_generator/core/application.py",
            "src/smart_test_generator/core/llm_factory.py",
            "src/smart_test_generator/utils/user_feedback.py",
            "src/smart_test_generator/services/analysis_service.py",
            "tests/test_cli.py",
            "tests/test_config.py"
        ]
        
        for expected_file in expected_files:
            assert expected_file in file_paths
    
    @patch('smart_test_generator.utils.ui_demo.UserFeedback')
    @patch('smart_test_generator.utils.ui_demo.ProgressTracker')
    @patch('smart_test_generator.utils.ui_demo.time.sleep')
    def test_demo_world_class_ui_demonstrates_spinners(self, mock_sleep, mock_progress_tracker, mock_user_feedback):
        """Test that demo_world_class_ui demonstrates spinner functionality with different styles."""
        # Arrange
        mock_feedback = Mock()
        mock_context_manager = Mock()
        mock_feedback.status_spinner.return_value = mock_context_manager
        mock_user_feedback.return_value = mock_feedback
        mock_tracker = Mock()
        mock_progress_tracker.return_value = mock_tracker
        
        # Act
        demo_world_class_ui()
        
        # Assert - Verify spinner calls with different styles
        mock_feedback.status_spinner.assert_any_call("Processing large dataset", spinner_style="dots12")
        mock_feedback.status_spinner.assert_any_call("Generating AI responses", spinner_style="arc")
        
        # Verify context manager was entered/exited
        assert mock_context_manager.__enter__.call_count == 2
        assert mock_context_manager.__exit__.call_count == 2
        
        # Verify success messages after spinners
        mock_feedback.success.assert_any_call("Processing completed!")
        mock_feedback.success.assert_any_call("Generation completed!")
    
    @patch('smart_test_generator.utils.ui_demo.UserFeedback')
    @patch('smart_test_generator.utils.ui_demo.ProgressTracker')
    @patch('smart_test_generator.utils.ui_demo.time.sleep')
    def test_demo_world_class_ui_shows_operation_status(self, mock_sleep, mock_progress_tracker, mock_user_feedback):
        """Test that demo_world_class_ui displays operation status with correct operations."""
        # Arrange
        mock_feedback = Mock()
        mock_user_feedback.return_value = mock_feedback
        mock_tracker = Mock()
        mock_progress_tracker.return_value = mock_tracker
        
        # Act
        demo_world_class_ui()
        
        # Assert - Verify operation status is called with expected operations
        expected_operations = {
            "File Discovery": "completed",
            "Code Analysis": "completed", 
            "Coverage Calculation": "running",
            "Test Planning": "pending",
            "Quality Assessment": "pending"
        }
        mock_feedback.operation_status.assert_called_with(expected_operations)
    
    @patch('smart_test_generator.utils.ui_demo.UserFeedback')
    @patch('smart_test_generator.utils.ui_demo.ProgressTracker')
    @patch('smart_test_generator.utils.ui_demo.time.sleep')
    def test_demo_world_class_ui_displays_final_summary(self, mock_sleep, mock_progress_tracker, mock_user_feedback):
        """Test that demo_world_class_ui displays final summary with correct data."""
        # Arrange
        mock_feedback = Mock()
        mock_user_feedback.return_value = mock_feedback
        mock_tracker = Mock()
        mock_progress_tracker.return_value = mock_tracker
        
        # Act
        demo_world_class_ui()
        
        # Assert - Verify final summary panel
        expected_summary = {
            "Demo Mode": "Interactive Showcase",
            "Features Demonstrated": "8 categories",
            "Status": "All features working",
            "Performance": "Excellent"
        }
        mock_feedback.summary_panel.assert_any_call("Demo Summary", expected_summary, "green")
        
        # Verify final success and info messages
        mock_feedback.success.assert_any_call("CLI interface showcase completed!")
        mock_feedback.info.assert_any_call("Ready for production use with beautiful, professional interface")
    
    @patch('smart_test_generator.utils.ui_demo.UserFeedback')
    @patch('smart_test_generator.utils.ui_demo.ProgressTracker')
    @patch('smart_test_generator.utils.ui_demo.time.sleep')
    def test_demo_world_class_ui_handles_exceptions_gracefully(self, mock_sleep, mock_progress_tracker, mock_user_feedback):
        """Test that demo_world_class_ui handles exceptions from UserFeedback methods gracefully."""
        # Arrange
        mock_feedback = Mock()
        mock_feedback.section_header.side_effect = Exception("Test exception")
        mock_user_feedback.return_value = mock_feedback
        mock_tracker = Mock()
        mock_progress_tracker.return_value = mock_tracker
        
        # Act & Assert - Should raise the exception since function doesn't handle it
        with pytest.raises(Exception, match="Test exception"):
            demo_world_class_ui()
    
    @patch('smart_test_generator.utils.ui_demo.UserFeedback')
    @patch('smart_test_generator.utils.ui_demo.ProgressTracker')
    @patch('smart_test_generator.utils.ui_demo.time.sleep')
    def test_demo_world_class_ui_sleep_timing(self, mock_sleep, mock_progress_tracker, mock_user_feedback):
        """Test that demo_world_class_ui uses appropriate sleep timing for demonstrations."""
        # Arrange
        mock_feedback = Mock()
        mock_context_manager = Mock()
        mock_feedback.status_spinner.return_value = mock_context_manager
        mock_user_feedback.return_value = mock_feedback
        mock_tracker = Mock()
        mock_progress_tracker.return_value = mock_tracker
        
        # Act
        demo_world_class_ui()
        
        # Assert - Verify sleep calls with expected durations
        sleep_calls = mock_sleep.call_args_list
        
        # Should have sleep calls for progress steps (0.5s each) and spinners (2s, 1.5s)
        expected_sleep_values = [0.5] * 6 + [2, 1.5]  # 6 progress steps + 2 spinners
        actual_sleep_values = [call[0][0] for call in sleep_calls]
        
        # Verify we have the expected number of sleep calls
        assert len(actual_sleep_values) == len(expected_sleep_values)
        
        # Verify specific sleep durations are present
        assert 0.5 in actual_sleep_values  # Progress step timing
        assert 2 in actual_sleep_values    # First spinner timing
        assert 1.5 in actual_sleep_values  # Second spinner timing
    
    @patch('smart_test_generator.utils.ui_demo.UserFeedback')
    @patch('smart_test_generator.utils.ui_demo.ProgressTracker')
    @patch('smart_test_generator.utils.ui_demo.time.sleep')
    def test_demo_world_class_ui_divider_usage(self, mock_sleep, mock_progress_tracker, mock_user_feedback):
        """Test that demo_world_class_ui uses dividers appropriately between sections."""
        # Arrange
        mock_feedback = Mock()
        mock_user_feedback.return_value = mock_feedback
        mock_tracker = Mock()
        mock_progress_tracker.return_value = mock_tracker
        
        # Act
        demo_world_class_ui()
        
        # Assert - Verify divider calls
        divider_calls = mock_feedback.divider.call_args_list
        
        # Should have multiple divider calls with different labels
        assert len(divider_calls) >= 5
        
        # Check for specific divider calls
        divider_args = [call[0] if call[0] else () for call in divider_calls]
        
        # Verify specific labeled dividers
        assert any("Status Table Demo" in str(args) for args in divider_args)
        assert any("Configuration Panel Demo" in str(args) for args in divider_args)
        assert any("Progress Tracking Demo" in str(args) for args in divider_args)
        assert any("File Tree Demo" in str(args) for args in divider_args)
        assert any("Spinner Demo" in str(args) for args in divider_args)
        assert any("Operation Status Demo" in str(args) for args in divider_args)
        assert any("Final Summary" in str(args) for args in divider_args)
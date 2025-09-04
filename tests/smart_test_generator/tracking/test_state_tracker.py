import pytest
import os
import json
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch, mock_open

from smart_test_generator.tracking.state_tracker import TestGenerationTracker
from smart_test_generator.models.data_models import TestGenerationState, TestCoverage
from smart_test_generator.config import Config


class TestTestGenerationTracker:
    """Test TestGenerationTracker class."""

    def test_init_with_default_state_file(self):
        """Test initialization with default state file name."""
        with patch.object(TestGenerationTracker, '_load_state') as mock_load:
            mock_state = TestGenerationState(
                timestamp="2023-01-01T00:00:00",
                tested_elements={},
                coverage_history={},
                generation_log=[]
            )
            mock_load.return_value = mock_state
            
            tracker = TestGenerationTracker()
            
            assert tracker.state_file == ".testgen_state.json"
            assert tracker.state == mock_state
            mock_load.assert_called_once()

    def test_init_with_custom_state_file(self):
        """Test initialization with custom state file name."""
        with patch.object(TestGenerationTracker, '_load_state') as mock_load, \
             patch('os.path.exists', return_value=True):
            mock_state = TestGenerationState(
                timestamp="2023-01-01T00:00:00",
                tested_elements={},
                coverage_history={},
                generation_log=[]
            )
            mock_load.return_value = mock_state
            
            tracker = TestGenerationTracker("custom_state.json")
            
            assert tracker.state_file == "custom_state.json"
            assert tracker.state == mock_state
            mock_load.assert_called_once()

    @patch('builtins.open', new_callable=mock_open, read_data='{"timestamp": "2023-01-01T00:00:00", "tested_elements": {"file1.py": ["func1"]}, "coverage_history": {"file1.py": [80.0]}, "generation_log": []}')
    @patch('os.path.exists', return_value=True)
    def test_load_state_existing_file(self, mock_exists, mock_file):
        """Test loading state from existing file."""
        tracker = TestGenerationTracker()
        
        assert tracker.state.timestamp == "2023-01-01T00:00:00"
        assert tracker.state.tested_elements == {"file1.py": ["func1"]}
        assert tracker.state.coverage_history == {"file1.py": [80.0]}
        assert tracker.state.generation_log == []
        mock_exists.assert_called_with(".testgen_state.json")

    @patch('os.path.exists', return_value=False)
    def test_load_state_no_file(self, mock_exists):
        """Test loading state when no file exists creates default state."""
        with patch('smart_test_generator.tracking.state_tracker.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T00:00:00"
            
            tracker = TestGenerationTracker()
            
            assert tracker.state.timestamp == "2023-01-01T00:00:00"
            assert tracker.state.tested_elements == {}
            assert tracker.state.coverage_history == {}
            assert tracker.state.generation_log == []

    @patch('builtins.open', new_callable=mock_open, read_data='invalid json')
    @patch('os.path.exists', return_value=True)
    def test_load_state_invalid_json(self, mock_exists, mock_file):
        """Test loading state with invalid JSON creates default state."""
        with patch('smart_test_generator.tracking.state_tracker.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T00:00:00"
            
            tracker = TestGenerationTracker()
            
            assert tracker.state.timestamp == "2023-01-01T00:00:00"
            assert tracker.state.tested_elements == {}
            assert tracker.state.coverage_history == {}
            assert tracker.state.generation_log == []

    @patch('builtins.open', new_callable=mock_open)
    def test_save_state_success(self, mock_file):
        """Test successful state saving."""
        with patch('smart_test_generator.tracking.state_tracker.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T12:00:00"
            
            tracker = TestGenerationTracker()
            tracker.save_state()
            
            # Check that the file was opened for writing (there may be other calls for reading during init)
            mock_file.assert_any_call(".testgen_state.json", 'w')
            handle = mock_file()
            written_data = ''.join(call.args[0] for call in handle.write.call_args_list)
            assert "2023-01-01T12:00:00" in written_data

    @patch('builtins.open', side_effect=IOError("Permission denied"))
    def test_save_state_failure(self, mock_file):
        """Test state saving failure is handled gracefully."""
        tracker = TestGenerationTracker()
        
        # Should not raise exception
        tracker.save_state()
        
        # Check that the file was opened for writing (there may be other calls for reading during init)
        mock_file.assert_any_call(".testgen_state.json", 'w')

    def test_should_generate_tests_force_flag(self):
        """Test that force flag always returns True."""
        tracker = TestGenerationTracker()
        config = Mock()
        
        should_generate, reason = tracker.should_generate_tests(
            "test.py", None, config, force=True
        )
        
        assert should_generate is True
        assert reason == "Force flag set"

    def test_should_generate_tests_new_file(self):
        """Test that new files always generate tests."""
        tracker = TestGenerationTracker()
        config = Mock()
        
        should_generate, reason = tracker.should_generate_tests(
            "new_file.py", None, config
        )
        
        assert should_generate is True
        assert reason == "New file detected"

    def test_should_generate_tests_no_coverage_no_elements(self):
        """Test generation when no coverage data and no tested elements."""
        tracker = TestGenerationTracker()
        tracker.state.tested_elements = {}
        config = Mock()
        
        should_generate, reason = tracker.should_generate_tests(
            "test.py", None, config
        )
        
        assert should_generate is True
        assert "New file detected" in reason

    def test_should_generate_tests_no_coverage_with_elements(self):
        """Test skipping generation when no coverage but elements exist."""
        tracker = TestGenerationTracker()
        tracker.state.tested_elements = {"test.py": ["func1"]}
        config = Mock()
        
        should_generate, reason = tracker.should_generate_tests(
            "test.py", None, config
        )
        
        assert should_generate is False
        assert "Tests exist but no recent coverage data - skipping to avoid regeneration" in reason

    def test_should_generate_tests_coverage_below_minimum(self):
        """Test generation when coverage is below minimum threshold."""
        tracker = TestGenerationTracker()
        config = Mock()
        # Setup proper config mocking - return False for always_analyze_new_files, 80 for coverage threshold
        def mock_config_get(key, default=None):
            if key == 'test_generation.generation.always_analyze_new_files':
                return False
            elif key == 'test_generation.coverage.minimum_line_coverage':
                return 80
            elif key == 'test_generation.coverage.minimum_branch_coverage':
                return 70
            return default
        config.get.side_effect = mock_config_get
        
        coverage = TestCoverage(
            filepath="test.py",
            line_coverage=60.0,
            branch_coverage=50.0,
            missing_lines=[1, 2, 3],
            covered_functions=["func1"],
            uncovered_functions=["func2"]
        )
        
        should_generate, reason = tracker.should_generate_tests(
            "test.py", coverage, config
        )
        
        assert should_generate is True
        assert "Line coverage (60.0%) below minimum (80%)" in reason

    def test_should_generate_tests_coverage_dropped_significantly(self):
        """Test generation when coverage drops significantly."""
        tracker = TestGenerationTracker()
        tracker.state.coverage_history = {"test.py": [90.0]}
        config = Mock()
        # Setup proper config mocking
        def mock_config_get(key, default=None):
            if key == 'test_generation.generation.always_analyze_new_files':
                return False
            elif key == 'test_generation.coverage.minimum_line_coverage':
                return 80
            elif key == 'test_generation.coverage.minimum_branch_coverage':
                return 70
            return default
        config.get.side_effect = mock_config_get
        
        coverage = TestCoverage(
            filepath="test.py",
            line_coverage=75.0,
            branch_coverage=70.0,
            missing_lines=[1, 2],
            covered_functions=["func1"],
            uncovered_functions=[]
        )
        
        should_generate, reason = tracker.should_generate_tests(
            "test.py", coverage, config
        )
        
        assert should_generate is True
        assert "Coverage dropped by 15.0%" in reason

    def test_should_generate_tests_untested_functions(self):
        """Test generation when untested functions exist."""
        tracker = TestGenerationTracker()
        config = Mock()
        # Setup proper config mocking
        def mock_config_get(key, default=None):
            if key == 'test_generation.generation.always_analyze_new_files':
                return False
            elif key == 'test_generation.coverage.minimum_line_coverage':
                return 80
            elif key == 'test_generation.coverage.minimum_branch_coverage':
                return 70
            return default
        config.get.side_effect = mock_config_get
        
        coverage = TestCoverage(
            filepath="test.py",
            line_coverage=85.0,
            branch_coverage=80.0,
            missing_lines=[],
            covered_functions=["func1"],
            uncovered_functions=["func2", "func3"]
        )
        
        should_generate, reason = tracker.should_generate_tests(
            "test.py", coverage, config
        )
        
        assert should_generate is True
        assert "2 untested functions found" in reason

    def test_should_generate_tests_adequate_coverage(self):
        """Test skipping generation when coverage is adequate."""
        tracker = TestGenerationTracker()
        config = Mock()
        # Setup proper config mocking
        def mock_config_get(key, default=None):
            if key == 'test_generation.generation.always_analyze_new_files':
                return False
            elif key == 'test_generation.coverage.minimum_line_coverage':
                return 80
            elif key == 'test_generation.coverage.minimum_branch_coverage':
                return 70
            return default
        config.get.side_effect = mock_config_get
        
        coverage = TestCoverage(
            filepath="test.py",
            line_coverage=90.0,
            branch_coverage=85.0,
            missing_lines=[],
            covered_functions=["func1", "func2"],
            uncovered_functions=[]
        )
        
        should_generate, reason = tracker.should_generate_tests(
            "test.py", coverage, config
        )
        
        assert should_generate is False
        assert reason == "Adequate test coverage exists"

    def test_update_coverage_new_file(self):
        """Test updating coverage for a new file."""
        tracker = TestGenerationTracker()
        
        tracker.update_coverage("test.py", 85.0)
        
        assert tracker.state.coverage_history["test.py"] == [85.0]

    def test_update_coverage_existing_file(self):
        """Test updating coverage for an existing file."""
        tracker = TestGenerationTracker()
        tracker.state.coverage_history = {"test.py": [80.0, 82.0]}
        
        tracker.update_coverage("test.py", 85.0)
        
        assert tracker.state.coverage_history["test.py"] == [80.0, 82.0, 85.0]

    def test_update_coverage_limits_history_size(self):
        """Test that coverage history is limited to 10 entries."""
        tracker = TestGenerationTracker()
        # Add 12 coverage values
        for i in range(12):
            tracker.update_coverage("test.py", float(i))
        
        # Should only keep the last 10
        assert len(tracker.state.coverage_history["test.py"]) == 10
        assert tracker.state.coverage_history["test.py"] == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]

    def test_record_generation_new_file(self):
        """Test recording generation for a new file."""
        with patch.object(TestGenerationTracker, '_load_state') as mock_load:
            mock_state = TestGenerationState(
                timestamp="2023-01-01T00:00:00",
                tested_elements={},
                coverage_history={},
                generation_log=[]
            )
            mock_load.return_value = mock_state
            
            tracker = TestGenerationTracker()
            elements = ["test_func1", "test_func2"]
        
            with patch('smart_test_generator.tracking.state_tracker.datetime') as mock_datetime:
                mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T12:00:00"
                
                tracker.record_generation(
                    "test.py", elements, 60.0, 85.0, "Coverage below minimum"
                )
            
            assert len(tracker.state.generation_log) == 1
            log_entry = tracker.state.generation_log[0]
            assert log_entry["filepath"] == "test.py"
            assert log_entry["reason"] == "Coverage below minimum"
            assert log_entry["elements_generated"] == 2
            assert log_entry["coverage_before"] == 60.0
            assert log_entry["coverage_after"] == 85.0
            assert log_entry["improvement"] == 25.0
            
            assert tracker.state.tested_elements["test.py"] == elements

    def test_record_generation_existing_file(self):
        """Test recording generation for an existing file."""
        tracker = TestGenerationTracker()
        tracker.state.tested_elements = {"test.py": ["existing_test"]}
        elements = ["test_func1", "existing_test"]  # Include duplicate
        
        tracker.record_generation(
            "test.py", elements, 70.0, 90.0, "New functions added"
        )
        
        # Should deduplicate elements
        assert set(tracker.state.tested_elements["test.py"]) == {"existing_test", "test_func1"}

    def test_record_generation_limits_log_size(self):
        """Test that generation log is limited to 100 entries."""
        tracker = TestGenerationTracker()
        
        # Add 102 log entries
        for i in range(102):
            tracker.record_generation(
                f"test{i}.py", ["func1"], 60.0, 80.0, "Test reason"
            )
        
        # Should only keep the last 100
        assert len(tracker.state.generation_log) == 100
        assert tracker.state.generation_log[0]["filepath"] == "test2.py"
        assert tracker.state.generation_log[-1]["filepath"] == "test101.py"

    def test_record_generation_truncates_elements_list(self):
        """Test that elements list is truncated to first 10 in log."""
        with patch.object(TestGenerationTracker, '_load_state') as mock_load:
            mock_state = TestGenerationState(
                timestamp="2023-01-01T00:00:00",
                tested_elements={},
                coverage_history={},
                generation_log=[]
            )
            mock_load.return_value = mock_state
            
            tracker = TestGenerationTracker()
            elements = [f"test_func{i}" for i in range(15)]  # 15 elements
            
            tracker.record_generation(
                "test.py", elements, 60.0, 85.0, "Many functions"
            )
            
            log_entry = tracker.state.generation_log[0]
            assert log_entry["elements_generated"] == 15
            assert len(log_entry["elements"]) == 10
            assert log_entry["elements"] == elements[:10]

    def test_reset_state(self):
        """Test resetting the state."""
        tracker = TestGenerationTracker()
        # Add some data
        tracker.state.tested_elements = {"test.py": ["func1"]}
        tracker.state.coverage_history = {"test.py": [80.0]}
        tracker.state.generation_log = [{"test": "data"}]
        
        with patch('smart_test_generator.tracking.state_tracker.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T12:00:00"
            
            with patch.object(tracker, 'save_state') as mock_save:
                tracker.reset_state()
                
                assert tracker.state.timestamp == "2023-01-01T12:00:00"
                assert tracker.state.tested_elements == {}
                assert tracker.state.coverage_history == {}
                assert tracker.state.generation_log == []
                mock_save.assert_called_once()

    def test_force_mark_as_tested_new_file(self):
        """Test force marking elements as tested for a new file."""
        with patch.object(TestGenerationTracker, '_load_state') as mock_load:
            mock_state = TestGenerationState(
                timestamp="2023-01-01T00:00:00",
                tested_elements={},
                coverage_history={},
                generation_log=[]
            )
            mock_load.return_value = mock_state
            
            tracker = TestGenerationTracker()
            elements = ["func1", "func2"]
            
            with patch('smart_test_generator.tracking.state_tracker.datetime') as mock_datetime:
                mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T12:00:00"
                
                with patch.object(tracker, 'save_state') as mock_save:
                    tracker.force_mark_as_tested("test.py", elements, "Manual override")
                    
                    assert tracker.state.tested_elements["test.py"] == elements
                    assert len(tracker.state.generation_log) == 1
                    
                    log_entry = tracker.state.generation_log[0]
                    assert log_entry["reason"] == "Manual override"
                    assert log_entry["elements_generated"] == 2
                    assert log_entry["coverage_after"] == 100
                
                mock_save.assert_called_once()

    def test_force_mark_as_tested_existing_file(self):
        """Test force marking elements as tested for an existing file."""
        tracker = TestGenerationTracker()
        tracker.state.tested_elements = {"test.py": ["existing_func"]}
        elements = ["func1", "existing_func"]  # Include duplicate
        
        with patch.object(tracker, 'save_state') as mock_save:
            tracker.force_mark_as_tested("test.py", elements)
            
            # Should deduplicate
            assert set(tracker.state.tested_elements["test.py"]) == {"existing_func", "func1"}
            mock_save.assert_called_once()

    def test_force_mark_as_tested_default_reason(self):
        """Test force marking with default reason."""
        tracker = TestGenerationTracker()
        
        with patch.object(tracker, 'save_state'):
            tracker.force_mark_as_tested("test.py", ["func1"])
            
            log_entry = tracker.state.generation_log[0]
            # The implementation now automatically detects existing test files instead of using "Manual override"
            assert "Synced with existing tests" in log_entry["reason"]

    def test_get_state_summary(self):
        """Test getting state summary."""
        tracker = TestGenerationTracker()
        tracker.state.timestamp = "2023-01-01T12:00:00"
        tracker.state.tested_elements = {
            "file1.py": ["func1", "func2"],
            "file2.py": ["func3"]
        }
        tracker.state.coverage_history = {
            "file1.py": [80.0, 85.0],
            "file2.py": [90.0]
        }
        tracker.state.generation_log = [{"test": "entry1"}, {"test": "entry2"}]
        
        summary = tracker.get_state_summary()
        
        assert summary["timestamp"] == "2023-01-01T12:00:00"
        assert summary["files_with_tests"] == 2
        assert summary["total_tested_elements"] == 3
        assert summary["files_with_coverage_history"] == 2
        assert summary["generation_log_entries"] == 2
        assert set(summary["tested_files"]) == {"file1.py", "file2.py"}

    def test_get_state_summary_empty_state(self):
        """Test getting state summary with empty state."""
        with patch.object(TestGenerationTracker, '_load_state') as mock_load:
            mock_state = TestGenerationState(
                timestamp="2023-01-01T00:00:00",
                tested_elements={},
                coverage_history={},
                generation_log=[]
            )
            mock_load.return_value = mock_state
            
            tracker = TestGenerationTracker()
            
            summary = tracker.get_state_summary()
            
            assert summary["files_with_tests"] == 0
            assert summary["total_tested_elements"] == 0
            assert summary["files_with_coverage_history"] == 0
            assert summary["generation_log_entries"] == 0
            assert summary["tested_files"] == []

    def test_should_generate_tests_coverage_history_empty_list(self):
        """Test coverage drop check with empty history list."""
        tracker = TestGenerationTracker()
        tracker.state.coverage_history = {"test.py": []}
        config = Mock()
        # Setup proper config mocking
        def mock_config_get(key, default=None):
            if key == 'test_generation.generation.always_analyze_new_files':
                return False
            elif key == 'test_generation.coverage.minimum_line_coverage':
                return 80
            elif key == 'test_generation.coverage.minimum_branch_coverage':
                return 70
            return default
        config.get.side_effect = mock_config_get
        
        coverage = TestCoverage(
            filepath="test.py",
            line_coverage=85.0,
            branch_coverage=80.0,
            missing_lines=[],
            covered_functions=["func1"],
            uncovered_functions=[]
        )
        
        should_generate, reason = tracker.should_generate_tests(
            "test.py", coverage, config
        )
        
        assert should_generate is False
        assert reason == "Adequate test coverage exists"
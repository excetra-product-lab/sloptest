import pytest
import os
import json
import tempfile
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
from datetime import datetime, timedelta

from smart_test_generator.utils.cost_manager import CostManager


class TestCostManager:
    """Test suite for CostManager class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration object."""
        config = Mock()
        config.get.side_effect = lambda key, default: {
            'cost_management.max_file_size_kb': 50,
            'cost_management.max_context_size_chars': 100000,
            'cost_management.max_files_per_request': 15,
            'cost_management.use_cheaper_model_threshold_kb': 10,
            'cost_management.enable_content_compression': True,
            'cost_management.skip_trivial_files': True,
            'cost_management.token_usage_logging': True
        }.get(key, default)
        return config
    
    @pytest.fixture
    def cost_manager(self, mock_config):
        """Create a CostManager instance with mock config."""
        return CostManager(mock_config)
    
    def test_init_with_default_config(self, mock_config):
        """Test CostManager initialization with default configuration values."""
        # Act
        manager = CostManager(mock_config)
        
        # Assert
        assert manager.config == mock_config
        assert manager.token_usage_log == []
        assert manager.max_file_size_kb == 50
        assert manager.max_context_size_chars == 100000
        assert manager.max_files_per_request == 15
        assert manager.use_cheaper_model_threshold_kb == 10
        assert manager.enable_content_compression is True
        assert manager.skip_trivial_files is True
        assert manager.token_usage_logging is True
        assert 'claude-3-haiku-20240307' in manager.model_costs
        assert manager.model_costs['claude-3-haiku-20240307']['input'] == 0.00025
    
    def test_init_with_custom_config(self):
        """Test CostManager initialization with custom configuration values."""
        # Arrange
        config = Mock()
        config.get.side_effect = lambda key, default: {
            'cost_management.max_file_size_kb': 100,
            'cost_management.max_files_per_request': 20,
            'cost_management.enable_content_compression': False
        }.get(key, default)
        
        # Act
        manager = CostManager(config)
        
        # Assert
        assert manager.max_file_size_kb == 100
        assert manager.max_files_per_request == 20
        assert manager.enable_content_compression is False
    
    @patch('os.path.getsize')
    def test_should_skip_file_large_file(self, mock_getsize, cost_manager):
        """Test that large files are skipped."""
        # Arrange
        mock_getsize.return_value = 60 * 1024  # 60KB
        
        # Act
        should_skip, reason = cost_manager.should_skip_file('/path/to/large_file.py')
        
        # Assert
        assert should_skip is True
        assert '60.0KB > 50KB' in reason
    
    @patch('os.path.getsize')
    def test_should_skip_file_normal_size(self, mock_getsize, cost_manager):
        """Test that normal-sized files are not skipped for size reasons."""
        # Arrange
        mock_getsize.return_value = 30 * 1024  # 30KB
        
        # Act
        should_skip, reason = cost_manager.should_skip_file('/path/to/normal_file.py')
        
        # Assert
        assert should_skip is False
        assert reason == ''
    
    @patch('os.path.getsize')
    @patch('builtins.open', new_callable=mock_open, read_data='def func1(): pass\ndef func2(): pass')
    def test_should_skip_file_trivial_content(self, mock_file, mock_getsize, cost_manager):
        """Test that files with trivial content are skipped when enabled."""
        # Arrange
        mock_getsize.return_value = 10 * 1024  # 10KB
        
        # Act
        should_skip, reason = cost_manager.should_skip_file('/path/to/trivial_file.py')
        
        # Assert
        assert should_skip is True
        assert reason == 'File has minimal testable content'
    
    @patch('os.path.getsize')
    def test_should_skip_file_with_error(self, mock_getsize, cost_manager):
        """Test that files with access errors are not skipped."""
        # Arrange
        mock_getsize.side_effect = OSError('Permission denied')
        
        # Act
        should_skip, reason = cost_manager.should_skip_file('/path/to/error_file.py')
        
        # Assert
        assert should_skip is False
        assert reason == ''
    
    def test_optimize_batch_size_empty_files(self, cost_manager):
        """Test batch size optimization with empty file list."""
        # Act
        batch_size = cost_manager.optimize_batch_size([])
        
        # Assert
        assert batch_size == 15  # max_files_per_request
    
    @patch('os.path.getsize')
    def test_optimize_batch_size_large_files(self, mock_getsize, cost_manager):
        """Test batch size optimization with large files."""
        # Arrange
        mock_getsize.return_value = 25 * 1024  # 25KB per file
        files = ['/file1.py', '/file2.py', '/file3.py']
        
        # Act
        batch_size = cost_manager.optimize_batch_size(files)
        
        # Assert
        assert batch_size == 5  # Reduced for large files
    
    @patch('os.path.getsize')
    def test_optimize_batch_size_medium_files(self, mock_getsize, cost_manager):
        """Test batch size optimization with medium files."""
        # Arrange
        mock_getsize.return_value = 15 * 1024  # 15KB per file
        files = ['/file1.py', '/file2.py']
        
        # Act
        batch_size = cost_manager.optimize_batch_size(files)
        
        # Assert
        assert batch_size == 10  # Medium batch size
    
    @patch('os.path.getsize')
    def test_optimize_batch_size_small_files(self, mock_getsize, cost_manager):
        """Test batch size optimization with small files."""
        # Arrange
        mock_getsize.return_value = 5 * 1024  # 5KB per file
        files = ['/file1.py', '/file2.py']
        
        # Act
        batch_size = cost_manager.optimize_batch_size(files)
        
        # Assert
        assert batch_size == 15  # Full batch size for small files
    
    @patch('os.path.getsize')
    def test_optimize_batch_size_with_errors(self, mock_getsize, cost_manager):
        """Test batch size optimization when file access fails."""
        # Arrange
        mock_getsize.side_effect = OSError('File not found')
        files = ['/file1.py', '/file2.py']
        
        # Act
        batch_size = cost_manager.optimize_batch_size(files)
        
        # Assert
        assert batch_size == 15  # Default when no valid files
    
    def test_suggest_model_for_files_empty_list(self, cost_manager):
        """Test model suggestion for empty file list."""
        # Act
        model = cost_manager.suggest_model_for_files([])
        
        # Assert
        assert model == 'claude-3-5-sonnet-20241022'  # Default
    
    @patch('os.path.getsize')
    def test_suggest_model_for_files_small_simple(self, mock_getsize, cost_manager):
        """Test model suggestion for small, simple files."""
        # Arrange
        mock_getsize.return_value = 5 * 1024  # 5KB per file
        files = ['/simple_file.py']
        
        with patch.object(cost_manager, '_has_high_complexity', return_value=False):
            # Act
            model = cost_manager.suggest_model_for_files(files)
        
        # Assert
        assert model == 'claude-3-haiku-20240307'  # Cheaper model
    
    @patch('os.path.getsize')
    def test_suggest_model_for_files_large_complex(self, mock_getsize, cost_manager):
        """Test model suggestion for large, complex files."""
        # Arrange
        mock_getsize.return_value = 20 * 1024  # 20KB per file
        files = ['/complex_file.py']
        
        with patch.object(cost_manager, '_has_high_complexity', return_value=True):
            # Act
            model = cost_manager.suggest_model_for_files(files)
        
        # Assert
        assert model == 'claude-3-7-sonnet-20250219'  # Better model for complexity
    
    @patch('os.path.getsize')
    def test_suggest_model_for_files_large_simple(self, mock_getsize, cost_manager):
        """Test model suggestion for large but simple files."""
        # Arrange
        mock_getsize.return_value = 20 * 1024  # 20KB per file
        files = ['/large_simple_file.py']
        
        with patch.object(cost_manager, '_has_high_complexity', return_value=False):
            # Act
            model = cost_manager.suggest_model_for_files(files)
        
        # Assert
        assert model == 'claude-3-5-sonnet-20241022'  # Balanced model
    
    def test_compress_content_disabled(self, cost_manager):
        """Test content compression when disabled."""
        # Arrange
        cost_manager.enable_content_compression = False
        content = 'def func():\n    # Comment\n    pass\n\n\n'
        
        # Act
        result = cost_manager.compress_content(content)
        
        # Assert
        assert result == content  # Unchanged
    
    def test_compress_content_removes_excessive_whitespace(self, cost_manager):
        """Test that compression removes excessive whitespace."""
        # Arrange
        content = 'def func():\n    pass\n\n\n\n\ndef other():\n    pass'
        
        # Act
        result = cost_manager.compress_content(content)
        
        # Assert
        assert '\n\n\n' not in result
        assert 'def func():\n    pass\n\ndef other():\n    pass' == result
    
    def test_compress_content_removes_comments(self, cost_manager):
        """Test that compression removes single-line comments."""
        # Arrange
        content = 'def func():\n    # This is a comment\n    pass\n    """This is a docstring"""'
        
        # Act
        result = cost_manager.compress_content(content)
        
        # Assert
        assert '# This is a comment' not in result
        assert '"""This is a docstring"""' in result  # Docstrings preserved
    
    def test_compress_content_cleans_import_spacing(self, cost_manager):
        """Test that compression cleans up import spacing."""
        # Arrange
        content = 'import os\n\n\nimport sys\n\n\ndef func(): pass'
        
        # Act
        result = cost_manager.compress_content(content)
        
        # Assert
        assert 'import os\nimport sys\n\ndef func(): pass' == result
    
    def test_log_token_usage_disabled(self, cost_manager):
        """Test token usage logging when disabled."""
        # Arrange
        cost_manager.token_usage_logging = False
        
        # Act
        cost_manager.log_token_usage('claude-3-haiku-20240307', 100, 50)
        
        # Assert
        assert len(cost_manager.token_usage_log) == 0
    
    @patch('smart_test_generator.utils.cost_manager.CostManager._save_usage_log')
    def test_log_token_usage_enabled(self, mock_save, cost_manager):
        """Test token usage logging when enabled."""
        # Act
        cost_manager.log_token_usage('claude-3-haiku-20240307', 100, 50)
        
        # Assert
        assert len(cost_manager.token_usage_log) == 1
        entry = cost_manager.token_usage_log[0]
        assert entry['model'] == 'claude-3-haiku-20240307'
        assert entry['input_tokens'] == 100
        assert entry['output_tokens'] == 50
        assert 'timestamp' in entry
        assert 'estimated_cost' in entry
        mock_save.assert_called_once()
    
    def test_log_token_usage_calculates_cost(self, cost_manager):
        """Test that token usage logging calculates cost correctly."""
        # Act
        cost_manager.log_token_usage('claude-3-haiku-20240307', 1000, 500)
        
        # Assert
        entry = cost_manager.token_usage_log[0]
        expected_cost = (1000/1000 * 0.00025) + (500/1000 * 0.00125)  # 0.00025 + 0.000625 = 0.000875
        assert entry['estimated_cost'] == expected_cost
    
    def test_get_usage_summary_no_log_file(self, cost_manager):
        """Test usage summary when no log file exists."""
        # Act
        with patch('pathlib.Path.exists', return_value=False):
            summary = cost_manager.get_usage_summary()
        
        # Assert
        assert summary == {'total_cost': 0, 'requests': 0, 'total_tokens': 0}
    
    def test_get_usage_summary_with_data(self, cost_manager):
        """Test usage summary with existing log data."""
        # Arrange
        now = datetime.now()
        log_data = [
            {
                'timestamp': (now - timedelta(days=1)).isoformat(),
                'estimated_cost': 0.001,
                'input_tokens': 100,
                'output_tokens': 50
            },
            {
                'timestamp': (now - timedelta(days=10)).isoformat(),  # Too old
                'estimated_cost': 0.002,
                'input_tokens': 200,
                'output_tokens': 100
            }
        ]
        
        # Act
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json.dumps(log_data))):
            summary = cost_manager.get_usage_summary(days=7)
        
        # Assert
        assert summary['total_cost'] == 0.001  # Only recent entry
        assert summary['requests'] == 1
        assert summary['total_tokens'] == 150
        assert summary['average_cost_per_request'] == 0.001
    
    def test_get_usage_summary_with_error(self, cost_manager):
        """Test usage summary when file reading fails."""
        # Act
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', side_effect=OSError('Permission denied')):
            summary = cost_manager.get_usage_summary()
        
        # Assert
        assert summary == {'total_cost': 0, 'requests': 0, 'total_tokens': 0}
    
    @patch('os.path.getsize')
    def test_get_optimization_suggestions_large_files(self, mock_getsize, cost_manager):
        """Test optimization suggestions for large files."""
        # Arrange
        mock_getsize.return_value = 45 * 1024  # 45KB (90% of 50KB limit)
        files = ['/large_file1.py', '/large_file2.py']
        
        # Act
        suggestions = cost_manager.get_optimization_suggestions(files)
        
        # Assert
        assert any('Consider splitting large files' in s for s in suggestions)
        assert any('large_file1.py (45.0KB)' in s for s in suggestions)
    
    @patch('os.path.getsize')
    def test_get_optimization_suggestions_batch_size(self, mock_getsize, cost_manager):
        """Test optimization suggestions for batch size."""
        # Arrange
        mock_getsize.return_value = 25 * 1024  # 25KB per file
        files = [f'/file{i}.py' for i in range(20)]  # 20 files
        
        # Act
        suggestions = cost_manager.get_optimization_suggestions(files)
        
        # Assert
        assert any('Consider reducing batch size to 5' in s for s in suggestions)
    
    @patch('os.path.getsize')
    def test_get_optimization_suggestions_model_recommendation(self, mock_getsize, cost_manager):
        """Test optimization suggestions include model recommendation."""
        # Arrange
        mock_getsize.return_value = 5 * 1024  # 5KB per file
        files = ['/small_file.py']
        
        # Act
        suggestions = cost_manager.get_optimization_suggestions(files)
        
        # Assert
        assert any('Recommended model for this batch:' in s for s in suggestions)
    
    @patch('os.path.getsize')
    def test_get_optimization_suggestions_with_errors(self, mock_getsize, cost_manager):
        """Test optimization suggestions when file access fails."""
        # Arrange
        mock_getsize.side_effect = OSError('File not found')
        files = ['/error_file.py']
        
        # Act
        suggestions = cost_manager.get_optimization_suggestions(files)
        
        # Assert
        assert len(suggestions) >= 1  # At least model recommendation
        assert any('Recommended model for this batch:' in s for s in suggestions)
    
    def test_get_optimization_suggestions_normal_files(self, cost_manager):
        """Test optimization suggestions for normal-sized files."""
        # Arrange
        files = ['/normal_file.py']
        
        with patch('os.path.getsize', return_value=10 * 1024):  # 10KB
            # Act
            suggestions = cost_manager.get_optimization_suggestions(files)
        
        # Assert
        # Should not suggest splitting files or reducing batch size
        assert not any('Consider splitting large files' in s for s in suggestions)
        assert not any('Consider reducing batch size' in s for s in suggestions)
        assert any('Recommended model for this batch:' in s for s in suggestions)
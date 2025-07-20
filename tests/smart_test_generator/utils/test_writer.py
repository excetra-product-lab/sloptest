import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from smart_test_generator.utils.writer import TestFileWriter
from smart_test_generator.config import Config


class TestTestFileWriter:
    """Test TestFileWriter class."""

    def test_init_with_root_dir_only(self):
        """Test initialization with only root_dir parameter."""
        # Arrange
        root_dir = "/path/to/project"
        
        # Act
        writer = TestFileWriter(root_dir)
        
        # Assert
        assert writer.root_dir == Path(root_dir).resolve()
        assert isinstance(writer.config, Config)

    def test_init_with_root_dir_and_config(self):
        """Test initialization with both root_dir and config parameters."""
        # Arrange
        root_dir = "/path/to/project"
        config = Mock(spec=Config)
        
        # Act
        writer = TestFileWriter(root_dir, config)
        
        # Assert
        assert writer.root_dir == Path(root_dir).resolve()
        assert writer.config is config

    def test_init_with_relative_path(self):
        """Test initialization with relative path resolves to absolute."""
        # Arrange
        root_dir = "./relative/path"
        
        # Act
        writer = TestFileWriter(root_dir)
        
        # Assert
        assert writer.root_dir.is_absolute()
        assert str(writer.root_dir).endswith("relative/path")

    def test_determine_test_path_for_existing_test_file(self):
        """Test determine_test_path returns same path for existing test files."""
        # Arrange
        writer = TestFileWriter("/project")
        source_path = "tests/test_example.py"
        
        # Act
        result = writer.determine_test_path(source_path)
        
        # Assert
        assert result == "tests/test_example.py"

    def test_determine_test_path_for_src_file(self):
        """Test determine_test_path replaces src with tests directory."""
        # Arrange
        writer = TestFileWriter("/project")
        source_path = "src/module/example.py"
        
        # Act
        result = writer.determine_test_path(source_path)
        
        # Assert
        assert result == "tests/module/test_example.py"

    def test_determine_test_path_for_non_src_file(self):
        """Test determine_test_path creates tests directory for non-src files."""
        # Arrange
        writer = TestFileWriter("/project")
        source_path = "lib/utils.py"
        
        # Act
        result = writer.determine_test_path(source_path)
        
        # Assert
        assert result == "tests/lib/test_utils.py"

    def test_determine_test_path_for_file_already_with_test_prefix(self):
        """Test determine_test_path doesn't add test_ prefix if already present."""
        # Arrange
        writer = TestFileWriter("/project")
        source_path = "src/test_helper.py"
        
        # Act
        result = writer.determine_test_path(source_path)
        
        # Assert
        assert result == "tests/test_helper.py"

    def test_determine_test_path_for_nested_src_structure(self):
        """Test determine_test_path handles deeply nested src structures."""
        # Arrange
        writer = TestFileWriter("/project")
        source_path = "src/package/subpackage/module.py"
        
        # Act
        result = writer.determine_test_path(source_path)
        
        # Assert
        assert result == "tests/package/subpackage/test_module.py"

    def test_determine_test_path_for_root_level_file(self):
        """Test determine_test_path handles root level files."""
        # Arrange
        writer = TestFileWriter("/project")
        source_path = "main.py"
        
        # Act
        result = writer.determine_test_path(source_path)
        
        # Assert
        assert result == "tests/test_main.py"

    @patch('smart_test_generator.utils.writer.logger')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=True)
    def test_write_test_file_success(self, mock_exists, mock_mkdir, mock_file, mock_logger):
        """Test write_test_file successfully writes content to file."""
        # Arrange
        writer = TestFileWriter("/project")
        source_path = "src/example.py"
        test_content = "import pytest\n\ndef test_example():\n    pass"
        
        # Act
        writer.write_test_file(source_path, test_content)
        
        # Assert
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_file.assert_called_once_with(Path("/project/tests/test_example.py"), 'w', encoding='utf-8')
        mock_file().write.assert_called_once_with(test_content)
        mock_logger.info.assert_called_once()

    @patch('smart_test_generator.utils.writer.logger')
    @patch('pathlib.Path.mkdir')
    def test_write_test_file_creates_directory(self, mock_mkdir, mock_logger):
        """Test write_test_file creates parent directories."""
        # Arrange
        writer = TestFileWriter("/project")
        source_path = "src/deep/nested/example.py"
        test_content = "test content"
        
        with patch('builtins.open', mock_open()):
            # Act
            writer.write_test_file(source_path, test_content)
            
            # Assert
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch('smart_test_generator.utils.writer.logger')
    @patch('builtins.open', side_effect=IOError("Permission denied"))
    @patch('pathlib.Path.mkdir')
    def test_write_test_file_handles_write_error(self, mock_mkdir, mock_file, mock_logger):
        """Test write_test_file handles and logs write errors."""
        # Arrange
        writer = TestFileWriter("/project")
        source_path = "src/example.py"
        test_content = "test content"
        
        # Act & Assert
        with pytest.raises(IOError, match="Permission denied"):
            writer.write_test_file(source_path, test_content)
        
        mock_logger.error.assert_called_once()

    @patch('smart_test_generator.utils.writer.logger')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists', return_value=False)
    def test_update_test_file_creates_new_file_when_not_exists(self, mock_exists, mock_file, mock_logger):
        """Test update_test_file creates new file when it doesn't exist."""
        # Arrange
        writer = TestFileWriter("/project")
        test_path = "tests/test_example.py"
        new_content = "new test content"
        
        with patch.object(writer, 'write_test_file') as mock_write:
            # Act
            writer.update_test_file(test_path, new_content)
            
            # Assert
            mock_write.assert_called_once_with(test_path, new_content)

    @patch('smart_test_generator.utils.writer.logger')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists', return_value=True)
    def test_update_test_file_appends_content_by_default(self, mock_exists, mock_file, mock_logger):
        """Test update_test_file appends content by default."""
        # Arrange
        writer = TestFileWriter("/project")
        test_path = "tests/test_example.py"
        new_content = "new test content"
        
        # Act
        writer.update_test_file(test_path, new_content)
        
        # Assert
        mock_file.assert_called_once_with(Path("/project/tests/test_example.py"), 'a', encoding='utf-8')
        mock_file().write.assert_called_once_with("\n\n" + new_content)
        mock_logger.info.assert_called_once()

    @patch('smart_test_generator.utils.writer.logger')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists', return_value=True)
    def test_update_test_file_replaces_content_when_specified(self, mock_exists, mock_file, mock_logger):
        """Test update_test_file replaces content when merge_strategy is 'replace'."""
        # Arrange
        writer = TestFileWriter("/project")
        test_path = "tests/test_example.py"
        new_content = "replacement content"
        
        # Act
        writer.update_test_file(test_path, new_content, merge_strategy='replace')
        
        # Assert
        mock_file.assert_called_once_with(Path("/project/tests/test_example.py"), 'w', encoding='utf-8')
        mock_file().write.assert_called_once_with(new_content)
        mock_logger.info.assert_called_once()

    @patch('smart_test_generator.utils.writer.logger')
    @patch('builtins.open', side_effect=IOError("File locked"))
    @patch('pathlib.Path.exists', return_value=True)
    def test_update_test_file_handles_update_error(self, mock_exists, mock_file, mock_logger):
        """Test update_test_file handles and logs update errors."""
        # Arrange
        writer = TestFileWriter("/project")
        test_path = "tests/test_example.py"
        new_content = "new content"
        
        # Act
        writer.update_test_file(test_path, new_content)
        
        # Assert
        mock_logger.error.assert_called_once()

    @patch('smart_test_generator.utils.writer.logger')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.exists', return_value=True)
    def test_update_test_file_with_custom_merge_strategy(self, mock_exists, mock_file, mock_logger):
        """Test update_test_file handles custom merge strategies."""
        # Arrange
        writer = TestFileWriter("/project")
        test_path = "tests/test_example.py"
        new_content = "custom content"
        
        # Act
        writer.update_test_file(test_path, new_content, merge_strategy='append')
        
        # Assert
        mock_file.assert_called_once_with(Path("/project/tests/test_example.py"), 'a', encoding='utf-8')
        mock_file().write.assert_called_once_with("\n\n" + new_content)

    def test_determine_test_path_with_pathlib_path_object(self):
        """Test determine_test_path works with Path objects."""
        # Arrange
        writer = TestFileWriter("/project")
        source_path = Path("src/module/example.py")
        
        # Act
        result = writer.determine_test_path(str(source_path))
        
        # Assert
        assert result == "tests/module/test_example.py"

    @patch('smart_test_generator.utils.writer.logger')
    def test_write_test_file_logs_debug_information(self, mock_logger):
        """Test write_test_file logs appropriate debug information."""
        # Arrange
        writer = TestFileWriter("/project")
        source_path = "src/example.py"
        test_content = "test content"
        
        with patch('builtins.open', mock_open()), \
             patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.exists', return_value=True):
            
            # Act
            writer.write_test_file(source_path, test_content)
            
            # Assert
            assert mock_logger.debug.call_count >= 3  # Multiple debug calls expected
            mock_logger.info.assert_called_once()

    @patch('smart_test_generator.utils.writer.logger')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=True)
    def test_write_test_file_with_large_content(self, mock_exists, mock_mkdir, mock_file, mock_logger):
        """Test write_test_file handles large test content."""
        # Arrange
        writer = TestFileWriter("/project")
        source_path = "src/example.py"
        test_content = "x" * 10000  # Large content
        
        # Act
        writer.write_test_file(source_path, test_content)
        
        # Assert
        mock_file().write.assert_called_once_with(test_content)
        # Check that character count is logged
        info_call_args = mock_logger.info.call_args[0][0]
        assert "10,000 characters" in info_call_args
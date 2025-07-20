import pytest
import os
import ast
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from smart_test_generator.utils.file_utils import (
    FileUtils,
    format_file_size,
    batch_files,
    should_exclude_file
)
from smart_test_generator.exceptions import FileOperationError


class TestFileUtils:
    """Test FileUtils class methods."""

    def test_find_files_by_pattern_finds_matching_files(self, tmp_path):
        """Test that find_files_by_pattern finds files matching given patterns."""
        # Arrange
        (tmp_path / "file1.py").write_text("# Python file")
        (tmp_path / "file2.txt").write_text("Text file")
        (tmp_path / "test_file.py").write_text("# Test file")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.py").write_text("# Nested file")
        
        # Act
        result = FileUtils.find_files_by_pattern(tmp_path, ["*.py"])
        
        # Assert
        assert len(result) == 3
        file_names = [f.name for f in result]
        assert "file1.py" in file_names
        assert "test_file.py" in file_names
        assert "nested.py" in file_names
        assert "file2.txt" not in [f.name for f in result]

    def test_find_files_by_pattern_excludes_directories(self, tmp_path):
        """Test that find_files_by_pattern excludes specified directories."""
        # Arrange
        (tmp_path / "main.py").write_text("# Main file")
        venv_dir = tmp_path / "venv"
        venv_dir.mkdir()
        (venv_dir / "lib.py").write_text("# Venv file")
        
        # Act
        result = FileUtils.find_files_by_pattern(tmp_path, ["*.py"])
        
        # Assert
        assert len(result) == 1
        assert result[0].name == "main.py"

    def test_find_files_by_pattern_with_multiple_patterns(self, tmp_path):
        """Test that find_files_by_pattern handles multiple patterns."""
        # Arrange
        (tmp_path / "file.py").write_text("# Python file")
        (tmp_path / "file.txt").write_text("Text file")
        (tmp_path / "file.md").write_text("Markdown file")
        
        # Act
        result = FileUtils.find_files_by_pattern(tmp_path, ["*.py", "*.txt"])
        
        # Assert
        assert len(result) == 2
        file_names = [f.name for f in result]
        assert "file.py" in file_names
        assert "file.txt" in file_names
        assert "file.md" not in file_names

    def test_find_files_by_pattern_empty_directory(self, tmp_path):
        """Test that find_files_by_pattern handles empty directories."""
        # Act
        result = FileUtils.find_files_by_pattern(tmp_path, ["*.py"])
        
        # Assert
        assert result == []

    def test_get_relative_path_with_common_base(self, tmp_path):
        """Test that get_relative_path returns correct relative path."""
        # Arrange
        file_path = tmp_path / "subdir" / "file.py"
        base_path = tmp_path
        
        # Act
        result = FileUtils.get_relative_path(file_path, base_path)
        
        # Assert
        assert result == "subdir/file.py" or result == "subdir\\file.py"

    def test_get_relative_path_without_common_base(self):
        """Test that get_relative_path returns absolute path when no common base."""
        # Arrange
        file_path = Path("/completely/different/path/file.py")
        base_path = Path("/another/path")
        
        # Act
        result = FileUtils.get_relative_path(file_path, base_path)
        
        # Assert
        assert result == str(file_path)

    def test_ensure_directory_exists_creates_directory(self, tmp_path):
        """Test that ensure_directory_exists creates directory when it doesn't exist."""
        # Arrange
        new_dir = tmp_path / "new" / "nested" / "directory"
        assert not new_dir.exists()
        
        # Act
        FileUtils.ensure_directory_exists(new_dir)
        
        # Assert
        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_ensure_directory_exists_handles_existing_directory(self, tmp_path):
        """Test that ensure_directory_exists handles existing directories gracefully."""
        # Arrange
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        
        # Act & Assert (should not raise)
        FileUtils.ensure_directory_exists(existing_dir)
        assert existing_dir.exists()

    @patch('pathlib.Path.mkdir')
    def test_ensure_directory_exists_raises_on_permission_error(self, mock_mkdir):
        """Test that ensure_directory_exists raises FileOperationError on permission error."""
        # Arrange
        mock_mkdir.side_effect = PermissionError("Permission denied")
        directory = Path("/restricted/path")
        
        # Act & Assert
        with pytest.raises(FileOperationError) as exc_info:
            FileUtils.ensure_directory_exists(directory)
        
        assert "Permission denied" in str(exc_info.value)
        assert exc_info.value.filepath == str(directory)

    @patch('pathlib.Path.mkdir')
    def test_ensure_directory_exists_raises_on_os_error(self, mock_mkdir):
        """Test that ensure_directory_exists raises FileOperationError on OS error."""
        # Arrange
        mock_mkdir.side_effect = OSError("Disk full")
        directory = Path("/some/path")
        
        # Act & Assert
        with pytest.raises(FileOperationError) as exc_info:
            FileUtils.ensure_directory_exists(directory)
        
        assert "Failed to create directory" in str(exc_info.value)

    def test_read_file_safely_reads_file_content(self, tmp_path):
        """Test that read_file_safely reads file content correctly."""
        # Arrange
        file_path = tmp_path / "test.txt"
        content = "Hello, World!\nThis is a test file."
        file_path.write_text(content, encoding='utf-8')
        
        # Act
        result = FileUtils.read_file_safely(file_path)
        
        # Assert
        assert result == content

    def test_read_file_safely_with_custom_encoding(self, tmp_path):
        """Test that read_file_safely handles custom encoding."""
        # Arrange
        file_path = tmp_path / "test.txt"
        content = "Test content"
        file_path.write_text(content, encoding='latin-1')
        
        # Act
        result = FileUtils.read_file_safely(file_path, encoding='latin-1')
        
        # Assert
        assert result == content

    def test_read_file_safely_raises_on_file_not_found(self):
        """Test that read_file_safely raises FileOperationError when file not found."""
        # Arrange
        file_path = Path("/nonexistent/file.txt")
        
        # Act & Assert
        with pytest.raises(FileOperationError) as exc_info:
            FileUtils.read_file_safely(file_path)
        
        assert "File not found" in str(exc_info.value)
        assert exc_info.value.filepath == str(file_path)

    @patch('builtins.open')
    def test_read_file_safely_raises_on_permission_error(self, mock_open_func):
        """Test that read_file_safely raises FileOperationError on permission error."""
        # Arrange
        mock_open_func.side_effect = PermissionError("Permission denied")
        file_path = Path("/restricted/file.txt")
        
        # Act & Assert
        with pytest.raises(FileOperationError) as exc_info:
            FileUtils.read_file_safely(file_path)
        
        assert "Permission denied" in str(exc_info.value)

    @patch('builtins.open')
    def test_read_file_safely_raises_on_unicode_decode_error(self, mock_open_func):
        """Test that read_file_safely raises FileOperationError on encoding error."""
        # Arrange
        mock_file = Mock()
        mock_file.read.side_effect = UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid start byte')
        mock_open_func.return_value.__enter__.return_value = mock_file
        file_path = Path("/binary/file.bin")
        
        # Act & Assert
        with pytest.raises(FileOperationError) as exc_info:
            FileUtils.read_file_safely(file_path)
        
        assert "File encoding error" in str(exc_info.value)

    def test_write_file_safely_writes_content(self, tmp_path):
        """Test that write_file_safely writes content to file."""
        # Arrange
        file_path = tmp_path / "output.txt"
        content = "Hello, World!\nThis is test content."
        
        # Act
        FileUtils.write_file_safely(file_path, content)
        
        # Assert
        assert file_path.exists()
        assert file_path.read_text(encoding='utf-8') == content

    def test_write_file_safely_creates_parent_directories(self, tmp_path):
        """Test that write_file_safely creates parent directories."""
        # Arrange
        file_path = tmp_path / "nested" / "deep" / "file.txt"
        content = "Test content"
        
        # Act
        FileUtils.write_file_safely(file_path, content)
        
        # Assert
        assert file_path.exists()
        assert file_path.read_text() == content

    def test_write_file_safely_with_custom_encoding(self, tmp_path):
        """Test that write_file_safely handles custom encoding."""
        # Arrange
        file_path = tmp_path / "encoded.txt"
        content = "Test content with special chars: ñáéíóú"
        
        # Act
        FileUtils.write_file_safely(file_path, content, encoding='latin-1')
        
        # Assert
        assert file_path.read_text(encoding='latin-1') == content

    @patch('builtins.open')
    def test_write_file_safely_raises_on_permission_error(self, mock_open_func):
        """Test that write_file_safely raises FileOperationError on permission error."""
        # Arrange
        mock_open_func.side_effect = PermissionError("Permission denied")
        file_path = Path("/restricted/file.txt")
        
        # Act & Assert
        with pytest.raises(FileOperationError) as exc_info:
            FileUtils.write_file_safely(file_path, "content")
        
        assert "Permission denied" in str(exc_info.value)

    @patch('builtins.open')
    def test_write_file_safely_raises_on_os_error(self, mock_open_func):
        """Test that write_file_safely raises FileOperationError on OS error."""
        # Arrange
        mock_open_func.side_effect = OSError("Disk full")
        file_path = Path("/some/file.txt")
        
        # Act & Assert
        with pytest.raises(FileOperationError) as exc_info:
            FileUtils.write_file_safely(file_path, "content")
        
        assert "Failed to write file" in str(exc_info.value)

    def test_get_file_size_returns_correct_size(self, tmp_path):
        """Test that get_file_size returns correct file size."""
        # Arrange
        file_path = tmp_path / "test.txt"
        content = "Hello, World!"  # 13 bytes
        file_path.write_text(content)
        
        # Act
        result = FileUtils.get_file_size(file_path)
        
        # Assert
        assert result == 13

    def test_get_file_size_returns_zero_on_error(self):
        """Test that get_file_size returns 0 when file doesn't exist."""
        # Arrange
        file_path = Path("/nonexistent/file.txt")
        
        # Act
        result = FileUtils.get_file_size(file_path)
        
        # Assert
        assert result == 0

    def test_is_python_file_identifies_python_files(self, tmp_path):
        """Test that is_python_file correctly identifies Python files."""
        # Arrange
        python_file = tmp_path / "script.py"
        python_file.write_text("# Python file")
        text_file = tmp_path / "readme.txt"
        text_file.write_text("Text file")
        
        # Act & Assert
        assert FileUtils.is_python_file(python_file) is True
        assert FileUtils.is_python_file(text_file) is False

    def test_is_python_file_requires_existing_file(self, tmp_path):
        """Test that is_python_file returns False for non-existent files."""
        # Arrange
        nonexistent_file = tmp_path / "nonexistent.py"
        
        # Act & Assert
        assert FileUtils.is_python_file(nonexistent_file) is False

    def test_is_test_file_identifies_test_files_in_test_directory(self, tmp_path):
        """Test that is_test_file identifies files in test directories."""
        # Arrange
        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        test_file = test_dir / "any_file.py"
        test_file.write_text("# Any file in test dir")
        
        # Act & Assert
        assert FileUtils.is_test_file(test_file) is True

    def test_is_test_file_identifies_test_filename_patterns(self, tmp_path):
        """Test that is_test_file identifies test filename patterns."""
        # Arrange
        test_files = [
            tmp_path / "test_main.py",
            tmp_path / "test_utils.py",
            tmp_path / "module_test.py",
            tmp_path / "conftest.py"
        ]
        
        for test_file in test_files:
            test_file.write_text("# Test file")
        
        # Act & Assert
        for test_file in test_files:
            assert FileUtils.is_test_file(test_file) is True

    def test_is_test_file_rejects_complex_test_names(self, tmp_path):
        """Test that is_test_file rejects complex test file names outside test dirs."""
        # Arrange
        complex_files = [
            tmp_path / "test_data_processor.py",
            tmp_path / "test_generator.py",
            tmp_path / "test_complex_name.py"
        ]
        
        for test_file in complex_files:
            test_file.write_text("# Complex test file")
        
        # Act & Assert
        for test_file in complex_files:
            assert FileUtils.is_test_file(test_file) is False

    def test_is_test_file_accepts_non_test_files(self, tmp_path):
        """Test that is_test_file correctly identifies non-test files."""
        # Arrange
        regular_file = tmp_path / "main.py"
        regular_file.write_text("# Regular file")
        
        # Act & Assert
        assert FileUtils.is_test_file(regular_file) is False

    def test_is_test_file_ast_analyzes_file_content(self, tmp_path):
        """Test that is_test_file_ast analyzes file content for test functions."""
        # Arrange
        test_file = tmp_path / "analysis.py"
        test_content = """
def test_something():
    assert True

def regular_function():
    pass
"""
        test_file.write_text(test_content)
        
        # Act & Assert
        assert FileUtils.is_test_file_ast(test_file) is True

    def test_is_test_file_ast_identifies_test_classes(self, tmp_path):
        """Test that is_test_file_ast identifies test classes."""
        # Arrange
        test_file = tmp_path / "test_class.py"
        test_content = """
import unittest

class TestSomething(unittest.TestCase):
    def test_method(self):
        self.assertTrue(True)
"""
        test_file.write_text(test_content)
        
        # Act & Assert
        assert FileUtils.is_test_file_ast(test_file) is True

    def test_is_test_file_ast_handles_non_test_files(self, tmp_path):
        """Test that is_test_file_ast correctly identifies non-test files."""
        # Arrange
        regular_file = tmp_path / "regular.py"
        regular_content = """
def regular_function():
    return "Hello, World!"

class RegularClass:
    def method(self):
        pass
"""
        regular_file.write_text(regular_content)
        
        # Act & Assert
        assert FileUtils.is_test_file_ast(regular_file) is False

    def test_is_test_file_ast_handles_syntax_errors(self, tmp_path):
        """Test that is_test_file_ast handles files with syntax errors."""
        # Arrange
        broken_file = tmp_path / "broken.py"
        broken_file.write_text("def broken_syntax(:\n    pass")
        
        # Act & Assert
        assert FileUtils.is_test_file_ast(broken_file) is False

    def test_find_test_files_ast_finds_test_files(self, tmp_path):
        """Test that find_test_files_ast finds test files using AST analysis."""
        # Arrange
        test_file = tmp_path / "test_module.py"
        test_file.write_text("def test_function(): pass")
        
        regular_file = tmp_path / "regular.py"
        regular_file.write_text("def regular_function(): pass")
        
        # Act
        result = FileUtils.find_test_files_ast(tmp_path)
        
        # Assert
        assert len(result) == 1
        assert result[0].name == "test_module.py"

    def test_find_test_files_ast_excludes_directories(self, tmp_path):
        """Test that find_test_files_ast excludes specified directories."""
        # Arrange
        test_file = tmp_path / "test_main.py"
        test_file.write_text("def test_function(): pass")
        
        venv_dir = tmp_path / "venv"
        venv_dir.mkdir()
        venv_test = venv_dir / "test_venv.py"
        venv_test.write_text("def test_function(): pass")
        
        # Act
        result = FileUtils.find_test_files_ast(tmp_path)
        
        # Assert
        assert len(result) == 1
        assert result[0].name == "test_main.py"

    def test_is_config_file_identifies_config_files(self, tmp_path):
        """Test that is_config_file identifies configuration files."""
        # Arrange
        config_files = [
            tmp_path / "config.py",
            tmp_path / "settings.py",
            tmp_path / "setup.py",
            tmp_path / "__init__.py",
            tmp_path / ".env"
        ]
        
        for config_file in config_files:
            config_file.write_text("# Config file")
        
        # Act & Assert
        for config_file in config_files:
            assert FileUtils.is_config_file(config_file) is True

    def test_is_config_file_rejects_regular_files(self, tmp_path):
        """Test that is_config_file rejects regular files."""
        # Arrange
        regular_file = tmp_path / "main.py"
        regular_file.write_text("# Regular file")
        
        # Act & Assert
        assert FileUtils.is_config_file(regular_file) is False

    def test_filter_python_files_filters_by_extension(self, tmp_path):
        """Test that filter_python_files filters files by Python extension."""
        # Arrange
        python_file = tmp_path / "script.py"
        python_file.write_text("# Python file")
        text_file = tmp_path / "readme.txt"
        text_file.write_text("Text file")
        
        files = [python_file, text_file]
        
        # Act
        result = FileUtils.filter_python_files(files)
        
        # Assert
        assert len(result) == 1
        assert result[0] == python_file

    def test_filter_python_files_excludes_config_files(self, tmp_path):
        """Test that filter_python_files excludes config files when requested."""
        # Arrange
        regular_file = tmp_path / "main.py"
        regular_file.write_text("# Regular file")
        config_file = tmp_path / "config.py"
        config_file.write_text("# Config file")
        
        files = [regular_file, config_file]
        
        # Act
        result = FileUtils.filter_python_files(files, exclude_configs=True)
        
        # Assert
        assert len(result) == 1
        assert result[0] == regular_file

    def test_filter_python_files_excludes_test_files(self, tmp_path):
        """Test that filter_python_files excludes test files when requested."""
        # Arrange
        regular_file = tmp_path / "main.py"
        regular_file.write_text("# Regular file")
        test_file = tmp_path / "test_main.py"
        test_file.write_text("# Test file")
        
        files = [regular_file, test_file]
        
        # Act
        result = FileUtils.filter_python_files(files, exclude_tests=True)
        
        # Assert
        assert len(result) == 1
        assert result[0] == regular_file

    def test_group_files_by_directory_groups_correctly(self, tmp_path):
        """Test that group_files_by_directory groups files by parent directory."""
        # Arrange
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        dir2 = tmp_path / "dir2"
        dir2.mkdir()
        
        file1 = dir1 / "file1.py"
        file1.write_text("# File 1")
        file2 = dir1 / "file2.py"
        file2.write_text("# File 2")
        file3 = dir2 / "file3.py"
        file3.write_text("# File 3")
        
        files = [file1, file2, file3]
        
        # Act
        result = FileUtils.group_files_by_directory(files)
        
        # Assert
        assert len(result) == 2
        assert str(dir1) in result
        assert str(dir2) in result
        assert len(result[str(dir1)]) == 2
        assert len(result[str(dir2)]) == 1

    def test_find_matching_test_file_finds_test_file(self, tmp_path):
        """Test that find_matching_test_file finds corresponding test file."""
        # Arrange
        source_file = tmp_path / "main.py"
        source_file.write_text("# Source file")
        
        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        test_file = test_dir / "test_main.py"
        test_file.write_text("# Test file")
        
        # Act
        result = FileUtils.find_matching_test_file(source_file, [test_dir])
        
        # Assert
        assert result == test_file

    def test_find_matching_test_file_returns_none_when_not_found(self, tmp_path):
        """Test that find_matching_test_file returns None when no test file found."""
        # Arrange
        source_file = tmp_path / "main.py"
        source_file.write_text("# Source file")
        
        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        
        # Act
        result = FileUtils.find_matching_test_file(source_file, [test_dir])
        
        # Assert
        assert result is None

    def test_calculate_test_file_path_with_src_structure(self, tmp_path):
        """Test that calculate_test_file_path handles src structure correctly."""
        # Arrange
        source_file = tmp_path / "src" / "module" / "main.py"
        
        # Act
        result = FileUtils.calculate_test_file_path(source_file, tmp_path)
        
        # Assert
        expected = tmp_path / "tests" / "module" / "test_main.py"
        assert result == expected

    def test_calculate_test_file_path_without_src_structure(self, tmp_path):
        """Test that calculate_test_file_path handles non-src structure correctly."""
        # Arrange
        source_file = tmp_path / "module" / "main.py"
        
        # Act
        result = FileUtils.calculate_test_file_path(source_file, tmp_path)
        
        # Assert
        expected = tmp_path / "tests" / "module" / "test_main.py"
        assert result == expected

    def test_get_module_name_from_path_returns_dot_notation(self, tmp_path):
        """Test that get_module_name_from_path returns module name in dot notation."""
        # Arrange
        file_path = tmp_path / "package" / "subpackage" / "module.py"
        
        # Act
        result = FileUtils.get_module_name_from_path(file_path, tmp_path)
        
        # Assert
        assert result == "package.subpackage.module"

    def test_get_module_name_from_path_handles_no_common_base(self):
        """Test that get_module_name_from_path handles files outside project root."""
        # Arrange
        file_path = Path("/different/path/module.py")
        project_root = Path("/project/root")
        
        # Act
        result = FileUtils.get_module_name_from_path(file_path, project_root)
        
        # Assert
        assert result == "module"

    def test_estimate_file_complexity_counts_lines_and_functions(self, tmp_path):
        """Test that estimate_file_complexity counts code lines and functions."""
        # Arrange
        file_path = tmp_path / "module.py"
        content = """
# This is a comment

def function1():
    return "hello"

def function2():
    return "world"

async def async_function():
    return "async"

class MyClass:
    pass
"""
        file_path.write_text(content)
        
        # Act
        lines, functions = FileUtils.estimate_file_complexity(file_path)
        
        # Assert
        assert lines > 0  # Should count non-empty, non-comment lines
        assert functions == 3  # Should count 3 functions

    def test_estimate_file_complexity_handles_empty_file(self, tmp_path):
        """Test that estimate_file_complexity handles empty files."""
        # Arrange
        file_path = tmp_path / "empty.py"
        file_path.write_text("")
        
        # Act
        lines, functions = FileUtils.estimate_file_complexity(file_path)
        
        # Assert
        assert lines == 0
        assert functions == 0

    def test_estimate_file_complexity_handles_read_error(self):
        """Test that estimate_file_complexity handles file read errors."""
        # Arrange
        file_path = Path("/nonexistent/file.py")
        
        # Act
        lines, functions = FileUtils.estimate_file_complexity(file_path)
        
        # Assert
        assert lines == 0
        assert functions == 0


class TestFormatFileSize:
    """Test format_file_size function."""

    def test_format_file_size_bytes(self):
        """Test that format_file_size formats bytes correctly."""
        # Act & Assert
        assert format_file_size(512) == "512.0 B"
        assert format_file_size(0) == "0.0 B"
        assert format_file_size(1023) == "1023.0 B"

    def test_format_file_size_kilobytes(self):
        """Test that format_file_size formats kilobytes correctly."""
        # Act & Assert
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1536) == "1.5 KB"
        assert format_file_size(1048575) == "1024.0 KB"

    def test_format_file_size_megabytes(self):
        """Test that format_file_size formats megabytes correctly."""
        # Act & Assert
        assert format_file_size(1048576) == "1.0 MB"
        assert format_file_size(1572864) == "1.5 MB"

    def test_format_file_size_gigabytes(self):
        """Test that format_file_size formats gigabytes correctly."""
        # Act & Assert
        assert format_file_size(1073741824) == "1.0 GB"
        assert format_file_size(1610612736) == "1.5 GB"

    def test_format_file_size_terabytes(self):
        """Test that format_file_size formats terabytes correctly."""
        # Act & Assert
        assert format_file_size(1099511627776) == "1.0 TB"


class TestBatchFiles:
    """Test batch_files function."""

    def test_batch_files_creates_correct_batches(self):
        """Test that batch_files creates batches of correct size."""
        # Arrange
        files = [Path(f"file{i}.py") for i in range(10)]
        batch_size = 3
        
        # Act
        result = batch_files(files, batch_size)
        
        # Assert
        assert len(result) == 4  # 10 files / 3 per batch = 4 batches
        assert len(result[0]) == 3
        assert len(result[1]) == 3
        assert len(result[2]) == 3
        assert len(result[3]) == 1  # Last batch has remainder

    def test_batch_files_handles_empty_list(self):
        """Test that batch_files handles empty file list."""
        # Arrange
        files = []
        batch_size = 5
        
        # Act
        result = batch_files(files, batch_size)
        
        # Assert
        assert result == []

    def test_batch_files_handles_single_batch(self):
        """Test that batch_files handles files that fit in single batch."""
        # Arrange
        files = [Path("file1.py"), Path("file2.py")]
        batch_size = 5
        
        # Act
        result = batch_files(files, batch_size)
        
        # Assert
        assert len(result) == 1
        assert result[0] == files

    def test_batch_files_handles_exact_division(self):
        """Test that batch_files handles exact division into batches."""
        # Arrange
        files = [Path(f"file{i}.py") for i in range(6)]
        batch_size = 3
        
        # Act
        result = batch_files(files, batch_size)
        
        # Assert
        assert len(result) == 2
        assert len(result[0]) == 3
        assert len(result[1]) == 3


class TestShouldExcludeFile:
    """Test should_exclude_file function."""

    def test_should_exclude_file_matches_pattern(self):
        """Test that should_exclude_file matches exclusion patterns."""
        # Arrange
        file_path = Path("/project/tests/test_file.py")
        exclude_patterns = ["*/tests/*", "*.pyc"]
        
        # Act
        result = should_exclude_file(file_path, exclude_patterns)
        
        # Assert
        assert result is True

    def test_should_exclude_file_no_match(self):
        """Test that should_exclude_file returns False when no pattern matches."""
        # Arrange
        file_path = Path("/project/src/main.py")
        exclude_patterns = ["*/tests/*", "*.pyc"]
        
        # Act
        result = should_exclude_file(file_path, exclude_patterns)
        
        # Assert
        assert result is False

    def test_should_exclude_file_empty_patterns(self):
        """Test that should_exclude_file handles empty pattern list."""
        # Arrange
        file_path = Path("/project/src/main.py")
        exclude_patterns = []
        
        # Act
        result = should_exclude_file(file_path, exclude_patterns)
        
        # Assert
        assert result is False

    def test_should_exclude_file_multiple_patterns(self):
        """Test that should_exclude_file works with multiple patterns."""
        # Arrange
        file_path = Path("/project/build/output.pyc")
        exclude_patterns = ["*/tests/*", "*.pyc", "*/build/*"]
        
        # Act
        result = should_exclude_file(file_path, exclude_patterns)
        
        # Assert
        assert result is True

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from xml.etree import ElementTree as ET

from smart_test_generator.utils.parser import PythonCodebaseParser
from smart_test_generator.models.data_models import FileInfo, TestGenerationPlan
from smart_test_generator.config import Config


class TestPythonCodebaseParser:
    """Test suite for PythonCodebaseParser class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config object."""
        config = Mock(spec=Config)
        config.get.return_value = []
        return config

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_python_file(self, temp_dir):
        """Create a sample Python file for testing."""
        file_path = temp_dir / "sample.py"
        file_path.write_text("def hello():\n    return 'world'")
        return file_path

    def test_init_with_minimal_parameters(self, temp_dir, mock_config):
        """Test PythonCodebaseParser initialization with minimal parameters."""
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            
            assert parser.root_dir == temp_dir.resolve()
            assert parser.config == mock_config
            assert isinstance(parser.exclude_dirs, list)
            assert '__pycache__' in parser.exclude_dirs
            assert '.git' in parser.exclude_dirs

    def test_init_with_custom_exclude_dirs(self, temp_dir, mock_config):
        """Test PythonCodebaseParser initialization with custom exclude directories."""
        custom_excludes = ['custom_dir', 'another_exclude']
        
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config, custom_excludes)
            
            assert 'custom_dir' in parser.exclude_dirs
            assert 'another_exclude' in parser.exclude_dirs
            assert '__pycache__' in parser.exclude_dirs  # Default excludes still present

    def test_init_with_config_excludes(self, temp_dir, mock_config):
        """Test PythonCodebaseParser initialization with config-based excludes."""
        mock_config.get.return_value = ['config_exclude', 'another_config_exclude']
        
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            
            assert 'config_exclude' in parser.exclude_dirs
            assert 'another_config_exclude' in parser.exclude_dirs
            mock_config.get.assert_called_once_with('test_generation.exclude_dirs', [])

    def test_init_initializes_components(self, temp_dir, mock_config):
        """Test that initialization creates required components."""
        with patch('smart_test_generator.utils.parser.TestGenerationTracker') as mock_tracker, \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer') as mock_coverage, \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator') as mock_generator:
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            
            mock_tracker.assert_called_once()
            mock_coverage.assert_called_once_with(temp_dir.resolve(), mock_config)
            mock_generator.assert_called_once_with(temp_dir.resolve(), mock_config)
            assert parser.tracker is not None
            assert parser.coverage_analyzer is not None
            assert parser.incremental_generator is not None

    def test_find_python_files_empty_directory(self, temp_dir, mock_config):
        """Test finding Python files in an empty directory."""
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            files = parser.find_python_files()
            
            assert files == []

    def test_find_python_files_with_python_files(self, temp_dir, mock_config):
        """Test finding Python files in directory with Python files."""
        # Create test files
        (temp_dir / "module1.py").write_text("def func1(): pass")
        (temp_dir / "module2.py").write_text("def func2(): pass")
        (temp_dir / "not_python.txt").write_text("not python")
        
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'), \
             patch('smart_test_generator.utils.file_utils.FileUtils.is_test_file', return_value=False):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            files = parser.find_python_files()
            
            assert len(files) == 2
            assert any("module1.py" in f for f in files)
            assert any("module2.py" in f for f in files)

    def test_find_python_files_excludes_test_files(self, temp_dir, mock_config):
        """Test that test files are excluded from Python file search."""
        (temp_dir / "module.py").write_text("def func(): pass")
        (temp_dir / "test_module.py").write_text("def test_func(): pass")
        
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'), \
             patch('smart_test_generator.utils.file_utils.FileUtils.is_test_file') as mock_is_test:
            
            def is_test_side_effect(path):
                return "test_" in str(path)
            mock_is_test.side_effect = is_test_side_effect
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            files = parser.find_python_files()
            
            assert len(files) == 1
            assert "module.py" in files[0]
            assert not any("test_module.py" in f for f in files)

    def test_find_python_files_excludes_config_files(self, temp_dir, mock_config):
        """Test that config files are excluded from Python file search."""
        (temp_dir / "module.py").write_text("def func(): pass")
        (temp_dir / "config.py").write_text("CONFIG = {}")
        (temp_dir / "setup.py").write_text("from setuptools import setup")
        
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'), \
             patch('smart_test_generator.utils.file_utils.FileUtils.is_test_file', return_value=False):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            files = parser.find_python_files()
            
            assert len(files) == 1
            assert "module.py" in files[0]
            assert not any("config.py" in f for f in files)
            assert not any("setup.py" in f for f in files)

    def test_find_python_files_excludes_directories(self, temp_dir, mock_config):
        """Test that excluded directories are not traversed."""
        # Create files in excluded directory
        excluded_dir = temp_dir / "__pycache__"
        excluded_dir.mkdir()
        (excluded_dir / "cached.py").write_text("cached code")
        
        # Create file in main directory
        (temp_dir / "module.py").write_text("def func(): pass")
        
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'), \
             patch('smart_test_generator.utils.file_utils.FileUtils.is_test_file', return_value=False):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            files = parser.find_python_files()
            
            assert len(files) == 1
            assert "module.py" in files[0]
            assert not any("cached.py" in f for f in files)

    def test_generate_directory_structure_empty_directory(self, temp_dir, mock_config):
        """Test generating directory structure for empty directory."""
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            structure = parser.generate_directory_structure()
            
            # For empty directory, structure might be empty or contain just root
            assert isinstance(structure, str)

    def test_generate_directory_structure_with_files(self, temp_dir, mock_config):
        """Test generating directory structure with files and directories."""
        # Create test structure
        (temp_dir / "file1.py").write_text("content")
        (temp_dir / "file2.py").write_text("content")
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "subfile.py").write_text("content")
        
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            structure = parser.generate_directory_structure()
            
            assert "file1.py" in structure
            assert "file2.py" in structure
            assert "subdir/" in structure
            assert "subfile.py" in structure
            assert "└──" in structure or "├──" in structure  # Tree characters

    def test_generate_directory_structure_excludes_directories(self, temp_dir, mock_config):
        """Test that directory structure excludes configured directories."""
        # Create excluded directory
        excluded_dir = temp_dir / "__pycache__"
        excluded_dir.mkdir()
        (excluded_dir / "cached.py").write_text("content")
        
        # Create normal file
        (temp_dir / "normal.py").write_text("content")
        
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            structure = parser.generate_directory_structure()
            
            assert "normal.py" in structure
            assert "__pycache__" not in structure
            assert "cached.py" not in structure

    def test_print_directory_info_basic(self, temp_dir, mock_config, caplog):
        """Test printing directory info with basic file lists."""
        all_files = [str(temp_dir / "file1.py"), str(temp_dir / "file2.py")]
        files_to_process = [str(temp_dir / "file1.py")]
        
        # Create actual files for size calculation
        for file_path in all_files:
            Path(file_path).write_text("test content")
        
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            
            with caplog.at_level('INFO'):
                parser.print_directory_info(all_files, files_to_process)
            
            log_text = caplog.text
            assert "DIRECTORY STRUCTURE:" in log_text
            assert "FILE ANALYSIS:" in log_text
            assert "2 total files" in log_text
            assert "1 to process" in log_text
            assert "✓ TO PROCESS" in log_text
            assert "skip" in log_text

    def test_print_directory_info_with_test_plans(self, temp_dir, mock_config, caplog):
        """Test printing directory info with test generation plans."""
        all_files = [str(temp_dir / "file1.py")]
        files_to_process = [str(temp_dir / "file1.py")]
        
        # Create file
        Path(all_files[0]).write_text("test content")
        
        # Create mock test plan
        mock_element = Mock()
        mock_element.type = "function"
        mock_element.name = "test_func"
        mock_element.line_number = 10
        
        mock_coverage = Mock()
        mock_coverage.line_coverage = 75.0
        
        test_plan = TestGenerationPlan(
            source_file=all_files[0],
            existing_test_files=[],
            elements_to_test=[mock_element],
            coverage_before=mock_coverage,
            estimated_coverage_after=85.0
        )
        
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            
            with caplog.at_level('INFO'):
                parser.print_directory_info(all_files, files_to_process, [test_plan])
            
            log_text = caplog.text
            assert "TEST GENERATION PLAN:" in log_text
            assert "Total elements to test: 1" in log_text
            assert "function: test_func" in log_text
            assert "Current coverage: 75.0%" in log_text
            assert "Estimated after: 85.0%" in log_text

    def test_parse_file_success(self, temp_dir, mock_config):
        """Test successful file parsing."""
        file_path = temp_dir / "test_file.py"
        file_content = "def hello():\n    return 'world'"
        file_path.write_text(file_content)
        
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            file_info = parser.parse_file(str(file_path))
            
            assert isinstance(file_info, FileInfo)
            assert file_info.filename == "test_file.py"
            assert file_info.filepath.endswith("test_file.py")
            assert file_info.content == file_content

    def test_parse_file_with_subdirectory(self, temp_dir, mock_config):
        """Test parsing file in subdirectory."""
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        file_path = subdir / "test_file.py"
        file_content = "def hello(): pass"
        file_path.write_text(file_content)
        
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            file_info = parser.parse_file(str(file_path))
            
            assert file_info.filename == "test_file.py"
            assert file_info.filepath.endswith("subdir/test_file.py") or file_info.filepath.endswith("subdir\\test_file.py")
            assert file_info.content == file_content

    def test_parse_file_unicode_content(self, temp_dir, mock_config):
        """Test parsing file with unicode content."""
        file_path = temp_dir / "unicode_file.py"
        file_content = "# -*- coding: utf-8 -*-\ndef greet():\n    return '你好世界'"
        file_path.write_text(file_content, encoding='utf-8')
        
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            file_info = parser.parse_file(str(file_path))
            
            assert "你好世界" in file_info.content
            assert file_info.filename == "unicode_file.py"

    def test_generate_xml_content_empty_list(self, temp_dir, mock_config):
        """Test generating XML content with empty file list."""
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            xml_content = parser.generate_xml_content([])
            
            assert "<?xml version=" in xml_content
            assert "<codebase" in xml_content
            assert "</codebase>" in xml_content

    def test_generate_xml_content_single_file(self, temp_dir, mock_config):
        """Test generating XML content with single file."""
        file_info = FileInfo(
            filepath="test.py",
            filename="test.py",
            content="def hello(): pass"
        )
        
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            xml_content = parser.generate_xml_content([file_info])
            
            assert "<?xml version=" in xml_content
            assert "<codebase" in xml_content
            assert 'filename="test.py"' in xml_content
            assert 'filepath="test.py"' in xml_content
            assert "def hello(): pass" in xml_content
            assert "</codebase>" in xml_content

    def test_generate_xml_content_multiple_files(self, temp_dir, mock_config):
        """Test generating XML content with multiple files."""
        file_info1 = FileInfo(
            filepath="file1.py",
            filename="file1.py",
            content="def func1(): pass"
        )
        file_info2 = FileInfo(
            filepath="dir/file2.py",
            filename="file2.py",
            content="def func2(): pass"
        )
        
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            xml_content = parser.generate_xml_content([file_info1, file_info2])
            
            assert xml_content.count('<file') == 2
            assert 'filename="file1.py"' in xml_content
            assert 'filename="file2.py"' in xml_content
            assert 'filepath="file1.py"' in xml_content
            assert 'filepath="dir/file2.py"' in xml_content
            assert "def func1(): pass" in xml_content
            assert "def func2(): pass" in xml_content

    def test_generate_xml_content_special_characters(self, temp_dir, mock_config):
        """Test generating XML content with special characters."""
        file_info = FileInfo(
            filepath="special.py",
            filename="special.py",
            content="def func():\n    return '<>&\"test\"'"
        )
        
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            xml_content = parser.generate_xml_content([file_info])
            
            # XML should be properly escaped
            assert "<?xml version=" in xml_content
            assert "<codebase" in xml_content
            # Content should be present (exact escaping depends on implementation)
            assert "func()" in xml_content

    def test_generate_xml_content_valid_xml_structure(self, temp_dir, mock_config):
        """Test that generated XML content is valid XML."""
        file_info = FileInfo(
            filepath="test.py",
            filename="test.py",
            content="def hello(): pass"
        )
        
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            xml_content = parser.generate_xml_content([file_info])
            
            # Should be parseable as XML
            try:
                ET.fromstring(xml_content)
            except ET.ParseError:
                pytest.fail("Generated XML content is not valid XML")

    def test_parse_file_file_not_found(self, temp_dir, mock_config):
        """Test parsing non-existent file raises appropriate error."""
        non_existent_file = str(temp_dir / "non_existent.py")
        
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            
            with pytest.raises(FileNotFoundError):
                parser.parse_file(non_existent_file)

    def test_find_python_files_permission_error_handling(self, temp_dir, mock_config):
        """Test handling of permission errors during file search."""
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'), \
             patch('os.walk') as mock_walk:
            
            # Simulate permission error
            mock_walk.side_effect = PermissionError("Access denied")
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            
            with pytest.raises(PermissionError):
                parser.find_python_files()

    def test_print_directory_info_handles_large_test_plans(self, temp_dir, mock_config, caplog):
        """Test that print_directory_info handles large numbers of test elements properly."""
        all_files = [str(temp_dir / "file1.py")]
        files_to_process = [str(temp_dir / "file1.py")]
        
        # Create file
        Path(all_files[0]).write_text("test content")
        
        # Create many mock elements
        mock_elements = []
        for i in range(10):
            mock_element = Mock()
            mock_element.type = "function"
            mock_element.name = f"test_func_{i}"
            mock_element.line_number = i + 10
            mock_elements.append(mock_element)
        
        test_plan = TestGenerationPlan(
            source_file=all_files[0],
            existing_test_files=[],
            elements_to_test=mock_elements,
            coverage_before=None,
            estimated_coverage_after=85.0
        )
        
        with patch('smart_test_generator.utils.parser.TestGenerationTracker'), \
             patch('smart_test_generator.utils.parser.CoverageAnalyzer'), \
             patch('smart_test_generator.utils.parser.IncrementalTestGenerator'):
            
            parser = PythonCodebaseParser(str(temp_dir), mock_config)
            
            with caplog.at_level('INFO'):
                parser.print_directory_info(all_files, files_to_process, [test_plan])
            
            log_text = caplog.text
            assert "Total elements to test: 10" in log_text
            assert "... and 5 more" in log_text  # Should truncate after 5
            assert "test_func_0" in log_text
            assert "test_func_4" in log_text
            assert "test_func_9" not in log_text  # Should be truncated
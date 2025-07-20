import pytest
import ast
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open

from smart_test_generator.analysis.coverage_analyzer import ASTCoverageAnalyzer, CoverageAnalyzer
from smart_test_generator.models.data_models import TestCoverage
from smart_test_generator.config import Config


class TestASTCoverageAnalyzer:
    """Test ASTCoverageAnalyzer class."""
    
    def test_init_sets_project_root(self):
        """Test that __init__ correctly sets project_root."""
        # Arrange
        project_root = Path("/test/project")
        
        # Act
        analyzer = ASTCoverageAnalyzer(project_root)
        
        # Assert
        assert analyzer.project_root == project_root
    
    def test_init_with_pathlib_path(self):
        """Test that __init__ works with pathlib.Path objects."""
        # Arrange
        project_root = Path("/test/project/path")
        
        # Act
        analyzer = ASTCoverageAnalyzer(project_root)
        
        # Assert
        assert analyzer.project_root == project_root
        assert isinstance(analyzer.project_root, Path)
    
    @patch('builtins.open', new_callable=mock_open, read_data="def test_function():\n    return 42\n")
    @patch('ast.parse')
    def test_analyze_file_coverage_success(self, mock_parse, mock_file):
        """Test successful file coverage analysis."""
        # Arrange
        project_root = Path("/test/project")
        analyzer = ASTCoverageAnalyzer(project_root)
        filepath = "/test/file.py"
        
        # Mock AST tree
        mock_tree = Mock()
        mock_parse.return_value = mock_tree
        
        # Mock analyzer methods
        analyzer._get_executable_lines = Mock(return_value={1, 2, 3})
        analyzer._get_all_functions_and_methods = Mock(return_value={'test_function'})
        analyzer._estimate_coverage = Mock(return_value=({1, 2}, {'test_function'}))
        analyzer._estimate_branch_coverage = Mock(return_value=75.0)
        
        # Act
        result = analyzer.analyze_file_coverage(filepath)
        
        # Assert
        assert isinstance(result, TestCoverage)
        assert result.filepath == filepath
        assert result.line_coverage == pytest.approx(66.67, rel=1e-2)  # 2/3 * 100
        assert result.branch_coverage == 75.0
        assert result.missing_lines == [3]
        assert result.covered_functions == {'test_function'}
        assert result.uncovered_functions == set()
        
        mock_file.assert_called_once_with(filepath, 'r', encoding='utf-8')
        mock_parse.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open, read_data="def test_function():\n    return 42\n")
    @patch('ast.parse')
    def test_analyze_file_coverage_with_test_files(self, mock_parse, mock_file):
        """Test file coverage analysis with test files provided."""
        # Arrange
        project_root = Path("/test/project")
        analyzer = ASTCoverageAnalyzer(project_root)
        filepath = "/test/file.py"
        test_files = ["/test/test_file.py"]
        
        mock_tree = Mock()
        mock_parse.return_value = mock_tree
        
        analyzer._get_executable_lines = Mock(return_value={1, 2})
        analyzer._get_all_functions_and_methods = Mock(return_value={'test_function'})
        analyzer._estimate_coverage = Mock(return_value=({1}, {'test_function'}))
        analyzer._estimate_branch_coverage = Mock(return_value=50.0)
        
        # Act
        result = analyzer.analyze_file_coverage(filepath, test_files)
        
        # Assert
        assert isinstance(result, TestCoverage)
        assert result.filepath == filepath
        assert result.line_coverage == 50.0  # 1/2 * 100
        analyzer._estimate_coverage.assert_called_once_with(filepath, {1, 2}, {'test_function'}, test_files)
    
    @patch('builtins.open', side_effect=FileNotFoundError("File not found"))
    def test_analyze_file_coverage_file_not_found(self, mock_file):
        """Test file coverage analysis when file doesn't exist."""
        # Arrange
        project_root = Path("/test/project")
        analyzer = ASTCoverageAnalyzer(project_root)
        filepath = "/nonexistent/file.py"
        
        analyzer._create_zero_coverage = Mock(return_value=TestCoverage(
            filepath=filepath,
            line_coverage=0.0,
            branch_coverage=0.0,
            missing_lines=[],
            covered_functions=set(),
            uncovered_functions=set()
        ))
        
        # Act
        result = analyzer.analyze_file_coverage(filepath)
        
        # Assert
        assert isinstance(result, TestCoverage)
        assert result.line_coverage == 0.0
        analyzer._create_zero_coverage.assert_called_once_with(filepath)
    
    @patch('builtins.open', new_callable=mock_open, read_data="invalid python code {{{")
    @patch('ast.parse', side_effect=SyntaxError("Invalid syntax"))
    def test_analyze_file_coverage_syntax_error(self, mock_parse, mock_file):
        """Test file coverage analysis with syntax error in file."""
        # Arrange
        project_root = Path("/test/project")
        analyzer = ASTCoverageAnalyzer(project_root)
        filepath = "/test/invalid.py"
        
        analyzer._create_zero_coverage = Mock(return_value=TestCoverage(
            filepath=filepath,
            line_coverage=0.0,
            branch_coverage=0.0,
            missing_lines=[],
            covered_functions=set(),
            uncovered_functions=set()
        ))
        
        # Act
        result = analyzer.analyze_file_coverage(filepath)
        
        # Assert
        assert isinstance(result, TestCoverage)
        assert result.line_coverage == 0.0
        analyzer._create_zero_coverage.assert_called_once_with(filepath)
    
    @patch('builtins.open', new_callable=mock_open, read_data="def func():\n    pass\n")
    @patch('ast.parse')
    def test_analyze_file_coverage_zero_executable_lines(self, mock_parse, mock_file):
        """Test file coverage analysis with zero executable lines."""
        # Arrange
        project_root = Path("/test/project")
        analyzer = ASTCoverageAnalyzer(project_root)
        filepath = "/test/empty.py"
        
        mock_tree = Mock()
        mock_parse.return_value = mock_tree
        
        analyzer._get_executable_lines = Mock(return_value=set())
        analyzer._get_all_functions_and_methods = Mock(return_value=set())
        analyzer._estimate_coverage = Mock(return_value=(set(), set()))
        analyzer._estimate_branch_coverage = Mock(return_value=0.0)
        
        # Act
        result = analyzer.analyze_file_coverage(filepath)
        
        # Assert
        assert isinstance(result, TestCoverage)
        assert result.line_coverage == 0.0
        assert result.missing_lines == []


class TestCoverageAnalyzer:
    """Test CoverageAnalyzer class."""
    
    def test_init_sets_attributes(self):
        """Test that __init__ correctly sets all attributes."""
        # Arrange
        project_root = Path("/test/project")
        config = Mock(spec=Config)
        
        # Act
        analyzer = CoverageAnalyzer(project_root, config)
        
        # Assert
        assert analyzer.project_root == project_root
        assert analyzer.config == config
        assert analyzer.coverage_data == {}
        assert isinstance(analyzer.ast_analyzer, ASTCoverageAnalyzer)
        assert analyzer.ast_analyzer.project_root == project_root
    
    def test_init_with_different_config(self):
        """Test __init__ with different config object."""
        # Arrange
        project_root = Path("/different/project")
        config = Mock(spec=Config)
        config.some_setting = "test_value"
        
        # Act
        analyzer = CoverageAnalyzer(project_root, config)
        
        # Assert
        assert analyzer.project_root == project_root
        assert analyzer.config == config
        assert analyzer.config.some_setting == "test_value"
    
    @patch('smart_test_generator.analysis.coverage_analyzer.logger')
    def test_run_coverage_analysis_pytest_success(self, mock_logger):
        """Test successful pytest coverage analysis."""
        # Arrange
        project_root = Path("/test/project")
        config = Mock(spec=Config)
        analyzer = CoverageAnalyzer(project_root, config)
        source_files = ["/test/file1.py", "/test/file2.py"]
        
        # Mock successful pytest coverage
        mock_coverage_map = {
            "/test/file1.py": TestCoverage(
                filepath="/test/file1.py",
                line_coverage=80.0,
                branch_coverage=70.0,
                missing_lines=[5, 10],
                covered_functions={'func1'},
                uncovered_functions={'func2'}
            )
        }
        
        analyzer._run_pytest_coverage = Mock(return_value=mock_coverage_map)
        
        # Act
        result = analyzer.run_coverage_analysis(source_files)
        
        # Assert
        assert result == mock_coverage_map
        analyzer._run_pytest_coverage.assert_called_once_with(source_files)
        mock_logger.info.assert_any_call("Attempting pytest coverage analysis...")
        mock_logger.info.assert_any_call("Pytest coverage analysis successful")
    
    @patch('smart_test_generator.analysis.coverage_analyzer.logger')
    def test_run_coverage_analysis_pytest_empty_results_fallback(self, mock_logger):
        """Test fallback to AST analysis when pytest returns empty results."""
        # Arrange
        project_root = Path("/test/project")
        config = Mock(spec=Config)
        analyzer = CoverageAnalyzer(project_root, config)
        source_files = ["/test/file1.py"]
        test_files = ["/test/test_file1.py"]
        
        # Mock empty pytest coverage results
        empty_coverage_map = {
            "/test/file1.py": TestCoverage(
                filepath="/test/file1.py",
                line_coverage=0.0,
                branch_coverage=0.0,
                missing_lines=[],
                covered_functions=set(),
                uncovered_functions=set()
            )
        }
        
        # Mock AST fallback results
        ast_coverage_map = {
            "/test/file1.py": TestCoverage(
                filepath="/test/file1.py",
                line_coverage=60.0,
                branch_coverage=50.0,
                missing_lines=[3, 7],
                covered_functions={'func1'},
                uncovered_functions={'func2'}
            )
        }
        
        analyzer._run_pytest_coverage = Mock(return_value=empty_coverage_map)
        analyzer._run_ast_coverage_fallback = Mock(return_value=ast_coverage_map)
        
        # Act
        result = analyzer.run_coverage_analysis(source_files, test_files)
        
        # Assert
        assert result == ast_coverage_map
        analyzer._run_pytest_coverage.assert_called_once_with(source_files)
        analyzer._run_ast_coverage_fallback.assert_called_once_with(source_files, test_files)
        mock_logger.info.assert_any_call("Pytest coverage analysis returned empty results, falling back to AST analysis")
    
    @patch('smart_test_generator.analysis.coverage_analyzer.logger')
    def test_run_coverage_analysis_pytest_exception_fallback(self, mock_logger):
        """Test fallback to AST analysis when pytest raises exception."""
        # Arrange
        project_root = Path("/test/project")
        config = Mock(spec=Config)
        analyzer = CoverageAnalyzer(project_root, config)
        source_files = ["/test/file1.py"]
        
        # Mock pytest exception
        analyzer._run_pytest_coverage = Mock(side_effect=Exception("Pytest failed"))
        
        # Mock AST fallback results
        ast_coverage_map = {
            "/test/file1.py": TestCoverage(
                filepath="/test/file1.py",
                line_coverage=40.0,
                branch_coverage=30.0,
                missing_lines=[1, 2, 3],
                covered_functions=set(),
                uncovered_functions={'func1'}
            )
        }
        
        analyzer._run_ast_coverage_fallback = Mock(return_value=ast_coverage_map)
        
        # Act
        result = analyzer.run_coverage_analysis(source_files)
        
        # Assert
        assert result == ast_coverage_map
        analyzer._run_pytest_coverage.assert_called_once_with(source_files)
        analyzer._run_ast_coverage_fallback.assert_called_once_with(source_files, None)
        mock_logger.warning.assert_called_once_with("Pytest coverage analysis failed: Pytest failed")
        mock_logger.info.assert_any_call("Falling back to AST-based coverage analysis")
    
    def test_run_coverage_analysis_with_test_files(self):
        """Test coverage analysis with explicit test files provided."""
        # Arrange
        project_root = Path("/test/project")
        config = Mock(spec=Config)
        analyzer = CoverageAnalyzer(project_root, config)
        source_files = ["/test/file1.py"]
        test_files = ["/test/test_file1.py", "/test/test_file2.py"]
        
        # Mock successful pytest coverage
        mock_coverage_map = {
            "/test/file1.py": TestCoverage(
                filepath="/test/file1.py",
                line_coverage=90.0,
                branch_coverage=85.0,
                missing_lines=[],
                covered_functions={'func1', 'func2'},
                uncovered_functions=set()
            )
        }
        
        analyzer._run_pytest_coverage = Mock(return_value=mock_coverage_map)
        
        # Act
        result = analyzer.run_coverage_analysis(source_files, test_files)
        
        # Assert
        assert result == mock_coverage_map
        analyzer._run_pytest_coverage.assert_called_once_with(source_files)
    
    def test_run_coverage_analysis_empty_source_files(self):
        """Test coverage analysis with empty source files list."""
        # Arrange
        project_root = Path("/test/project")
        config = Mock(spec=Config)
        analyzer = CoverageAnalyzer(project_root, config)
        source_files = []
        
        analyzer._run_pytest_coverage = Mock(return_value={})
        
        # Act
        result = analyzer.run_coverage_analysis(source_files)
        
        # Assert
        assert result == {}
        analyzer._run_pytest_coverage.assert_called_once_with(source_files)
    
    @patch('smart_test_generator.analysis.coverage_analyzer.logger')
    def test_run_coverage_analysis_ast_fallback_exception(self, mock_logger):
        """Test behavior when both pytest and AST analysis fail."""
        # Arrange
        project_root = Path("/test/project")
        config = Mock(spec=Config)
        analyzer = CoverageAnalyzer(project_root, config)
        source_files = ["/test/file1.py"]
        
        # Mock both methods failing
        analyzer._run_pytest_coverage = Mock(side_effect=Exception("Pytest failed"))
        analyzer._run_ast_coverage_fallback = Mock(side_effect=Exception("AST failed"))
        
        # Act & Assert
        with pytest.raises(Exception, match="AST failed"):
            analyzer.run_coverage_analysis(source_files)
        
        analyzer._run_pytest_coverage.assert_called_once_with(source_files)
        analyzer._run_ast_coverage_fallback.assert_called_once_with(source_files, None)
        mock_logger.warning.assert_called_once_with("Pytest coverage analysis failed: Pytest failed")

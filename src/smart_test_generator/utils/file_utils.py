"""File system utilities for smart test generator."""

import os
import logging
from pathlib import Path
from typing import List, Optional, Set, Tuple
import fnmatch
import ast

from smart_test_generator.exceptions import FileOperationError

logger = logging.getLogger(__name__)


class FileUtils:
    """Utility functions for file system operations."""

    # Comprehensive exclusion patterns for virtual environments and build directories
    DEFAULT_EXCLUDE_DIRS = [
        # Virtual environments
        'venv', 'env', '.env', '.venv', 'virtualenv',
        'ENV', 'env.bak', 'venv.bak',
        # Python build directories
        'build', 'dist', '__pycache__',
        'site-packages', 'lib', 'lib64', 'include',
        'bin', 'Scripts', 'share',
        # Cache directories
        '.pytest_cache', '.coverage', '.cache',
        '.mypy_cache', '.ruff_cache', '.tox', '.nox',
        '.hypothesis',
        # IDE and editor directories
        '.vscode', '.idea', '.vs', '.atom',
        # Version control
        '.git', '.hg', '.svn', '.bzr',
        # Package managers
        'node_modules', 'bower_components',
        # Documentation build
        'docs/_build', '_build', 'site',
        # Testing and CI
        'htmlcov', '.stestr', '.testrepository',
        # Jupyter
        '.ipynb_checkpoints',
        # Temporary directories
        'tmp', 'temp', '.tmp', '.temp'
    ]

    @staticmethod
    def _is_excluded_directory(dir_path: Path, exclude_dirs: Optional[List[str]] = None) -> bool:
        """Check if a directory should be excluded."""
        exclude_dirs = exclude_dirs or FileUtils.DEFAULT_EXCLUDE_DIRS
        
        dir_name = dir_path.name
        
        # Check exact matches first (most common case)
        if dir_name in exclude_dirs:
            return True
            
        # Check glob patterns
        for pattern in exclude_dirs:
            if '*' in pattern or '?' in pattern:
                if fnmatch.fnmatch(dir_name, pattern):
                    return True
                    
        # Check for virtual environment indicators
        return FileUtils._is_virtual_environment(dir_path)

    @staticmethod
    def _is_virtual_environment(dir_path: Path) -> bool:
        """Check if directory is likely a virtual environment."""
        # Check for virtual environment indicators
        venv_indicators = [
            'pyvenv.cfg',  # Standard venv indicator
            'activate',    # activation script
            'pip',         # pip executable
            'python',      # python executable
            'site-packages'  # packages directory
        ]
        
        # For directories with common venv names, check for indicators
        venv_names = ['venv', 'env', '.venv', '.env', 'virtualenv', 'ENV']
        if dir_path.name in venv_names:
            # Check if it has typical venv structure
            try:
                for indicator in venv_indicators:
                    if any(dir_path.rglob(indicator)):
                        return True
            except (PermissionError, OSError):
                # If we can't access it, assume it's a venv to be safe
                return True
                
        return False

    @staticmethod
    def find_files_by_pattern(root_dir: Path, patterns: List[str],
                             exclude_dirs: Optional[List[str]] = None) -> List[Path]:
        """
        Find files matching given patterns.

        Args:
            root_dir: Root directory to search
            patterns: List of file patterns (e.g., ['*.py', 'test_*.py'])
            exclude_dirs: Directories to exclude from search

        Returns:
            List of matching file paths
        """
        exclude_dirs = exclude_dirs or FileUtils.DEFAULT_EXCLUDE_DIRS
        matching_files = []

        for root, dirs, files in os.walk(root_dir):
            # Filter out excluded directories
            root_path = Path(root)
            
            # Remove excluded directories from dirs to prevent walking into them
            dirs_to_remove = []
            for d in dirs:
                dir_path = root_path / d
                if FileUtils._is_excluded_directory(dir_path, exclude_dirs):
                    dirs_to_remove.append(d)
                    logger.debug(f"Excluding directory: {dir_path}")
            
            for d in dirs_to_remove:
                dirs.remove(d)

            for pattern in patterns:
                for filename in fnmatch.filter(files, pattern):
                    matching_files.append(Path(root) / filename)

        return matching_files

    @staticmethod
    def get_relative_path(file_path: Path, base_path: Path) -> str:
        """
        Get relative path from base path.

        Args:
            file_path: File path to convert
            base_path: Base path for relative conversion

        Returns:
            Relative path as string
        """
        try:
            return str(file_path.relative_to(base_path))
        except ValueError:
            # If paths don't have common base, return absolute path
            return str(file_path)

    @staticmethod
    def ensure_directory_exists(directory: Path) -> None:
        """
        Ensure directory exists, create if necessary.

        Args:
            directory: Directory path

        Raises:
            FileOperationError: If directory cannot be created
        """
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise FileOperationError(
                f"Permission denied when creating directory: {directory}",
                filepath=str(directory),
                suggestion="Check directory permissions or choose a different location."
            )
        except OSError as e:
            raise FileOperationError(
                f"Failed to create directory: {e}",
                filepath=str(directory),
                suggestion="Check that the path is valid and the parent directory exists."
            )
        except Exception as e:
            raise FileOperationError(
                f"Unexpected error creating directory: {e}",
                filepath=str(directory),
                suggestion="Check the path and try again."
            )

    @staticmethod
    def read_file_safely(file_path: Path, encoding: str = 'utf-8') -> str:
        """
        Safely read file contents.

        Args:
            file_path: Path to file
            encoding: File encoding

        Returns:
            File contents

        Raises:
            FileOperationError: If file cannot be read
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except FileNotFoundError:
            raise FileOperationError(
                f"File not found: {file_path}",
                filepath=str(file_path),
                suggestion="Check that the file exists and the path is correct."
            )
        except PermissionError:
            raise FileOperationError(
                f"Permission denied reading file: {file_path}",
                filepath=str(file_path),
                suggestion="Check file permissions or run with appropriate privileges."
            )
        except UnicodeDecodeError as e:
            raise FileOperationError(
                f"File encoding error: {e}",
                filepath=str(file_path),
                suggestion=f"Try a different encoding or check if the file is a binary file."
            )
        except Exception as e:
            raise FileOperationError(
                f"Failed to read file: {e}",
                filepath=str(file_path),
                suggestion="Check the file and try again."
            )

    @staticmethod
    def write_file_safely(file_path: Path, content: str, encoding: str = 'utf-8') -> None:
        """
        Safely write content to file.

        Args:
            file_path: Path to file
            content: Content to write
            encoding: File encoding

        Raises:
            FileOperationError: If file cannot be written
        """
        try:
            # Ensure parent directory exists
            FileUtils.ensure_directory_exists(file_path.parent)

            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
        except PermissionError:
            raise FileOperationError(
                f"Permission denied writing file: {file_path}",
                filepath=str(file_path),
                suggestion="Check file and directory permissions or run with appropriate privileges."
            )
        except OSError as e:
            raise FileOperationError(
                f"Failed to write file: {e}",
                filepath=str(file_path),
                suggestion="Check available disk space and file path validity."
            )
        except Exception as e:
            raise FileOperationError(
                f"Unexpected error writing file: {e}",
                filepath=str(file_path),
                suggestion="Check the file path and try again."
            )

    @staticmethod
    def get_file_size(file_path: Path) -> int:
        """
        Get file size in bytes.

        Args:
            file_path: Path to file

        Returns:
            File size in bytes, or 0 if error
        """
        try:
            return file_path.stat().st_size
        except Exception:
            return 0

    @staticmethod
    def is_python_file(file_path: Path) -> bool:
        """
        Check if file is a Python file.

        Args:
            file_path: Path to check

        Returns:
            True if Python file
        """
        return file_path.suffix == '.py' and file_path.is_file()

    @staticmethod
    def is_test_file(file_path: Path) -> bool:
        """
        Check if file is a test file based on common patterns.

        Args:
            file_path: Path to check

        Returns:
            True if likely a test file
        """
        name = file_path.name.lower()
        
        # Check if file is in a test directory
        test_dir_names = {'test', 'tests', 'testing'}
        is_in_test_dir = any(part.lower() in test_dir_names for part in file_path.parts[:-1])
        
        # If file is in a test directory, it's probably a test file
        if is_in_test_dir:
            return True
        
        # For files outside test directories, be very conservative
        # Only exclude files with clear test patterns
        is_test_filename = (
            name.endswith('_test.py') or  # module_test.py pattern
            name == 'conftest.py' or     # pytest configuration file
            (name.startswith('test_') and FileUtils._is_simple_test_name(name))
        )
        
        return is_test_filename
    
    @staticmethod
    def _is_simple_test_name(filename: str) -> bool:
        """
        Check if a filename starting with 'test_' follows simple test naming patterns.
        
        Args:
            filename: The filename to check (should be lowercase)
            
        Returns:
            True if it's a simple test filename like test_main.py, test_utils.py
        """
        # Remove .py extension for analysis
        name_without_ext = filename[:-3] if filename.endswith('.py') else filename
        
        if name_without_ext.startswith('test_'):
            remaining = name_without_ext[5:]  # Remove 'test_' prefix
            
            # Very conservative: only single words with no underscores and short length
            # test_main.py, test_utils.py, test_auth.py -> YES
            # test_data_processor.py, test_generator.py -> NO
            return '_' not in remaining and len(remaining) <= 6
        
        return False

    @staticmethod
    def is_test_file_ast(file_path: Path) -> bool:
        """
        Check if file is a test file using AST analysis.
        More accurate but slower than is_test_file().

        Args:
            file_path: Path to check

        Returns:
            True if file contains test functions
        """
        # First check basic patterns for performance
        if not FileUtils.is_test_file(file_path):
            # If it doesn't match naming patterns, check content
            return FileUtils._contains_test_functions(file_path)
        
        # If it matches patterns, verify with AST
        return FileUtils._contains_test_functions(file_path)

    @staticmethod
    def _contains_test_functions(file_path: Path) -> bool:
        """
        Check if a Python file contains test functions using AST.

        Args:
            file_path: Path to Python file

        Returns:
            True if file contains test functions
        """
        if not FileUtils.is_python_file(file_path):
            return False
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            # Look for test functions or classes
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if FileUtils._is_test_function_node(node):
                        return True
                elif isinstance(node, ast.ClassDef):
                    if FileUtils._is_test_class_node(node):
                        return True
                        
        except Exception as e:
            logger.debug(f"Failed to analyze {file_path} for test functions: {e}")
            
        return False

    @staticmethod
    def _is_test_function_node(node: ast.FunctionDef) -> bool:
        """
        Check if an AST function node is a test function.

        Args:
            node: AST function definition node

        Returns:
            True if it's a test function
        """
        # Check function name patterns
        if (node.name.startswith('test_') or 
            node.name.endswith('_test')):
            return True
            
        # Check for pytest decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                if decorator.id in ['pytest', 'test']:
                    return True
            elif isinstance(decorator, ast.Attribute):
                if (isinstance(decorator.value, ast.Name) and 
                    decorator.value.id == 'pytest'):
                    return True
                    
        # Check for unittest inheritance (would be in class context)
        return False

    @staticmethod
    def _is_test_class_node(node: ast.ClassDef) -> bool:
        """
        Check if an AST class node is a test class.

        Args:
            node: AST class definition node

        Returns:
            True if it's a test class
        """
        # Check class name patterns
        if (node.name.startswith('Test') or 
            node.name.endswith('Test') or 
            node.name.endswith('Tests')):
            return True
            
        # Check for unittest.TestCase inheritance
        for base in node.bases:
            if isinstance(base, ast.Attribute):
                if (isinstance(base.value, ast.Name) and 
                    base.value.id == 'unittest' and 
                    base.attr == 'TestCase'):
                    return True
            elif isinstance(base, ast.Name):
                if base.id in ['TestCase', 'AsyncTestCase']:
                    return True
                    
        # Check if class contains test methods
        for class_node in node.body:
            if (isinstance(class_node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                FileUtils._is_test_function_node(class_node)):
                return True
                
        return False

    @staticmethod
    def find_test_files_ast(root_dir: Path, exclude_dirs: Optional[List[str]] = None) -> List[Path]:
        """
        Find test files using AST analysis for accurate detection.

        Args:
            root_dir: Root directory to search
            exclude_dirs: Directories to exclude from search

        Returns:
            List of test file paths
        """
        exclude_dirs = exclude_dirs or FileUtils.DEFAULT_EXCLUDE_DIRS
        test_files = []

        for root, dirs, files in os.walk(root_dir):
            # Filter out excluded directories
            root_path = Path(root)
            
            # Remove excluded directories from dirs to prevent walking into them
            dirs_to_remove = []
            for d in dirs:
                dir_path = root_path / d
                if FileUtils._is_excluded_directory(dir_path, exclude_dirs):
                    dirs_to_remove.append(d)
            
            for d in dirs_to_remove:
                dirs.remove(d)

            # Check Python files for test content
            for filename in files:
                if filename.endswith('.py'):
                    file_path = Path(root) / filename
                    if FileUtils.is_test_file_ast(file_path):
                        test_files.append(file_path)

        return test_files

    @staticmethod
    def is_config_file(file_path: Path) -> bool:
        """
        Check if file is a configuration file.

        Args:
            file_path: Path to check

        Returns:
            True if likely a config file
        """
        config_patterns = [
            'config.py', 'conf.py', 'settings.py',
            'setup.py', '__init__.py', '.env'
        ]
        name = file_path.name.lower()
        return any(pattern in name for pattern in config_patterns)

    @staticmethod
    def filter_python_files(files: List[Path],
                           exclude_configs: bool = True,
                           exclude_tests: bool = False) -> List[Path]:
        """
        Filter list of files to Python files only.

        Args:
            files: List of file paths
            exclude_configs: Whether to exclude config files
            exclude_tests: Whether to exclude test files

        Returns:
            Filtered list of Python files
        """
        filtered = []

        for file_path in files:
            if not FileUtils.is_python_file(file_path):
                continue

            if exclude_configs and FileUtils.is_config_file(file_path):
                continue

            if exclude_tests and FileUtils.is_test_file(file_path):
                continue

            filtered.append(file_path)

        return filtered

    @staticmethod
    def group_files_by_directory(files: List[Path]) -> dict[str, List[Path]]:
        """
        Group files by their parent directory.

        Args:
            files: List of file paths

        Returns:
            Dictionary mapping directory to list of files
        """
        grouped = {}

        for file_path in files:
            dir_path = str(file_path.parent)
            if dir_path not in grouped:
                grouped[dir_path] = []
            grouped[dir_path].append(file_path)

        return grouped

    @staticmethod
    def find_matching_test_file(source_file: Path, test_dirs: List[Path]) -> Optional[Path]:
        """
        Find test file corresponding to a source file.

        Args:
            source_file: Source file path
            test_dirs: List of test directories to search

        Returns:
            Path to test file if found, None otherwise
        """
        module_name = source_file.stem
        test_patterns = [
            f"test_{module_name}.py",
            f"{module_name}_test.py",
        ]

        for test_dir in test_dirs:
            if not test_dir.exists():
                continue

            for pattern in test_patterns:
                test_file = test_dir / pattern
                if test_file.exists():
                    return test_file

        return None

    @staticmethod
    def calculate_test_file_path(source_file: Path, project_root: Path) -> Path:
        """
        Calculate where test file should be located for a source file.

        Args:
            source_file: Source file path
            project_root: Project root directory

        Returns:
            Calculated test file path
        """
        rel_path = source_file.relative_to(project_root)

        # Common patterns for test file locations
        if 'src' in rel_path.parts:
            # Replace 'src' with 'tests'
            parts = list(rel_path.parts)
            src_index = parts.index('src')
            parts[src_index] = 'tests'
            test_path = Path(*parts)
        else:
            # Put in tests directory
            test_path = Path('tests') / rel_path

        # Change filename to test_<filename>
        test_filename = f"test_{test_path.name}"
        return test_path.parent / test_filename

    @staticmethod
    def get_module_name_from_path(file_path: Path, project_root: Path) -> str:
        """
        Get Python module name from file path.

        Args:
            file_path: Python file path
            project_root: Project root directory

        Returns:
            Module name in dot notation
        """
        try:
            rel_path = file_path.relative_to(project_root)
            # Remove .py extension and convert path separators to dots
            module_parts = list(rel_path.parts[:-1]) + [rel_path.stem]
            return '.'.join(module_parts)
        except ValueError:
            return file_path.stem

    @staticmethod
    def estimate_file_complexity(file_path: Path) -> Tuple[int, int]:
        """
        Estimate file complexity based on lines of code and function count.

        Args:
            file_path: Python file path

        Returns:
            Tuple of (line_count, function_count)
        """
        try:
            content = FileUtils.read_file_safely(file_path)
            if not content:
                return 0, 0

            lines = content.split('\n')
            # Count non-empty, non-comment lines
            code_lines = sum(1 for line in lines
                           if line.strip() and not line.strip().startswith('#'))

            # Simple function count (not perfect but fast)
            function_count = content.count('\ndef ') + content.count('\nasync def ')

            return code_lines, function_count

        except Exception:
            return 0, 0


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 KB", "2.3 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def batch_files(files: List[Path], batch_size: int) -> List[List[Path]]:
    """
    Split files into batches.

    Args:
        files: List of files to batch
        batch_size: Maximum files per batch

    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        batches.append(batch)
    return batches


def should_exclude_file(file_path: Path, exclude_patterns: List[str]) -> bool:
    """
    Check if file should be excluded based on patterns.

    Args:
        file_path: File to check
        exclude_patterns: List of exclusion patterns

    Returns:
        True if file should be excluded
    """
    file_str = str(file_path)
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(file_str, pattern):
            return True
    return False

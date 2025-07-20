"""Python codebase parsing utilities."""

import os
import logging
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import List, Optional
import fnmatch

from smart_test_generator.models.data_models import FileInfo, TestGenerationPlan
from smart_test_generator.config import Config
from smart_test_generator.tracking.state_tracker import TestGenerationTracker
from smart_test_generator.analysis.coverage_analyzer import CoverageAnalyzer
from smart_test_generator.generation.test_generator import IncrementalTestGenerator
from smart_test_generator.utils.file_utils import FileUtils

logger = logging.getLogger(__name__)


class PythonCodebaseParser:
    """Parse Python codebase and generate XML content."""

    def __init__(self, root_dir: str, config: Config, exclude_dirs: List[str] = None):
        self.root_dir = Path(root_dir).resolve()
        self.config = config
        
        # Build comprehensive exclusion list from config and defaults
        default_excludes = [
            '__pycache__', '.git', 'venv', 'env', '.env', '.venv',
            'node_modules', '.pytest_cache', '.vscode', '.idea',
            'site-packages', 'lib', 'include', 'bin', 'Scripts'
        ]
        
        # Get exclusions from config
        config_excludes = self.config.get('test_generation.exclude_dirs', [])
        
        # Merge all exclusions
        if exclude_dirs:
            all_excludes = list(set(default_excludes + config_excludes + exclude_dirs))
        else:
            all_excludes = list(set(default_excludes + config_excludes))
            
        self.exclude_dirs = all_excludes
        self.exclude_patterns = self._prepare_exclusion_patterns()

        # Initialize components
        self.tracker = TestGenerationTracker()
        self.coverage_analyzer = CoverageAnalyzer(self.root_dir, config)
        self.incremental_generator = IncrementalTestGenerator(self.root_dir, config)

    def _prepare_exclusion_patterns(self) -> List[str]:
        """Prepare exclusion patterns for glob matching."""
        patterns = []
        for exclude in self.exclude_dirs:
            # Handle glob patterns
            if '*' in exclude or '?' in exclude:
                patterns.append(exclude)
            else:
                # Add exact match and pattern for subdirectories
                patterns.append(exclude)
                patterns.append(f"*/{exclude}")
                patterns.append(f"*/{exclude}/*")
        return patterns

    def _is_excluded_directory(self, dir_path: Path) -> bool:
        """Check if a directory should be excluded."""
        dir_name = dir_path.name
        relative_path = str(dir_path.relative_to(self.root_dir))
        
        # Check exact matches first (most common case)
        if dir_name in self.exclude_dirs:
            return True
            
        # Check glob patterns
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(dir_name, pattern) or fnmatch.fnmatch(relative_path, pattern):
                return True
                
        # Check for common virtual environment indicators
        if self._is_virtual_environment(dir_path):
            return True
            
        return False

    def _is_virtual_environment(self, dir_path: Path) -> bool:
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

    def find_python_files(self) -> List[str]:
        """Find all Python files in the directory, excluding configs, test files, and virtual environments."""
        python_files = []
        config_patterns = ['config.py', 'conf.py', 'settings.py', 'setup.py', '__init__.py']

        for root, dirs, files in os.walk(self.root_dir):
            # Filter out excluded directories
            root_path = Path(root)
            
            # Remove excluded directories from dirs to prevent walking into them
            dirs_to_remove = []
            for d in dirs:
                dir_path = root_path / d
                if self._is_excluded_directory(dir_path):
                    dirs_to_remove.append(d)
                    logger.debug(f"Excluding directory: {dir_path}")
            
            for d in dirs_to_remove:
                dirs.remove(d)

            for file in files:
                if file.endswith('.py'):
                    filepath = Path(root) / file
                    
                    # Skip config files
                    if any(pattern in file.lower() for pattern in config_patterns):
                        logger.debug(f"Excluding config file: {filepath}")
                        continue

                    # Skip test files
                    if FileUtils.is_test_file(filepath):
                        logger.debug(f"Excluding test file: {filepath}")
                        continue

                    python_files.append(str(filepath))

        logger.info(f"Found {len(python_files)} Python files after exclusions")
        return python_files

    def generate_directory_structure(self) -> str:
        """Generate a tree-like directory structure."""
        def build_tree(directory: Path, prefix: str = "", is_last: bool = True) -> List[str]:
            lines = []

            # Get all items in directory
            items = []
            try:
                items = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name))
            except PermissionError:
                return lines

            # Filter out excluded directories and files
            filtered_items = []
            for item in items:
                if item.is_dir() and self._is_excluded_directory(item):
                    continue
                filtered_items.append(item)
            
            items = filtered_items

            for i, item in enumerate(items):
                is_last_item = i == len(items) - 1

                # Add connection characters
                if is_last:
                    current_prefix = prefix + "└── "
                    extension_prefix = prefix + "    "
                else:
                    current_prefix = prefix + "├── "
                    extension_prefix = prefix + "│   "

                if item.is_file() and item.suffix == '.py':
                    lines.append(current_prefix + item.name)
                elif item.is_dir():
                    lines.append(current_prefix + item.name + "/")
                    # Recursively add subdirectory contents
                    lines.extend(build_tree(item, extension_prefix, is_last_item))

            return lines

        structure_lines = [str(self.root_dir) + "/"]
        structure_lines.extend(build_tree(self.root_dir))
        return "\n".join(structure_lines)

    def print_directory_info(self, all_files: List[str], files_to_process: List[str],
                           test_plans: List[TestGenerationPlan] = None):
        """Print detailed directory and file information."""
        logger.info("\n" + "=" * 70)
        logger.info("DIRECTORY STRUCTURE:")
        logger.info("=" * 70)

        # Print directory tree
        structure = self.generate_directory_structure()
        for line in structure.split('\n'):
            logger.info(line)

        logger.info("\n" + "=" * 70)
        logger.info(f"FILE ANALYSIS: {len(all_files)} total files, {len(files_to_process)} to process")
        logger.info("=" * 70)

        # Organize files by directory
        files_by_dir = {}
        for filepath in all_files:
            rel_path = os.path.relpath(filepath, self.root_dir)
            dir_name = os.path.dirname(rel_path) or "."
            if dir_name not in files_by_dir:
                files_by_dir[dir_name] = []
            files_by_dir[dir_name].append(filepath)

        # Print files by directory with process status
        for dir_name in sorted(files_by_dir.keys()):
            logger.info(f"\n{dir_name}/")
            for filepath in sorted(files_by_dir[dir_name]):
                rel_path = os.path.relpath(filepath, self.root_dir)
                status = "✓ TO PROCESS" if filepath in files_to_process else "  skip"
                size = os.path.getsize(filepath)
                logger.info(f"  [{status}] {os.path.basename(filepath)} ({size:,} bytes)")

        # Print test generation plans if available
        if test_plans:
            logger.info("\n" + "=" * 70)
            logger.info("TEST GENERATION PLAN:")
            logger.info("=" * 70)

            total_elements_to_test = sum(len(plan.elements_to_test) for plan in test_plans)
            logger.info(f"Total elements to test: {total_elements_to_test}")

            for plan in test_plans:
                if plan.elements_to_test:
                    logger.info(f"\n{os.path.relpath(plan.source_file, self.root_dir)}:")
                    logger.info(f"  Existing test files: {len(plan.existing_test_files)}")
                    if plan.existing_test_files:
                        for test_file in plan.existing_test_files:
                            logger.info(f"    - {os.path.relpath(test_file, self.root_dir)}")
                    logger.info(f"  Elements needing tests: {len(plan.elements_to_test)}")
                    for element in plan.elements_to_test[:5]:  # Show first 5
                        logger.info(f"    - {element.type}: {element.name} (line {element.line_number})")
                    if len(plan.elements_to_test) > 5:
                        logger.info(f"    ... and {len(plan.elements_to_test) - 5} more")
                    if plan.coverage_before:
                        logger.info(f"  Current coverage: {plan.coverage_before.line_coverage:.1f}%")
                        logger.info(f"  Estimated after: {plan.estimated_coverage_after:.1f}%")

        logger.info("=" * 70 + "\n")

    def parse_file(self, filepath: str) -> FileInfo:
        """Parse a Python file and extract information."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        filename = os.path.basename(filepath)
        relative_path = os.path.relpath(filepath, self.root_dir)

        return FileInfo(
            filepath=relative_path,
            filename=filename,
            content=content
        )

    def generate_xml_content(self, files_info: List[FileInfo]) -> str:
        """Generate XML content for all files."""
        root = ET.Element("codebase")

        for file_info in files_info:
            file_elem = ET.SubElement(root, "file")
            file_elem.set("filename", file_info.filename)
            file_elem.set("filepath", file_info.filepath)
            file_elem.text = file_info.content

        # Pretty print XML
        xml_str = ET.tostring(root, encoding='unicode')
        dom = minidom.parseString(xml_str)
        return dom.toprettyxml(indent="  ")

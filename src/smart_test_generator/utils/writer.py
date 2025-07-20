"""Test file writing utilities."""

import logging
from pathlib import Path
from typing import Optional

from smart_test_generator.config import Config

logger = logging.getLogger(__name__)


class TestFileWriter:
    """Write generated tests to appropriate locations."""

    def __init__(self, root_dir: str, config: Config = None):
        self.root_dir = Path(root_dir).resolve()
        self.config = config or Config()

    def determine_test_path(self, source_path: str) -> str:
        """Determine where to write the test file."""
        source_path = Path(source_path)
        
        # Check if this is already a test file path
        # If the path starts with 'tests' and filename starts with 'test_', it's already a test path
        if (source_path.parts and source_path.parts[0] == 'tests' and 
            source_path.name.startswith('test_')):
            logger.debug(f"Input is already a test path: {source_path}")
            return str(source_path)

        # Common test directory patterns for source files
        if 'src' in source_path.parts:
            # Replace 'src' with 'tests' in the path
            parts = list(source_path.parts)
            src_index = parts.index('src')
            parts[src_index] = 'tests'
            test_path = Path(*parts)
        else:
            # Create a tests directory at the root level
            test_path = Path('tests') / source_path

        # Change filename to test_<filename> only if it doesn't already start with test_
        if not test_path.name.startswith('test_'):
            test_filename = f"test_{source_path.name}"
            test_path = test_path.parent / test_filename

        return str(test_path)

    def write_test_file(self, source_path: str, test_content: str):
        """Write test content to appropriate file."""
        test_path = self.determine_test_path(source_path)
        full_test_path = self.root_dir / test_path

        logger.debug(f"Determined test path for {source_path}: {test_path}")
        logger.debug(f"Full test path: {full_test_path}")

        # Create directory if it doesn't exist
        full_test_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {full_test_path.parent}")

        # Write test file
        try:
            logger.debug(f"Writing {len(test_content):,} characters to {full_test_path}")
            with open(full_test_path, 'w', encoding='utf-8') as f:
                f.write(test_content)

            logger.info(f"Written test file: {test_path} ({len(test_content):,} characters)")
            logger.debug(f"File exists after write: {full_test_path.exists()}")
        except Exception as e:
            logger.error(f"Failed to write test file {test_path}: {e}")
            raise

    def update_test_file(self, test_path: str, new_content: str, merge_strategy: str = 'append'):
        """Update existing test file with new content."""
        full_test_path = self.root_dir / test_path

        if not full_test_path.exists():
            self.write_test_file(test_path, new_content)
            return

        try:
            if merge_strategy == 'append':
                with open(full_test_path, 'a', encoding='utf-8') as f:
                    f.write("\n\n" + new_content)
            elif merge_strategy == 'replace':
                with open(full_test_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

            logger.info(f"Updated test file: {test_path}")
        except Exception as e:
            logger.error(f"Failed to update test file {test_path}: {e}")

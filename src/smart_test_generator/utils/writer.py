"""Test file writing utilities."""

import logging
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
import os
import tempfile
import difflib

from smart_test_generator.config import Config
from smart_test_generator.utils.ast_merge import merge_modules

logger = logging.getLogger(__name__)


@dataclass
class MergeResult:
    changed: bool
    strategy_used: str
    diff: Optional[str] = None
    actions: Optional[List[str]] = None


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

    def write_or_merge_test_file(
        self,
        source_path: str,
        new_source: str,
        *,
        strategy: str = 'append',
        dry_run: bool = False,
    ) -> 'MergeResult':
        """Write or merge test content according to configured strategy.

        - Determines test path from source_path
        - Reads existing content if present
        - Computes merged output per strategy ('append' | 'replace' | 'ast-merge')
        - Optionally formats output (formatter: 'black' | 'none')
        - Returns MergeResult with changed flag, actions, and optional unified diff
        - When not dry_run and a change is needed, writes atomically
        """
        test_path = self.determine_test_path(source_path)
        full_test_path = self.root_dir / test_path

        full_test_path.parent.mkdir(parents=True, exist_ok=True)

        # Read existing content if present
        if full_test_path.exists():
            try:
                existing_text = full_test_path.read_text(encoding='utf-8')
            except Exception:
                existing_text = ''
        else:
            existing_text = ''

        # Compute desired output text and actions according to strategy
        actions: List[str] = []
        output_text: str

        if strategy == 'ast-merge':
            merged_text, merge_actions = merge_modules(existing_text, new_source)
            output_text = merged_text
            actions = merge_actions or ["merge"]
        elif strategy == 'replace':
            output_text = new_source
            actions = ["replace"] if existing_text else ["create"]
        else:
            # Default/fallback to append
            separator = "\n\n" if existing_text and not existing_text.endswith("\n\n") else "\n\n"
            output_text = (existing_text + separator + new_source) if existing_text else new_source
            actions = ["append"] if existing_text else ["create"]
            strategy = 'append' if strategy not in ('append', 'replace', 'ast-merge') else strategy

        # Optional formatting via Black
        try:
            formatter = (self.config.get('test_generation.generation.merge.formatter', 'none')
                         if isinstance(getattr(self.config, 'config', {}), dict) else 'none')
        except Exception:
            formatter = 'none'

        if formatter == 'black':
            try:
                import black  # type: ignore
                output_text = black.format_str(output_text, mode=black.FileMode())
            except Exception:
                # If Black is unavailable or fails, proceed without formatting
                pass

        # Determine change and diff
        changed = (existing_text != output_text)
        diff_text: Optional[str] = None
        if dry_run:
            diff_text = "".join(
                difflib.unified_diff(
                    existing_text.splitlines(keepends=True),
                    output_text.splitlines(keepends=True),
                    fromfile=str(test_path),
                    tofile=str(test_path),
                )
            )

        if dry_run:
            # Only report what would happen
            action_label = actions or (["noop"] if not changed else ["update"])
            log_msg = f"[dry-run] {('No changes' if not changed else 'Would update')} {test_path} via {strategy}"
            logger.info(log_msg)
            return MergeResult(changed=changed, strategy_used=strategy, diff=diff_text, actions=action_label)

        # Not dry-run: write if changed
        if not changed:
            logger.info(f"No changes for {test_path} (strategy={strategy})")
            return MergeResult(changed=False, strategy_used=strategy, actions=["noop"]) 

        # Perform atomic write to avoid partial writes
        try:
            tmp_dir = str(full_test_path.parent)
            with tempfile.NamedTemporaryFile('w', delete=False, dir=tmp_dir, prefix='.tmp_', suffix=full_test_path.suffix, encoding='utf-8') as tmp:
                tmp.write(output_text)
                tmp_path = tmp.name
            os.replace(tmp_path, full_test_path)
            logger.info(f"Updated test file: {test_path} (strategy={strategy})")
        except Exception as e:
            logger.error(f"Failed to write {test_path} atomically: {e}")
            raise

        return MergeResult(changed=True, strategy_used=strategy, actions=actions)

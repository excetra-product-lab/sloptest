"""Git service for analyzing repository changes and providing context for test refinement.

This service provides git operations to understand recent changes that may impact test failures,
helping the refinement system focus on relevant code modifications.
"""

from __future__ import annotations

import logging
import subprocess
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

logger = logging.getLogger(__name__)


@dataclass
class GitDiffEntry:
    """Represents a single file change in a git diff."""
    file_path: str
    status: str  # 'A' (added), 'M' (modified), 'D' (deleted), 'R' (renamed)
    lines_added: int
    lines_removed: int
    is_test_file: bool
    diff_content: Optional[str] = None  # Actual diff content if requested


@dataclass  
class GitChangeAnalysis:
    """Analysis of recent git changes with test-related context."""
    changed_files: List[GitDiffEntry] = field(default_factory=list)
    test_files_changed: List[GitDiffEntry] = field(default_factory=list)
    source_files_changed: List[GitDiffEntry] = field(default_factory=list)
    potential_impact_files: List[str] = field(default_factory=list)  # Files that might be affected
    commit_messages: List[str] = field(default_factory=list)
    time_range: Optional[str] = None
    total_changes: int = 0
    
    
@dataclass
class GitContext:
    """Git repository context for refinement."""
    current_branch: str
    current_commit: str
    has_changes: bool
    is_dirty: bool  # Uncommitted changes
    recent_changes: Optional[GitChangeAnalysis] = None


class GitService:
    """Service for git operations to support test refinement."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_file_patterns = [
            r"test_.*\.py$",
            r".*_test\.py$", 
            r"tests/.*\.py$",
            r"test/.*\.py$"
        ]
    
    def get_git_context(self, include_recent_changes: bool = True,
                       days_back: int = 7, max_commits: int = 10) -> GitContext:
        """Get comprehensive git context for the repository.
        
        Args:
            include_recent_changes: Whether to analyze recent changes
            days_back: How many days back to look for changes
            max_commits: Maximum number of recent commits to analyze
            
        Returns:
            GitContext with repository information
        """
        try:
            current_branch = self._get_current_branch()
            current_commit = self._get_current_commit()
            has_changes = self._has_git_changes()
            is_dirty = self._is_working_dir_dirty()
            
            recent_changes = None
            if include_recent_changes:
                recent_changes = self._analyze_recent_changes(days_back, max_commits)
            
            return GitContext(
                current_branch=current_branch,
                current_commit=current_commit,
                has_changes=has_changes,
                is_dirty=is_dirty,
                recent_changes=recent_changes
            )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Git operation failed: {e}")
            return GitContext(
                current_branch="unknown",
                current_commit="unknown", 
                has_changes=False,
                is_dirty=False
            )
    
    def get_recent_test_changes(self, since: Optional[str] = None,
                              include_diff_content: bool = False) -> List[GitDiffEntry]:
        """Get recent changes to test files specifically.
        
        Args:
            since: Git revision to compare from (default: last 7 days)
            include_diff_content: Whether to include actual diff content
            
        Returns:
            List of test file changes
        """
        if since is None:
            since = self._get_date_string_days_ago(7)
        
        try:
            # Get list of changed files
            cmd = ["git", "log", "--name-status", "--since", since, "--pretty=format:%H"]
            result = subprocess.run(cmd, cwd=self.project_root, 
                                  capture_output=True, text=True, check=True)
            
            test_changes = []
            current_commit = None
            
            for line in result.stdout.splitlines():
                if line and not line.startswith(('A', 'M', 'D', 'R')):
                    current_commit = line  # This is a commit hash
                    continue
                    
                if line and current_commit:
                    parts = line.split('\t', 1)
                    if len(parts) >= 2:
                        status = parts[0]
                        file_path = parts[1]
                        
                        if self._is_test_file(file_path):
                            diff_content = None
                            if include_diff_content:
                                diff_content = self._get_file_diff(file_path, current_commit)
                            
                            lines_added, lines_removed = self._get_line_changes(
                                file_path, current_commit
                            )
                            
                            test_changes.append(GitDiffEntry(
                                file_path=file_path,
                                status=status,
                                lines_added=lines_added,
                                lines_removed=lines_removed,
                                is_test_file=True,
                                diff_content=diff_content
                            ))
            
            return test_changes
            
        except subprocess.CalledProcessError as e:
            if e.returncode == 128:
                logger.warning(f"Git repository not accessible for test changes analysis: {e}")
            else:
                logger.warning(f"Failed to get recent test changes: {e}")
            return []
    
    def get_changes_affecting_tests(self, test_file_paths: List[str],
                                  since: Optional[str] = None) -> Dict[str, List[str]]:
        """Find source file changes that might affect specific test files.
        
        Args:
            test_file_paths: List of test file paths to analyze
            since: Git revision to compare from
            
        Returns:
            Dictionary mapping test files to potentially affected source files
        """
        if since is None:
            since = self._get_date_string_days_ago(7)
        
        test_to_sources = {}
        
        for test_file in test_file_paths:
            # Infer source files that this test might cover
            potential_sources = self._infer_tested_source_files(test_file)
            
            # Check which of these sources have changed recently
            changed_sources = []
            for source_file in potential_sources:
                if self._file_changed_since(source_file, since):
                    changed_sources.append(source_file)
            
            if changed_sources:
                test_to_sources[test_file] = changed_sources
        
        return test_to_sources
    
    def get_commit_context_for_failures(self, failed_test_files: List[str]) -> Dict[str, Any]:
        """Get commit context that might be relevant to test failures.
        
        Args:
            failed_test_files: List of test files that are failing
            
        Returns:
            Dictionary with commit context information
        """
        context = {
            "recent_commits": [],
            "changed_test_files": [],
            "changed_source_files": [],
            "commit_messages": []
        }
        
        try:
            # Get recent commits (last 10)
            cmd = ["git", "log", "--oneline", "-10"]
            result = subprocess.run(cmd, cwd=self.project_root,
                                  capture_output=True, text=True, check=True)
            
            context["recent_commits"] = result.stdout.strip().split('\n')
            
            # Get changes in the last week
            since = self._get_date_string_days_ago(7)
            recent_changes = self._analyze_recent_changes(days_back=7, max_commits=10)
            
            if recent_changes:
                context["changed_test_files"] = [
                    f.file_path for f in recent_changes.test_files_changed
                ]
                context["changed_source_files"] = [
                    f.file_path for f in recent_changes.source_files_changed
                ]
                context["commit_messages"] = recent_changes.commit_messages
            
            return context
            
        except subprocess.CalledProcessError as e:
            if e.returncode == 128:
                logger.warning(f"Git repository not accessible for commit context: {e}")
            else:
                logger.warning(f"Failed to get commit context: {e}")
            return context
    
    def _get_current_branch(self) -> str:
        """Get current git branch name."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.project_root, capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return "unknown"
    
    def _get_current_commit(self) -> str:
        """Get current commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root, capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return "unknown"
    
    def _has_git_changes(self) -> bool:
        """Check if there are any git changes in the repository."""
        try:
            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_root, capture_output=True, text=True, check=True
            )
            return bool(result.stdout.strip())
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
    
    def _is_working_dir_dirty(self) -> bool:
        """Check if working directory has uncommitted changes."""
        try:
            result = subprocess.run(
                ["git", "diff-index", "--quiet", "HEAD", "--"],
                cwd=self.project_root, capture_output=True, check=True
            )
            return False  # No changes if command succeeds
        except subprocess.CalledProcessError as e:
            # git diff-index exits with 1 specifically if there are changes
            # But only if the command actually contains "diff-index"
            if e.returncode == 1 and e.cmd and "diff-index" in str(e.cmd):
                return True  # Changes exist
            else:
                return False  # Other error or not diff-index command, assume clean state
        except subprocess.TimeoutExpired:
            # On timeout, default to False (assume clean/safe state)
            return False
    
    def _analyze_recent_changes(self, days_back: int, max_commits: int) -> GitChangeAnalysis:
        """Analyze recent changes to understand what might affect tests."""
        since = self._get_date_string_days_ago(days_back)
        
        try:
            # Get commit messages
            cmd = ["git", "log", "--since", since, f"--max-count={max_commits}", 
                   "--pretty=format:%s"]
            result = subprocess.run(cmd, cwd=self.project_root,
                                  capture_output=True, text=True, check=True)
            commit_messages = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Get changed files with stats
            cmd = ["git", "log", "--since", since, "--name-status", "--pretty=format:%H"]
            result = subprocess.run(cmd, cwd=self.project_root,
                                  capture_output=True, text=True, check=True)
            
            changed_files = []
            test_files_changed = []
            source_files_changed = []
            current_commit = None
            
            for line in result.stdout.splitlines():
                if line and not line.startswith(('A', 'M', 'D', 'R')):
                    current_commit = line
                    continue
                
                if line and current_commit:
                    parts = line.split('\t', 1)
                    if len(parts) >= 2:
                        status = parts[0]
                        file_path = parts[1]
                        
                        lines_added, lines_removed = self._get_line_changes(file_path, current_commit)
                        is_test = self._is_test_file(file_path)
                        
                        diff_entry = GitDiffEntry(
                            file_path=file_path,
                            status=status,
                            lines_added=lines_added,
                            lines_removed=lines_removed,
                            is_test_file=is_test
                        )
                        
                        changed_files.append(diff_entry)
                        
                        if is_test:
                            test_files_changed.append(diff_entry)
                        else:
                            source_files_changed.append(diff_entry)
            
            return GitChangeAnalysis(
                changed_files=changed_files,
                test_files_changed=test_files_changed,
                source_files_changed=source_files_changed,
                commit_messages=commit_messages,
                time_range=f"last {days_back} days",
                total_changes=len(changed_files)
            )
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            # Handle common git errors gracefully
            if hasattr(e, 'returncode') and e.returncode == 128:
                logger.warning(f"Git repository might not be initialized or accessible: {e}")
            else:
                logger.warning(f"Git operation failed: {e}")
            return GitChangeAnalysis()
    
    def _is_test_file(self, file_path: str) -> bool:
        """Check if a file path represents a test file."""
        for pattern in self.test_file_patterns:
            if re.search(pattern, file_path):
                return True
        return False
    
    def _get_file_diff(self, file_path: str, commit: str) -> Optional[str]:
        """Get diff content for a specific file at a commit."""
        try:
            cmd = ["git", "show", f"{commit}:{file_path}"]
            result = subprocess.run(cmd, cwd=self.project_root,
                                  capture_output=True, text=True, check=True)
            return result.stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
    
    def _get_line_changes(self, file_path: str, commit: str) -> Tuple[int, int]:
        """Get number of lines added and removed for a file in a commit."""
        try:
            cmd = ["git", "show", "--numstat", commit, "--", file_path]
            result = subprocess.run(cmd, cwd=self.project_root,
                                  capture_output=True, text=True, check=True)
            
            for line in result.stdout.splitlines():
                if line and '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        added = int(parts[0]) if parts[0] != '-' else 0
                        removed = int(parts[1]) if parts[1] != '-' else 0
                        return added, removed
            
            return 0, 0
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
            return 0, 0
    
    def _infer_tested_source_files(self, test_file_path: str) -> List[str]:
        """Infer which source files might be tested by a test file."""
        potential_sources = []
        
        # Convert test file path to potential source file paths
        # E.g., tests/test_module.py -> module.py or src/module.py
        test_name = Path(test_file_path).name
        
        # Remove test_ prefix and _test suffix
        source_name = test_name
        if source_name.startswith('test_'):
            source_name = source_name[5:]
        if source_name.endswith('_test.py'):
            source_name = source_name[:-8] + '.py'
        
        # Look in common source directories
        source_dirs = ['src', 'lib', '.', 'app']
        for source_dir in source_dirs:
            potential_path = str(Path(source_dir) / source_name)
            if (self.project_root / potential_path).exists():
                potential_sources.append(potential_path)
        
        return potential_sources
    
    def _file_changed_since(self, file_path: str, since: str) -> bool:
        """Check if a file has changed since a given time."""
        try:
            cmd = ["git", "log", "--since", since, "--name-only", "--", file_path]
            result = subprocess.run(cmd, cwd=self.project_root,
                                  capture_output=True, text=True, check=True)
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False
    
    def _get_date_string_days_ago(self, days: int) -> str:
        """Get date string for N days ago in git format."""
        target_date = datetime.now() - timedelta(days=days)
        return target_date.strftime("%Y-%m-%d")


def create_git_service(project_root: Path) -> GitService:
    """Factory function to create a GitService instance."""
    return GitService(project_root)

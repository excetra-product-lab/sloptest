"""Tests for GitService functionality."""

import pytest
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from smart_test_generator.services.git_service import (
    GitService, 
    GitContext, 
    GitChangeAnalysis, 
    GitDiffEntry,
    create_git_service
)


@pytest.fixture
def temp_git_repo():
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)
        
        # Initialize git repo
        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        
        # Create initial file and commit
        test_file = repo_path / "test_file.py"
        test_file.write_text("def hello():\n    pass\n")
        
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True)
        
        yield repo_path


@pytest.fixture
def git_service(temp_git_repo):
    """Create GitService instance with temporary repository."""
    return GitService(temp_git_repo)


class TestGitService:
    """Test cases for GitService."""
    
    def test_initialization(self, temp_git_repo):
        """Test GitService initialization."""
        service = GitService(temp_git_repo)
        assert service.project_root == temp_git_repo
        assert len(service.test_file_patterns) > 0
        
    def test_factory_function(self, temp_git_repo):
        """Test create_git_service factory function."""
        service = create_git_service(temp_git_repo)
        assert isinstance(service, GitService)
        assert service.project_root == temp_git_repo
    
    def test_get_current_branch(self, git_service, temp_git_repo):
        """Test getting current branch name."""
        # Should be on main/master branch
        branch = git_service._get_current_branch()
        assert branch in ["main", "master"]
        
        # Create and switch to new branch
        subprocess.run(["git", "checkout", "-b", "test-branch"], 
                      cwd=temp_git_repo, check=True, capture_output=True)
        
        branch = git_service._get_current_branch()
        assert branch == "test-branch"
    
    def test_get_current_commit(self, git_service):
        """Test getting current commit hash."""
        commit = git_service._get_current_commit()
        assert len(commit) == 40  # Full SHA
        assert all(c in "0123456789abcdef" for c in commit)
    
    def test_has_git_changes(self, git_service, temp_git_repo):
        """Test detecting git changes."""
        # No changes initially
        assert not git_service._has_git_changes()
        
        # Add a new file
        new_file = temp_git_repo / "new_file.py"
        new_file.write_text("print('new')")
        
        # Should detect changes now
        assert git_service._has_git_changes()
        
        # Stage the file
        subprocess.run(["git", "add", "new_file.py"], cwd=temp_git_repo, check=True)
        assert git_service._has_git_changes()
        
        # Commit the file
        subprocess.run(["git", "commit", "-m", "Add new file"], 
                      cwd=temp_git_repo, check=True)
        assert not git_service._has_git_changes()
    
    def test_is_working_dir_dirty(self, git_service, temp_git_repo):
        """Test detecting dirty working directory."""
        # Clean initially
        assert not git_service._is_working_dir_dirty()
        
        # Modify existing file
        test_file = temp_git_repo / "test_file.py"
        test_file.write_text("def hello():\n    return 'world'\n")
        
        # Should be dirty now
        assert git_service._is_working_dir_dirty()
        
        # Stage the change
        subprocess.run(["git", "add", "test_file.py"], cwd=temp_git_repo, check=True)
        assert git_service._is_working_dir_dirty()
        
        # Commit the change
        subprocess.run(["git", "commit", "-m", "Update test file"], 
                      cwd=temp_git_repo, check=True)
        assert not git_service._is_working_dir_dirty()
    
    def test_is_test_file(self, git_service):
        """Test test file pattern detection."""
        # Test files
        assert git_service._is_test_file("test_module.py")
        assert git_service._is_test_file("module_test.py")
        assert git_service._is_test_file("tests/test_something.py")
        assert git_service._is_test_file("test/unit_test.py")
        
        # Non-test files
        assert not git_service._is_test_file("module.py")
        assert not git_service._is_test_file("src/main.py")
        assert not git_service._is_test_file("README.md")
    
    def test_get_git_context_basic(self, git_service):
        """Test basic git context retrieval."""
        context = git_service.get_git_context(include_recent_changes=False)
        
        assert isinstance(context, GitContext)
        assert context.current_branch in ["main", "master"]
        assert len(context.current_commit) == 40
        assert isinstance(context.has_changes, bool)
        assert isinstance(context.is_dirty, bool)
        assert context.recent_changes is None  # Not included
    
    def test_get_git_context_with_changes(self, git_service, temp_git_repo):
        """Test git context with recent changes analysis."""
        # Add a test file
        test_file = temp_git_repo / "test_new.py"
        test_file.write_text("def test_something():\n    assert True\n")
        subprocess.run(["git", "add", "test_new.py"], cwd=temp_git_repo, check=True)
        subprocess.run(["git", "commit", "-m", "Add test file"], 
                      cwd=temp_git_repo, check=True)
        
        context = git_service.get_git_context(include_recent_changes=True)
        
        assert context.recent_changes is not None
        assert isinstance(context.recent_changes, GitChangeAnalysis)
    
    def test_infer_tested_source_files(self, git_service, temp_git_repo):
        """Test inference of source files from test files."""
        # Create source file
        src_dir = temp_git_repo / "src"
        src_dir.mkdir()
        src_file = src_dir / "module.py"
        src_file.write_text("def function():\n    pass\n")
        
        # Test inference
        sources = git_service._infer_tested_source_files("test_module.py")
        
        # Should find the source file
        assert "src/module.py" in sources
    
    @patch('subprocess.run')
    def test_git_command_failure_handling(self, mock_run, temp_git_repo):
        """Test handling of git command failures."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        
        service = GitService(temp_git_repo)
        context = service.get_git_context()
        
        # Should handle failure gracefully
        assert context.current_branch == "unknown"
        assert context.current_commit == "unknown"
        assert not context.has_changes
        assert not context.is_dirty


class TestGitChangeAnalysis:
    """Test cases for GitChangeAnalysis data structure."""
    
    def test_git_diff_entry_creation(self):
        """Test GitDiffEntry creation."""
        entry = GitDiffEntry(
            file_path="test_file.py",
            status="M",
            lines_added=5,
            lines_removed=2,
            is_test_file=True
        )
        
        assert entry.file_path == "test_file.py"
        assert entry.status == "M"
        assert entry.lines_added == 5
        assert entry.lines_removed == 2
        assert entry.is_test_file
        assert entry.diff_content is None
    
    def test_git_change_analysis_creation(self):
        """Test GitChangeAnalysis creation."""
        entry1 = GitDiffEntry("test_file.py", "M", 5, 2, True)
        entry2 = GitDiffEntry("src_file.py", "A", 10, 0, False)
        
        analysis = GitChangeAnalysis(
            changed_files=[entry1, entry2],
            test_files_changed=[entry1],
            source_files_changed=[entry2],
            commit_messages=["Fix bug", "Add feature"],
            time_range="last 7 days",
            total_changes=2
        )
        
        assert len(analysis.changed_files) == 2
        assert len(analysis.test_files_changed) == 1
        assert len(analysis.source_files_changed) == 1
        assert analysis.total_changes == 2
        assert "Fix bug" in analysis.commit_messages
    
    def test_git_context_creation(self):
        """Test GitContext creation."""
        context = GitContext(
            current_branch="main",
            current_commit="abc123",
            has_changes=False,
            is_dirty=True
        )
        
        assert context.current_branch == "main"
        assert context.current_commit == "abc123"
        assert not context.has_changes
        assert context.is_dirty
        assert context.recent_changes is None


class TestGitServiceIntegration:
    """Integration tests for GitService."""
    
    def test_recent_test_changes_detection(self, git_service, temp_git_repo):
        """Test detection of recent test file changes."""
        # Add multiple commits with test files
        for i in range(3):
            test_file = temp_git_repo / f"test_feature_{i}.py"
            test_file.write_text(f"def test_feature_{i}():\n    assert True\n")
            subprocess.run(["git", "add", f"test_feature_{i}.py"], 
                          cwd=temp_git_repo, check=True)
            subprocess.run(["git", "commit", "-m", f"Add test {i}"], 
                          cwd=temp_git_repo, check=True)
        
        # Get recent test changes
        changes = git_service.get_recent_test_changes()
        
        # Should find the test files
        test_file_names = [change.file_path for change in changes]
        assert any("test_feature" in name for name in test_file_names)
    
    def test_changes_affecting_tests_analysis(self, git_service, temp_git_repo):
        """Test analysis of changes that might affect specific tests."""
        # Create source file
        src_file = temp_git_repo / "module.py"
        src_file.write_text("def function():\n    return 'original'\n")
        subprocess.run(["git", "add", "module.py"], cwd=temp_git_repo, check=True)
        subprocess.run(["git", "commit", "-m", "Add module"], 
                      cwd=temp_git_repo, check=True)
        
        # Modify source file
        src_file.write_text("def function():\n    return 'modified'\n")
        subprocess.run(["git", "add", "module.py"], cwd=temp_git_repo, check=True)
        subprocess.run(["git", "commit", "-m", "Modify module"], 
                      cwd=temp_git_repo, check=True)
        
        # Test the analysis
        test_files = ["test_module.py"]
        affected = git_service.get_changes_affecting_tests(test_files)
        
        # Results depend on whether the source file actually exists
        assert isinstance(affected, dict)
    
    def test_commit_context_for_failures(self, git_service, temp_git_repo):
        """Test getting commit context for failed tests."""
        # Add some commits
        for i in range(2):
            test_file = temp_git_repo / f"failing_test_{i}.py"
            test_file.write_text(f"def test_that_fails_{i}():\n    assert False\n")
            subprocess.run(["git", "add", f"failing_test_{i}.py"], 
                          cwd=temp_git_repo, check=True)
            subprocess.run(["git", "commit", "-m", f"Add failing test {i}"], 
                          cwd=temp_git_repo, check=True)
        
        # Get context for failed tests
        failed_tests = ["failing_test_0.py", "failing_test_1.py"]
        context = git_service.get_commit_context_for_failures(failed_tests)
        
        assert "recent_commits" in context
        assert "commit_messages" in context
        assert isinstance(context["recent_commits"], list)
        assert isinstance(context["commit_messages"], list)


class TestErrorHandling:
    """Test error handling in GitService."""
    
    def test_non_git_repository(self):
        """Test behavior with non-git repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = GitService(Path(temp_dir))
            context = service.get_git_context()
            
            # Should handle gracefully
            assert context.current_branch == "unknown"
            assert context.current_commit == "unknown"
    
    @patch('subprocess.run')
    def test_subprocess_timeout(self, mock_run, temp_git_repo):
        """Test handling of subprocess timeouts."""
        mock_run.side_effect = subprocess.TimeoutExpired("git", 30)
        
        service = GitService(temp_git_repo)
        context = service.get_git_context()
        
        # Should handle timeout gracefully
        assert context.current_branch == "unknown"
        assert context.current_commit == "unknown"
    
    def test_empty_repository(self):
        """Test behavior with empty git repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)
            subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
            
            service = GitService(repo_path)
            context = service.get_git_context()
            
            # Empty repo behavior may vary by git version
            assert isinstance(context, GitContext)

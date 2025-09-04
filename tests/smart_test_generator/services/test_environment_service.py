"""Tests for EnvironmentService detection methods."""

import os
import subprocess
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from smart_test_generator.services.environment_service import (
    EnvironmentService,
    EnvironmentManager,
    EnvironmentInfo
)


class TestEnvironmentService:
    """Test suite for EnvironmentService detection methods."""
    
    def setup_method(self, method):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Mock()
        self.feedback = Mock()
        self.service = EnvironmentService(
            project_root=self.temp_dir,
            config=self.config,
            feedback=self.feedback
        )
    
    def teardown_method(self, method):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_detect_poetry_environment_with_poetry_lock(self):
        """Test detection of Poetry environment with poetry.lock."""
        # Create pyproject.toml with Poetry configuration
        pyproject_content = """
[tool.poetry]
name = "test-project"
version = "0.1.0"
description = ""

[tool.poetry.dependencies]
python = "^3.8"

[tool.poetry.group.dev.dependencies]
pytest = "^7.0.0"
"""
        (self.temp_dir / "pyproject.toml").write_text(pyproject_content)
        (self.temp_dir / "poetry.lock").touch()  # Create empty lock file
        
        with patch.object(self.service, '_get_poetry_python_executable') as mock_python, \
             patch.object(self.service, '_get_poetry_venv_path') as mock_venv, \
             patch.object(self.service, '_get_python_version') as mock_version:
            
            mock_python.return_value = "/path/to/poetry/python"
            mock_venv.return_value = "/path/to/poetry/venv"
            mock_version.return_value = "3.9.18"
            
            env_info = self.service._detect_poetry()
            
            assert env_info is not None
            assert env_info.manager == EnvironmentManager.POETRY
            assert env_info.python_executable == "/path/to/poetry/python"
            assert env_info.virtual_env_path == "/path/to/poetry/venv"
            assert env_info.is_virtual_env is True
            assert str(self.temp_dir / "pyproject.toml") in env_info.detected_files
            assert str(self.temp_dir / "poetry.lock") in env_info.detected_files
    
    def test_detect_poetry_environment_without_poetry_lock(self):
        """Test detection of Poetry environment without poetry.lock."""
        pyproject_content = """
[tool.poetry]
name = "test-project"
version = "0.1.0"
"""
        (self.temp_dir / "pyproject.toml").write_text(pyproject_content)
        
        with patch.object(self.service, '_get_poetry_python_executable') as mock_python, \
             patch.object(self.service, '_get_poetry_venv_path') as mock_venv, \
             patch.object(self.service, '_get_python_version') as mock_version:
            
            mock_python.return_value = "/path/to/poetry/python"
            mock_venv.return_value = None
            mock_version.return_value = "3.9.18"
            
            env_info = self.service._detect_poetry()
            
            assert env_info is not None
            assert env_info.manager == EnvironmentManager.POETRY
            assert env_info.lockfile is None
            assert env_info.is_virtual_env is False
    
    def test_detect_poetry_no_pyproject_toml(self):
        """Test Poetry detection fails when no pyproject.toml exists."""
        env_info = self.service._detect_poetry()
        assert env_info is None
    
    def test_detect_poetry_pyproject_without_poetry_section(self):
        """Test Poetry detection fails when pyproject.toml has no [tool.poetry] section."""
        pyproject_content = """
[tool.other]
name = "test-project"
"""
        (self.temp_dir / "pyproject.toml").write_text(pyproject_content)
        
        env_info = self.service._detect_poetry()
        assert env_info is None
    
    def test_detect_poetry_invalid_toml(self):
        """Test Poetry detection handles invalid TOML gracefully."""
        (self.temp_dir / "pyproject.toml").write_text("invalid toml content [[[")
        
        env_info = self.service._detect_poetry()
        assert env_info is None
    
    def test_detect_pipenv_environment(self):
        """Test detection of Pipenv environment."""
        (self.temp_dir / "Pipfile").write_text("[dev-packages]\npytest = '*'")
        (self.temp_dir / "Pipfile.lock").write_text("{}")
        
        with patch.object(self.service, '_get_pipenv_python_executable') as mock_python, \
             patch.object(self.service, '_get_pipenv_venv_path') as mock_venv, \
             patch.object(self.service, '_get_python_version') as mock_version:
            
            mock_python.return_value = "/path/to/pipenv/python"
            mock_venv.return_value = "/path/to/pipenv/venv"
            mock_version.return_value = "3.9.18"
            
            env_info = self.service._detect_pipenv()
            
            assert env_info is not None
            assert env_info.manager == EnvironmentManager.PIPENV
            assert env_info.python_executable == "/path/to/pipenv/python"
            assert env_info.virtual_env_path == "/path/to/pipenv/venv"
            assert env_info.is_virtual_env is True
    
    def test_detect_pipenv_no_pipfile(self):
        """Test Pipenv detection fails when no Pipfile exists."""
        env_info = self.service._detect_pipenv()
        assert env_info is None
    
    def test_detect_conda_with_environment_yml(self):
        """Test detection of Conda environment with environment.yml."""
        env_yml_content = """
name: test-env
channels:
  - defaults
dependencies:
  - python=3.9
  - pytest
"""
        (self.temp_dir / "environment.yml").write_text(env_yml_content)
        
        with patch.dict(os.environ, {
            "CONDA_DEFAULT_ENV": "test-env",
            "CONDA_PREFIX": "/path/to/conda/envs/test-env"
        }):
            with patch.object(self.service, '_get_conda_python_executable') as mock_python, \
                 patch.object(self.service, '_get_python_version') as mock_version:
                
                mock_python.return_value = "/path/to/conda/envs/test-env/bin/python"
                mock_version.return_value = "3.9.18"
                
                env_info = self.service._detect_conda()
                
                assert env_info is not None
                assert env_info.manager == EnvironmentManager.CONDA
                assert env_info.env_name == "test-env"
                assert env_info.virtual_env_path == "/path/to/conda/envs/test-env"
                assert env_info.is_virtual_env is True
    
    def test_detect_conda_environment_variables_only(self):
        """Test detection of Conda environment with only environment variables."""
        with patch.dict(os.environ, {
            "CONDA_DEFAULT_ENV": "base",
            "CONDA_PREFIX": "/opt/conda"
        }):
            with patch.object(self.service, '_get_conda_python_executable') as mock_python, \
                 patch.object(self.service, '_get_python_version') as mock_version:
                
                mock_python.return_value = "/opt/conda/bin/python"
                mock_version.return_value = "3.9.18"
                
                env_info = self.service._detect_conda()
                
                assert env_info is not None
                assert env_info.manager == EnvironmentManager.CONDA
                assert env_info.env_name == "base"
                assert env_info.project_file is None
    
    def test_detect_conda_no_indicators(self):
        """Test Conda detection fails when no conda indicators exist."""
        with patch.dict(os.environ, {}, clear=True):
            env_info = self.service._detect_conda()
            assert env_info is None
    
    def test_detect_uv_environment(self):
        """Test detection of uv environment."""
        (self.temp_dir / "uv.lock").write_text("# uv lock file")
        pyproject_content = """
[project]
name = "test-project"
version = "0.1.0"

[tool.uv]
dev-dependencies = ["pytest>=7.0.0"]
"""
        (self.temp_dir / "pyproject.toml").write_text(pyproject_content)
        
        with patch.object(self.service, '_get_uv_python_executable') as mock_python, \
             patch.object(self.service, '_get_uv_venv_path') as mock_venv, \
             patch.object(self.service, '_get_python_version') as mock_version:
            
            mock_python.return_value = "/path/to/uv/python"
            mock_venv.return_value = "/path/to/uv/venv"
            mock_version.return_value = "3.9.18"
            
            env_info = self.service._detect_uv()
            
            assert env_info is not None
            assert env_info.manager == EnvironmentManager.UV
            assert env_info.python_executable == "/path/to/uv/python"
            assert env_info.lockfile == str(self.temp_dir / "uv.lock")
    
    def test_detect_uv_no_lock_file(self):
        """Test uv detection fails when no uv.lock exists."""
        env_info = self.service._detect_uv()
        assert env_info is None
    
    def test_detect_venv_active_environment(self):
        """Test detection of active venv environment."""
        venv_path = "/path/to/virtual/env"
        
        with patch.dict(os.environ, {"VIRTUAL_ENV": venv_path}):
            with patch.object(self.service, '_get_venv_python_executable') as mock_python, \
                 patch.object(self.service, '_get_python_version') as mock_version:
                
                mock_python.return_value = f"{venv_path}/bin/python"
                mock_version.return_value = "3.9.18"
                
                # Mock pyvenv.cfg file exists
                with patch.object(Path, 'exists') as mock_exists:
                    mock_exists.return_value = True
                    
                    env_info = self.service._detect_venv()
                    
                    assert env_info is not None
                    assert env_info.manager == EnvironmentManager.VENV
                    assert env_info.virtual_env_path == venv_path
                    assert env_info.is_virtual_env is True
    
    def test_detect_venv_project_directory(self):
        """Test detection of venv in project directory."""
        venv_dir = self.temp_dir / ".venv"
        venv_dir.mkdir()
        (venv_dir / "pyvenv.cfg").write_text("home = /usr/bin")
        
        # Clear VIRTUAL_ENV so the method looks in the project directory instead of using the current venv
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(self.service, '_get_venv_python_executable') as mock_python, \
             patch.object(self.service, '_get_python_version') as mock_version:
            
            mock_python.return_value = str(venv_dir / "bin" / "python")
            mock_version.return_value = "3.9.18"
            
            env_info = self.service._detect_venv()
            
            assert env_info is not None
            assert env_info.manager == EnvironmentManager.VENV
            assert env_info.virtual_env_path == str(venv_dir)
    
    def test_detect_venv_no_virtual_env(self):
        """Test venv detection fails when no virtual environment exists."""
        with patch.dict(os.environ, {}, clear=True):
            env_info = self.service._detect_venv()
            assert env_info is None
    
    def test_detect_pyenv_environment(self):
        """Test detection of pyenv environment."""
        (self.temp_dir / ".python-version").write_text("3.9.18")
        
        with patch.object(self.service, '_get_pyenv_python_executable') as mock_python:
            mock_python.return_value = "/opt/pyenv/versions/3.9.18/bin/python"
            
            env_info = self.service._detect_pyenv()
            
            assert env_info is not None
            assert env_info.manager == EnvironmentManager.PYENV
            assert env_info.python_version == "3.9.18"
            assert env_info.python_executable == "/opt/pyenv/versions/3.9.18/bin/python"
    
    def test_detect_pyenv_parent_directory(self):
        """Test detection of pyenv environment in parent directory."""
        parent_dir = self.temp_dir.parent
        version_file = parent_dir / ".python-version"
        version_file.write_text("3.8.10")
        
        try:
            with patch.object(self.service, '_get_pyenv_python_executable') as mock_python:
                mock_python.return_value = "/opt/pyenv/versions/3.8.10/bin/python"
                
                env_info = self.service._detect_pyenv()
                
                assert env_info is not None
                assert env_info.python_version == "3.8.10"
        finally:
            # Clean up
            if version_file.exists():
                version_file.unlink()
    
    def test_detect_pyenv_no_version_file(self):
        """Test pyenv detection fails when no .python-version file exists."""
        env_info = self.service._detect_pyenv()
        assert env_info is None
    
    def test_detect_system_environment(self):
        """Test detection of system Python environment."""
        env_info = self.service._detect_system()
        
        assert env_info is not None
        assert env_info.manager == EnvironmentManager.SYSTEM
        assert env_info.python_executable == sys.executable
        assert env_info.python_version.startswith(f"{sys.version_info.major}.{sys.version_info.minor}")
    
    def test_detect_current_environment_cache(self):
        """Test that environment detection is cached."""
        # Mock the first detection method to return a result
        with patch.object(self.service, '_detect_poetry') as mock_poetry:
            mock_poetry.return_value = EnvironmentInfo(
                manager=EnvironmentManager.POETRY,
                python_executable="/test/python",
                python_version="3.9.0"
            )
            
            # First call should detect and cache
            env_info1 = self.service.detect_current_environment()
            assert mock_poetry.call_count == 1
            
            # Second call should use cache
            env_info2 = self.service.detect_current_environment()
            assert mock_poetry.call_count == 1  # Should not be called again
            assert env_info1 is env_info2  # Same object from cache
    
    def test_detect_current_environment_force_refresh(self):
        """Test force refresh bypasses cache."""
        with patch.object(self.service, '_detect_poetry') as mock_poetry:
            mock_poetry.return_value = EnvironmentInfo(
                manager=EnvironmentManager.POETRY,
                python_executable="/test/python",
                python_version="3.9.0"
            )
            
            # First call
            env_info1 = self.service.detect_current_environment()
            assert mock_poetry.call_count == 1
            
            # Force refresh should bypass cache
            env_info2 = self.service.detect_current_environment(force_refresh=True)
            assert mock_poetry.call_count == 2
    
    def test_detect_current_environment_detection_priority(self):
        """Test that environment detection follows priority order."""
        with patch.object(self.service, '_detect_poetry') as mock_poetry, \
             patch.object(self.service, '_detect_pipenv') as mock_pipenv, \
             patch.object(self.service, '_detect_conda') as mock_conda:
            
            # Set up mocks to return None for higher priority methods
            mock_poetry.return_value = None
            mock_pipenv.return_value = None
            mock_conda.return_value = EnvironmentInfo(
                manager=EnvironmentManager.CONDA,
                python_executable="/conda/python",
                python_version="3.9.0"
            )
            
            env_info = self.service.detect_current_environment()
            
            # Should try poetry and pipenv first, then succeed with conda
            assert mock_poetry.called
            assert mock_pipenv.called
            assert mock_conda.called
            assert env_info.manager == EnvironmentManager.CONDA
    
    def test_get_current_python_executable(self):
        """Test getting current Python executable."""
        with patch.object(self.service, 'detect_current_environment') as mock_detect:
            mock_detect.return_value = EnvironmentInfo(
                manager=EnvironmentManager.POETRY,
                python_executable="/poetry/python",
                python_version="3.9.0"
            )
            
            executable = self.service.get_current_python_executable()
            assert executable == "/poetry/python"
    
    def test_get_virtual_env_path(self):
        """Test getting virtual environment path."""
        with patch.object(self.service, 'detect_current_environment') as mock_detect:
            mock_detect.return_value = EnvironmentInfo(
                manager=EnvironmentManager.VENV,
                python_executable="/venv/bin/python",
                python_version="3.9.0",
                virtual_env_path="/venv",
                is_virtual_env=True
            )
            
            venv_path = self.service.get_virtual_env_path()
            assert venv_path == "/venv"
    
    def test_has_virtual_env(self):
        """Test checking if running in virtual environment."""
        with patch.object(self.service, 'detect_current_environment') as mock_detect:
            mock_detect.return_value = EnvironmentInfo(
                manager=EnvironmentManager.VENV,
                python_executable="/venv/bin/python",
                python_version="3.9.0",
                is_virtual_env=True
            )
            
            assert self.service.has_virtual_env() is True
    
    def test_get_current_env_recommendations_system(self):
        """Test getting recommendations for system Python."""
        with patch.object(self.service, 'detect_current_environment') as mock_detect:
            mock_detect.return_value = EnvironmentInfo(
                manager=EnvironmentManager.SYSTEM,
                python_executable="/usr/bin/python",
                python_version="3.9.0"
            )
            
            recommendations = self.service.get_current_env_recommendations()
            assert len(recommendations) > 0
            assert any("virtual environment" in rec for rec in recommendations)


class TestEnvironmentServiceHelperMethods:
    """Test helper methods for getting environment-specific executables."""
    
    def setup_method(self, method):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Mock()
        self.feedback = Mock()
        self.service = EnvironmentService(
            project_root=self.temp_dir,
            config=self.config,
            feedback=self.feedback
        )
    
    def teardown_method(self, method):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('shutil.which')
    @patch('subprocess.run')
    def test_get_poetry_python_executable_success(self, mock_run, mock_which):
        """Test getting Poetry Python executable successfully."""
        mock_which.return_value = "/usr/local/bin/poetry"
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "/path/to/poetry/python\n"
        mock_run.return_value = mock_result
        
        executable = self.service._get_poetry_python_executable()
        assert executable == "/path/to/poetry/python"
        
        mock_run.assert_called_with(
            ["poetry", "env", "info", "--executable"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=self.temp_dir
        )
    
    @patch('shutil.which')
    def test_get_poetry_python_executable_not_available(self, mock_which):
        """Test Poetry Python executable when poetry is not available."""
        mock_which.return_value = None
        
        executable = self.service._get_poetry_python_executable()
        assert executable is None
    
    @patch('shutil.which')
    @patch('subprocess.run')
    def test_get_poetry_python_executable_command_fails(self, mock_run, mock_which):
        """Test Poetry Python executable when command fails."""
        mock_which.return_value = "/usr/local/bin/poetry"
        mock_result = Mock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result
        
        executable = self.service._get_poetry_python_executable()
        assert executable is None
    
    @patch('shutil.which')
    @patch('subprocess.run')
    def test_get_pipenv_python_executable_success(self, mock_run, mock_which):
        """Test getting Pipenv Python executable successfully."""
        mock_which.return_value = "/usr/local/bin/pipenv"
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "/path/to/pipenv/python\n"
        mock_run.return_value = mock_result
        
        executable = self.service._get_pipenv_python_executable()
        assert executable == "/path/to/pipenv/python"
    
    def test_get_venv_python_executable_unix(self):
        """Test getting venv Python executable on Unix systems."""
        venv_dir = self.temp_dir / ".venv"
        venv_dir.mkdir()
        python_executable = venv_dir / "bin" / "python"
        python_executable.parent.mkdir(parents=True, exist_ok=True)
        python_executable.touch()
        
        executable = self.service._get_venv_python_executable(venv_dir)
        assert executable == str(python_executable)
    
    def test_get_venv_python_executable_not_found(self):
        """Test venv Python executable when not found."""
        venv_dir = self.temp_dir / ".venv"
        venv_dir.mkdir()
        
        executable = self.service._get_venv_python_executable(venv_dir)
        assert executable is None
    
    @patch('subprocess.run')
    def test_get_python_version_success(self, mock_run):
        """Test getting Python version successfully."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Python 3.9.18\n"
        mock_run.return_value = mock_result
        
        version = self.service._get_python_version("/test/python")
        assert version == "3.9.18"
    
    @patch('subprocess.run')
    def test_get_python_version_failure(self, mock_run):
        """Test Python version fallback when command fails."""
        mock_run.side_effect = FileNotFoundError()
        
        version = self.service._get_python_version("/nonexistent/python")
        assert version.startswith(f"{sys.version_info.major}.{sys.version_info.minor}")


class TestEnvironmentServiceEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self, method):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Mock()
        self.feedback = Mock()
        self.service = EnvironmentService(
            project_root=self.temp_dir,
            config=self.config,
            feedback=self.feedback
        )
    
    def teardown_method(self, method):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_detect_poetry_missing_tomllib(self):
        """Test Poetry detection when tomllib/tomli is not available."""
        pyproject_content = """
[tool.poetry]
name = "test-project"
"""
        (self.temp_dir / "pyproject.toml").write_text(pyproject_content)
        
        # Mock missing tomllib and tomli
        with patch.dict('sys.modules', {'tomllib': None, 'tomli': None}):
            with patch('builtins.__import__', side_effect=ImportError()):
                env_info = self.service._detect_poetry()
                assert env_info is None
    
    def test_detect_conda_json_parsing_error(self):
        """Test Conda detection handles JSON parsing errors."""
        with patch.dict(os.environ, {"CONDA_DEFAULT_ENV": "test"}):
            with patch('subprocess.run') as mock_run:
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stdout = "invalid json"
                mock_run.return_value = mock_result
                
                with patch.object(self.service, '_get_conda_python_executable') as mock_python:
                    mock_python.return_value = None  # Should trigger JSON parsing
                    
                    env_info = self.service._detect_conda()
                    # Should still work with environment variables
                    assert env_info is not None
                    assert env_info.manager == EnvironmentManager.CONDA
    
    def test_pyenv_version_file_read_error(self):
        """Test pyenv detection handles file read errors."""
        version_file = self.temp_dir / ".python-version"
        version_file.touch()
        
        # Mock file read error
        with patch.object(Path, 'read_text', side_effect=PermissionError()):
            env_info = self.service._detect_pyenv()
            assert env_info is None
    
    def test_subprocess_timeout_handling(self):
        """Test handling of subprocess timeouts."""
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired(['cmd'], 10)):
            executable = self.service._get_poetry_python_executable()
            assert executable is None
    
    def test_all_detection_methods_fail(self):
        """Test fallback when all detection methods fail."""
        # Mock all detection methods to return None
        detection_methods = [
            '_detect_poetry', '_detect_pipenv', '_detect_conda',
            '_detect_uv', '_detect_venv', '_detect_pyenv', '_detect_system'
        ]
        
        with patch.multiple(self.service, **{method: Mock(return_value=None) for method in detection_methods}):
            env_info = self.service.detect_current_environment()
            
            # Should create system fallback
            assert env_info is not None
            assert env_info.manager == EnvironmentManager.SYSTEM


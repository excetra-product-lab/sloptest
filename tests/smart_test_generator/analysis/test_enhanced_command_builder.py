"""Tests for enhanced command builder with environment awareness."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from smart_test_generator.analysis.coverage.command_builder import build_pytest_command, CommandSpec
from smart_test_generator.services.environment_service import EnvironmentInfo, EnvironmentManager


class TestEnhancedCommandBuilder:
    """Test suite for environment-aware command builder."""
    
    def setup_method(self, method):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Mock()
        # Set up default config values
        self.config.get = Mock(side_effect=self._config_side_effect)
        self.mock_env_service = Mock()
    
    def teardown_method(self, method):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _config_side_effect(self, key, default=None):
        """Mock config.get() method with default values."""
        config_values = {
            'test_generation.coverage.runner.mode': 'python-module',
            'test_generation.coverage.runner.python': None,
            'test_generation.coverage.runner.pytest_path': 'pytest',
            'test_generation.coverage.runner.custom_cmd': [],
            'test_generation.coverage.runner.args': [],
            'test_generation.coverage.pytest_args': [],
            'test_generation.coverage.runner.cwd': None,
            'test_generation.coverage.env.propagate': True,
            'test_generation.coverage.env.extra': {},
            'test_generation.coverage.env.append_pythonpath': [],
            'environment.auto_detect': True,
            'environment.overrides.poetry.use_poetry_run': True,
            'environment.overrides.pipenv.use_pipenv_run': True,
        }
        return config_values.get(key, default)


class TestPoetryEnvironmentCommands(TestEnhancedCommandBuilder):
    """Test Poetry environment command construction."""
    
    def test_poetry_environment_uses_poetry_run(self):
        """Test that Poetry environment uses 'poetry run pytest'."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.POETRY,
            python_executable="/path/to/poetry/python",
            python_version="3.9.18"
        )
        self.mock_env_service.detect_current_environment.return_value = env_info
        
        spec = build_pytest_command(
            project_root=self.temp_dir,
            config=self.config,
            environment_service=self.mock_env_service
        )
        
        assert spec.argv[0] == "poetry"
        assert spec.argv[1] == "run"
        assert spec.argv[2] == "pytest"
        assert f"--cov={self.temp_dir}" in spec.argv
        assert "--cov-report=term-missing" in spec.argv
    
    def test_poetry_environment_with_custom_args(self):
        """Test Poetry environment with custom pytest args."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.POETRY,
            python_executable="/path/to/poetry/python",
            python_version="3.9.18"
        )
        self.mock_env_service.detect_current_environment.return_value = env_info
        
        # Add custom args via config
        self.config.get = Mock(side_effect=lambda key, default=None: {
            **{k: v for k, v in [
                ('test_generation.coverage.runner.mode', 'python-module'),
                ('test_generation.coverage.runner.args', ['--verbose']),
                ('test_generation.coverage.pytest_args', ['--tb=short']),
                ('test_generation.coverage.env.propagate', True),
                ('environment.auto_detect', True),
                ('environment.overrides.poetry.use_poetry_run', True),
            ]},
            'test_generation.coverage.runner.python': None,
            'test_generation.coverage.runner.pytest_path': 'pytest',
            'test_generation.coverage.runner.custom_cmd': [],
            'test_generation.coverage.runner.cwd': None,
            'test_generation.coverage.env.extra': {},
            'test_generation.coverage.env.append_pythonpath': [],
        }.get(key, default))
        
        spec = build_pytest_command(
            project_root=self.temp_dir,
            config=self.config,
            environment_service=self.mock_env_service
        )
        
        expected_cmd = ["poetry", "run", "pytest", f"--cov={self.temp_dir}", "--cov-report=term-missing", "--verbose", "--tb=short"]
        assert spec.argv == expected_cmd
    
    def test_poetry_environment_disabled_poetry_run(self):
        """Test Poetry environment when poetry run is disabled."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.POETRY,
            python_executable="/path/to/poetry/python",
            python_version="3.9.18"
        )
        self.mock_env_service.detect_current_environment.return_value = env_info
        
        # Disable poetry run
        override_config = {
            'environment.overrides.poetry.use_poetry_run': False,
            'environment.auto_detect': True,
        }
        self.config.get = Mock(side_effect=lambda key, default=None: 
                             override_config[key] if key in override_config else self._config_side_effect(key, default))
        
        spec = build_pytest_command(
            project_root=self.temp_dir,
            config=self.config,
            environment_service=self.mock_env_service
        )
        
        # Should fall back to python -m pytest with Poetry's Python executable
        expected_cmd = ["/path/to/poetry/python", "-m", "pytest", f"--cov={self.temp_dir}", "--cov-report=term-missing"]
        assert spec.argv == expected_cmd


class TestPipenvEnvironmentCommands(TestEnhancedCommandBuilder):
    """Test Pipenv environment command construction."""
    
    def test_pipenv_environment_uses_pipenv_run(self):
        """Test that Pipenv environment uses 'pipenv run pytest'."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.PIPENV,
            python_executable="/path/to/pipenv/python",
            python_version="3.9.18"
        )
        self.mock_env_service.detect_current_environment.return_value = env_info
        
        spec = build_pytest_command(
            project_root=self.temp_dir,
            config=self.config,
            environment_service=self.mock_env_service
        )
        
        assert spec.argv[0] == "pipenv"
        assert spec.argv[1] == "run"
        assert spec.argv[2] == "pytest"
    
    def test_pipenv_environment_disabled_pipenv_run(self):
        """Test Pipenv environment when pipenv run is disabled."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.PIPENV,
            python_executable="/path/to/pipenv/python",
            python_version="3.9.18"
        )
        self.mock_env_service.detect_current_environment.return_value = env_info
        
        # Disable pipenv run
        override_config = {
            'environment.overrides.pipenv.use_pipenv_run': False,
            'environment.auto_detect': True,
        }
        self.config.get = Mock(side_effect=lambda key, default=None: 
                             override_config[key] if key in override_config else self._config_side_effect(key, default))
        
        spec = build_pytest_command(
            project_root=self.temp_dir,
            config=self.config,
            environment_service=self.mock_env_service
        )
        
        # Should fall back to python -m pytest with Pipenv's Python executable
        expected_cmd = ["/path/to/pipenv/python", "-m", "pytest", f"--cov={self.temp_dir}", "--cov-report=term-missing"]
        assert spec.argv == expected_cmd


class TestOtherEnvironmentCommands(TestEnhancedCommandBuilder):
    """Test command construction for other environment managers."""
    
    def test_conda_environment_uses_python_executable(self):
        """Test that Conda environment uses its Python executable directly."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.CONDA,
            python_executable="/opt/conda/envs/test/bin/python",
            python_version="3.9.18"
        )
        self.mock_env_service.detect_current_environment.return_value = env_info
        
        spec = build_pytest_command(
            project_root=self.temp_dir,
            config=self.config,
            environment_service=self.mock_env_service
        )
        
        expected_cmd = ["/opt/conda/envs/test/bin/python", "-m", "pytest", f"--cov={self.temp_dir}", "--cov-report=term-missing"]
        assert spec.argv == expected_cmd
    
    def test_uv_environment_uses_python_executable(self):
        """Test that uv environment uses its Python executable directly."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.UV,
            python_executable="/path/to/uv/python",
            python_version="3.9.18"
        )
        self.mock_env_service.detect_current_environment.return_value = env_info
        
        spec = build_pytest_command(
            project_root=self.temp_dir,
            config=self.config,
            environment_service=self.mock_env_service
        )
        
        expected_cmd = ["/path/to/uv/python", "-m", "pytest", f"--cov={self.temp_dir}", "--cov-report=term-missing"]
        assert spec.argv == expected_cmd
    
    def test_venv_environment_uses_python_executable(self):
        """Test that venv environment uses its Python executable directly."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.VENV,
            python_executable="/path/to/venv/bin/python",
            python_version="3.9.18"
        )
        self.mock_env_service.detect_current_environment.return_value = env_info
        
        spec = build_pytest_command(
            project_root=self.temp_dir,
            config=self.config,
            environment_service=self.mock_env_service
        )
        
        expected_cmd = ["/path/to/venv/bin/python", "-m", "pytest", f"--cov={self.temp_dir}", "--cov-report=term-missing"]
        assert spec.argv == expected_cmd
    
    def test_system_environment_uses_python_executable(self):
        """Test that system environment uses its Python executable directly."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.SYSTEM,
            python_executable=sys.executable,
            python_version="3.9.18"
        )
        self.mock_env_service.detect_current_environment.return_value = env_info
        
        spec = build_pytest_command(
            project_root=self.temp_dir,
            config=self.config,
            environment_service=self.mock_env_service
        )
        
        expected_cmd = [sys.executable, "-m", "pytest", f"--cov={self.temp_dir}", "--cov-report=term-missing"]
        assert spec.argv == expected_cmd


class TestFallbackBehavior(TestEnhancedCommandBuilder):
    """Test fallback behavior when environment detection fails or is disabled."""
    
    def test_no_environment_service_fallback(self):
        """Test fallback when no environment service is provided."""
        spec = build_pytest_command(
            project_root=self.temp_dir,
            config=self.config,
            environment_service=None
        )
        
        # Should use sys.executable as fallback
        expected_cmd = [sys.executable, "-m", "pytest", f"--cov={self.temp_dir}", "--cov-report=term-missing"]
        assert spec.argv == expected_cmd
    
    def test_environment_detection_exception_fallback(self):
        """Test fallback when environment detection raises exception."""
        self.mock_env_service.detect_current_environment.side_effect = Exception("Detection failed")
        
        spec = build_pytest_command(
            project_root=self.temp_dir,
            config=self.config,
            environment_service=self.mock_env_service
        )
        
        # Should fall back to original behavior
        expected_cmd = [sys.executable, "-m", "pytest", f"--cov={self.temp_dir}", "--cov-report=term-missing"]
        assert spec.argv == expected_cmd
    
    def test_auto_detect_disabled_fallback(self):
        """Test fallback when auto detection is disabled."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.POETRY,
            python_executable="/path/to/poetry/python",
            python_version="3.9.18"
        )
        self.mock_env_service.detect_current_environment.return_value = env_info
        
        # Disable auto detection
        override_config = {
            'environment.auto_detect': False,
        }
        self.config.get = Mock(side_effect=lambda key, default=None: 
                             override_config[key] if key in override_config else self._config_side_effect(key, default))
        
        spec = build_pytest_command(
            project_root=self.temp_dir,
            config=self.config,
            environment_service=self.mock_env_service
        )
        
        # Should use sys.executable instead of environment-specific executable
        expected_cmd = [sys.executable, "-m", "pytest", f"--cov={self.temp_dir}", "--cov-report=term-missing"]
        assert spec.argv == expected_cmd


class TestRunnerModeOverrides(TestEnhancedCommandBuilder):
    """Test that runner mode overrides work with environment detection."""
    
    def test_pytest_path_mode_with_environment(self):
        """Test pytest-path mode works with environment detection."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.POETRY,
            python_executable="/path/to/poetry/python",
            python_version="3.9.18"
        )
        self.mock_env_service.detect_current_environment.return_value = env_info
        
        # Set runner mode to pytest-path
        self.config.get = Mock(side_effect=lambda key, default=None: {
            'test_generation.coverage.runner.mode': 'pytest-path',
            'test_generation.coverage.runner.pytest_path': '/custom/pytest',
            'environment.auto_detect': True,
            'environment.overrides.poetry.use_poetry_run': True,
        }.get(key, default) or self._config_side_effect(key, default))
        
        spec = build_pytest_command(
            project_root=self.temp_dir,
            config=self.config,
            environment_service=self.mock_env_service
        )
        
        # Should still use Poetry run since it's preferred for Poetry environment
        assert spec.argv[0] == "poetry"
        assert spec.argv[1] == "run"
        assert spec.argv[2] == "pytest"
    
    def test_custom_mode_with_environment(self):
        """Test custom mode works with environment detection."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.POETRY,
            python_executable="/path/to/poetry/python",
            python_version="3.9.18"
        )
        self.mock_env_service.detect_current_environment.return_value = env_info
        
        # Set runner mode to custom
        self.config.get = Mock(side_effect=lambda key, default=None: {
            'test_generation.coverage.runner.mode': 'custom',
            'test_generation.coverage.runner.custom_cmd': ['custom', 'test', 'runner'],
            'environment.auto_detect': True,
            'environment.overrides.poetry.use_poetry_run': True,
        }.get(key, default) or self._config_side_effect(key, default))
        
        spec = build_pytest_command(
            project_root=self.temp_dir,
            config=self.config,
            environment_service=self.mock_env_service
        )
        
        # Should still prefer Poetry run for Poetry environment
        assert spec.argv[0] == "poetry"
        assert spec.argv[1] == "run"
        assert spec.argv[2] == "pytest"
    
    def test_custom_mode_fallback_when_no_environment_override(self):
        """Test custom mode falls back when no environment override is applicable."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.SYSTEM,
            python_executable=sys.executable,
            python_version="3.9.18"
        )
        self.mock_env_service.detect_current_environment.return_value = env_info
        
        # Set runner mode to custom
        self.config.get = Mock(side_effect=lambda key, default=None: {
            'test_generation.coverage.runner.mode': 'custom',
            'test_generation.coverage.runner.custom_cmd': ['custom', 'test', 'runner'],
            'environment.auto_detect': True,
        }.get(key, default) or self._config_side_effect(key, default))
        
        spec = build_pytest_command(
            project_root=self.temp_dir,
            config=self.config,
            environment_service=self.mock_env_service
        )
        
        # Should use custom command since no environment-specific override applies
        expected_start = ['custom', 'test', 'runner', f"--cov={self.temp_dir}", "--cov-report=term-missing"]
        assert spec.argv[:len(expected_start)] == expected_start


class TestEnvironmentConfiguration(TestEnhancedCommandBuilder):
    """Test environment configuration and variable handling."""
    
    def test_working_directory_configuration(self):
        """Test that working directory configuration is respected."""
        custom_cwd = self.temp_dir / "subdir"
        custom_cwd.mkdir()
        
        self.config.get = Mock(side_effect=lambda key, default=None: {
            'test_generation.coverage.runner.cwd': str(custom_cwd),
        }.get(key, default) or self._config_side_effect(key, default))
        
        spec = build_pytest_command(
            project_root=self.temp_dir,
            config=self.config,
            environment_service=None
        )
        
        assert spec.cwd == custom_cwd
    
    def test_environment_variable_propagation(self):
        """Test environment variable propagation."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            spec = build_pytest_command(
                project_root=self.temp_dir,
                config=self.config,
                environment_service=None
            )
            
            assert "TEST_VAR" in spec.env
            assert spec.env["TEST_VAR"] == "test_value"
    
    def test_extra_environment_variables(self):
        """Test extra environment variables configuration."""
        self.config.get = Mock(side_effect=lambda key, default=None: {
            'test_generation.coverage.env.extra': {"CUSTOM_VAR": "custom_value"},
        }.get(key, default) or self._config_side_effect(key, default))
        
        spec = build_pytest_command(
            project_root=self.temp_dir,
            config=self.config,
            environment_service=None
        )
        
        assert spec.env["CUSTOM_VAR"] == "custom_value"
    
    def test_pythonpath_append_configuration(self):
        """Test PYTHONPATH append configuration."""
        additional_path = str(self.temp_dir / "additional")
        
        self.config.get = Mock(side_effect=lambda key, default=None: {
            'test_generation.coverage.env.append_pythonpath': [additional_path],
        }.get(key, default) or self._config_side_effect(key, default))
        
        with patch.dict(os.environ, {"PYTHONPATH": "/existing/path"}):
            spec = build_pytest_command(
                project_root=self.temp_dir,
                config=self.config,
                environment_service=None
            )
            
            expected_pythonpath = f"/existing/path{os.pathsep}{additional_path}"
            assert spec.env["PYTHONPATH"] == expected_pythonpath
    
    def test_pythonpath_append_no_existing(self):
        """Test PYTHONPATH append when no existing PYTHONPATH."""
        additional_path = str(self.temp_dir / "additional")
        
        self.config.get = Mock(side_effect=lambda key, default=None: {
            'test_generation.coverage.env.append_pythonpath': [additional_path],
        }.get(key, default) or self._config_side_effect(key, default))
        
        # Ensure PYTHONPATH is not set
        with patch.dict(os.environ, {}, clear=True):
            spec = build_pytest_command(
                project_root=self.temp_dir,
                config=self.config,
                environment_service=None
            )
            
            assert spec.env["PYTHONPATH"] == additional_path


class TestIntegrationScenarios(TestEnhancedCommandBuilder):
    """Test integration scenarios combining multiple features."""
    
    def test_poetry_with_all_configurations(self):
        """Test Poetry environment with all configuration options."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.POETRY,
            python_executable="/path/to/poetry/python",
            python_version="3.9.18"
        )
        self.mock_env_service.detect_current_environment.return_value = env_info
        
        custom_cwd = self.temp_dir / "subdir"
        custom_cwd.mkdir()
        
        self.config.get = Mock(side_effect=lambda key, default=None: {
            'test_generation.coverage.runner.args': ['--verbose', '--tb=short'],
            'test_generation.coverage.pytest_args': ['--maxfail=1'],
            'test_generation.coverage.runner.cwd': str(custom_cwd),
            'test_generation.coverage.env.extra': {"PYTEST_CURRENT_TEST": "1"},
            'test_generation.coverage.env.append_pythonpath': [str(self.temp_dir / "src")],
            'environment.auto_detect': True,
            'environment.overrides.poetry.use_poetry_run': True,
        }.get(key, default) or self._config_side_effect(key, default))
        
        spec = build_pytest_command(
            project_root=self.temp_dir,
            config=self.config,
            environment_service=self.mock_env_service
        )
        
        # Verify command structure
        expected_cmd = [
            "poetry", "run", "pytest",
            f"--cov={self.temp_dir}", "--cov-report=term-missing",
            "--verbose", "--tb=short",
            "--maxfail=1"
        ]
        assert spec.argv == expected_cmd
        
        # Verify working directory
        assert spec.cwd == custom_cwd
        
        # Verify environment variables
        assert spec.env["PYTEST_CURRENT_TEST"] == "1"
        assert str(self.temp_dir / "src") in spec.env["PYTHONPATH"]
    
    def test_fallback_with_custom_python_executable(self):
        """Test fallback behavior with custom Python executable configuration."""
        custom_python = "/custom/python3.10"
        
        override_config = {
            'test_generation.coverage.runner.python': custom_python,
            'environment.auto_detect': False,
        }
        self.config.get = Mock(side_effect=lambda key, default=None: 
                             override_config[key] if key in override_config else self._config_side_effect(key, default))
        
        spec = build_pytest_command(
            project_root=self.temp_dir,
            config=self.config,
            environment_service=self.mock_env_service
        )
        
        # Should use custom Python executable
        expected_cmd = [custom_python, "-m", "pytest", f"--cov={self.temp_dir}", "--cov-report=term-missing"]
        assert spec.argv == expected_cmd

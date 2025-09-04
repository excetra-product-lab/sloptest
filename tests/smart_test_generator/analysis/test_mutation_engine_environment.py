"""Tests for MutationTestingEngine environment integration."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from smart_test_generator.analysis.mutation_engine import MutationTestingEngine
from smart_test_generator.services.environment_service import EnvironmentInfo, EnvironmentManager


class TestMutationEngineEnvironmentIntegration:
    """Test environment integration in MutationTestingEngine."""
    
    def setup_method(self, method):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Mock()
        self.config.get = Mock(side_effect=self._config_side_effect)
        self.environment_service = Mock()
        self.engine = MutationTestingEngine(
            environment_service=self.environment_service,
            config=self.config
        )
    
    def teardown_method(self, method):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _config_side_effect(self, key, default=None):
        """Mock config.get() method with default values."""
        config_values = {
            'environment.auto_detect': True,
            'environment.overrides.poetry.use_poetry_run': True,
            'environment.overrides.pipenv.use_pipenv_run': True,
            'test_generation.coverage.env.propagate': True,
            'test_generation.coverage.env.extra': {},
            'test_generation.coverage.env.append_pythonpath': [],
        }
        return config_values.get(key, default)
    
    def test_build_test_command_poetry_environment(self):
        """Test command building for Poetry environment."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.POETRY,
            python_executable="/path/to/poetry/python",
            python_version="3.9.18"
        )
        self.environment_service.detect_current_environment.return_value = env_info
        
        test_files = ["test_example.py"]
        cmd = self.engine._build_test_command(test_files, self.temp_dir)
        
        expected = ["poetry", "run", "pytest", "test_example.py", "--tb=short", "-q", "--no-header"]
        assert cmd == expected
    
    def test_build_test_command_pipenv_environment(self):
        """Test command building for Pipenv environment."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.PIPENV,
            python_executable="/path/to/pipenv/python",
            python_version="3.9.18"
        )
        self.environment_service.detect_current_environment.return_value = env_info
        
        test_files = ["test_example.py"]
        cmd = self.engine._build_test_command(test_files, self.temp_dir)
        
        expected = ["pipenv", "run", "pytest", "test_example.py", "--tb=short", "-q", "--no-header"]
        assert cmd == expected
    
    def test_build_test_command_conda_environment(self):
        """Test command building for Conda environment."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.CONDA,
            python_executable="/opt/conda/envs/test/bin/python",
            python_version="3.9.18"
        )
        self.environment_service.detect_current_environment.return_value = env_info
        
        test_files = ["test_example.py"]
        cmd = self.engine._build_test_command(test_files, self.temp_dir)
        
        expected = ["/opt/conda/envs/test/bin/python", "-m", "pytest", "test_example.py", "--tb=short", "-q", "--no-header"]
        assert cmd == expected
    
    def test_build_test_command_uv_environment(self):
        """Test command building for uv environment."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.UV,
            python_executable="/path/to/uv/python",
            python_version="3.9.18"
        )
        self.environment_service.detect_current_environment.return_value = env_info
        
        test_files = ["test_example.py"]
        cmd = self.engine._build_test_command(test_files, self.temp_dir)
        
        expected = ["/path/to/uv/python", "-m", "pytest", "test_example.py", "--tb=short", "-q", "--no-header"]
        assert cmd == expected
    
    def test_build_test_command_venv_environment(self):
        """Test command building for venv environment."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.VENV,
            python_executable="/path/to/venv/bin/python",
            python_version="3.9.18"
        )
        self.environment_service.detect_current_environment.return_value = env_info
        
        test_files = ["test_example.py"]
        cmd = self.engine._build_test_command(test_files, self.temp_dir)
        
        expected = ["/path/to/venv/bin/python", "-m", "pytest", "test_example.py", "--tb=short", "-q", "--no-header"]
        assert cmd == expected
    
    def test_build_test_command_system_environment(self):
        """Test command building for system environment."""
        import sys
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.SYSTEM,
            python_executable=sys.executable,
            python_version="3.9.18"
        )
        self.environment_service.detect_current_environment.return_value = env_info
        
        test_files = ["test_example.py"]
        cmd = self.engine._build_test_command(test_files, self.temp_dir)
        
        expected = [sys.executable, "-m", "pytest", "test_example.py", "--tb=short", "-q", "--no-header"]
        assert cmd == expected
    
    def test_build_test_command_no_environment_service(self):
        """Test command building without environment service."""
        engine = MutationTestingEngine(environment_service=None, config=self.config)
        
        test_files = ["test_example.py"]
        cmd = engine._build_test_command(test_files, self.temp_dir)
        
        expected = ["python", "-m", "pytest", "test_example.py", "--tb=short", "-q", "--no-header"]
        assert cmd == expected
    
    def test_build_test_command_environment_detection_exception(self):
        """Test command building when environment detection fails."""
        self.environment_service.detect_current_environment.side_effect = Exception("Detection failed")
        
        test_files = ["test_example.py"]
        cmd = self.engine._build_test_command(test_files, self.temp_dir)
        
        # Should fall back to default
        expected = ["python", "-m", "pytest", "test_example.py", "--tb=short", "-q", "--no-header"]
        assert cmd == expected
    
    def test_build_test_command_auto_detect_disabled(self):
        """Test command building when auto detection is disabled."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.POETRY,
            python_executable="/path/to/poetry/python",
            python_version="3.9.18"
        )
        self.environment_service.detect_current_environment.return_value = env_info
        
        # Disable auto detection
        def mock_config_get(key, default=None):
            if key == 'environment.auto_detect':
                return False
            return self._config_side_effect(key, default)
        
        self.config.get = Mock(side_effect=mock_config_get)
        
        test_files = ["test_example.py"]
        cmd = self.engine._build_test_command(test_files, self.temp_dir)
        
        # Should fall back to default python
        expected = ["python", "-m", "pytest", "test_example.py", "--tb=short", "-q", "--no-header"]
        assert cmd == expected
    
    def test_build_test_command_poetry_run_disabled(self):
        """Test command building when Poetry run is disabled."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.POETRY,
            python_executable="/path/to/poetry/python",
            python_version="3.9.18"
        )
        self.environment_service.detect_current_environment.return_value = env_info
        
        # Disable poetry run
        def mock_config_get(key, default=None):
            if key == 'environment.overrides.poetry.use_poetry_run':
                return False
            return self._config_side_effect(key, default)
        
        self.config.get = Mock(side_effect=mock_config_get)
        
        test_files = ["test_example.py"]
        cmd = self.engine._build_test_command(test_files, self.temp_dir)
        
        # Should use Poetry's Python executable directly
        expected = ["/path/to/poetry/python", "-m", "pytest", "test_example.py", "--tb=short", "-q", "--no-header"]
        assert cmd == expected
    
    def test_build_test_command_pipenv_run_disabled(self):
        """Test command building when Pipenv run is disabled."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.PIPENV,
            python_executable="/path/to/pipenv/python",
            python_version="3.9.18"
        )
        self.environment_service.detect_current_environment.return_value = env_info
        
        # Disable pipenv run
        def mock_config_get(key, default=None):
            if key == 'environment.overrides.pipenv.use_pipenv_run':
                return False
            return self._config_side_effect(key, default)
        
        self.config.get = Mock(side_effect=mock_config_get)
        
        test_files = ["test_example.py"]
        cmd = self.engine._build_test_command(test_files, self.temp_dir)
        
        # Should use Pipenv's Python executable directly
        expected = ["/path/to/pipenv/python", "-m", "pytest", "test_example.py", "--tb=short", "-q", "--no-header"]
        assert cmd == expected
    
    def test_build_test_command_multiple_test_files(self):
        """Test command building with multiple test files."""
        env_info = EnvironmentInfo(
            manager=EnvironmentManager.POETRY,
            python_executable="/path/to/poetry/python",
            python_version="3.9.18"
        )
        self.environment_service.detect_current_environment.return_value = env_info
        
        test_files = ["test_example1.py", "test_example2.py", "test_example3.py"]
        cmd = self.engine._build_test_command(test_files, self.temp_dir)
        
        expected = [
            "poetry", "run", "pytest", 
            "test_example1.py", "test_example2.py", "test_example3.py",
            "--tb=short", "-q", "--no-header"
        ]
        assert cmd == expected


class TestMutationEngineTestEnvironment:
    """Test environment variable handling in MutationTestingEngine."""
    
    def setup_method(self, method):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Mock()
        self.config.get = Mock(side_effect=self._config_side_effect)
        self.engine = MutationTestingEngine(config=self.config)
    
    def teardown_method(self, method):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _config_side_effect(self, key, default=None):
        """Mock config.get() method with default values."""
        config_values = {
            'test_generation.coverage.env.propagate': True,
            'test_generation.coverage.env.extra': {},
            'test_generation.coverage.env.append_pythonpath': [],
        }
        return config_values.get(key, default)
    
    def test_get_test_environment_propagate_enabled(self):
        """Test environment propagation when enabled."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            env = self.engine._get_test_environment()
            
            assert env is not None
            assert "TEST_VAR" in env
            assert env["TEST_VAR"] == "test_value"
    
    def test_get_test_environment_propagate_disabled(self):
        """Test environment when propagation is disabled."""
        def mock_config_get(key, default=None):
            if key == 'test_generation.coverage.env.propagate':
                return False
            return self._config_side_effect(key, default)
        
        self.config.get = Mock(side_effect=mock_config_get)
        
        env = self.engine._get_test_environment()
        assert env is None  # Should use subprocess default
    
    def test_get_test_environment_with_extra_variables(self):
        """Test environment with extra variables."""
        self.config.get = Mock(side_effect=lambda key, default=None: {
            'test_generation.coverage.env.propagate': True,
            'test_generation.coverage.env.extra': {"CUSTOM_VAR": "custom_value"},
        }.get(key, default) or self._config_side_effect(key, default))
        
        env = self.engine._get_test_environment()
        
        assert env is not None
        assert env["CUSTOM_VAR"] == "custom_value"
    
    def test_get_test_environment_with_pythonpath_append(self):
        """Test environment with PYTHONPATH append."""
        additional_path = str(self.temp_dir / "additional")
        
        self.config.get = Mock(side_effect=lambda key, default=None: {
            'test_generation.coverage.env.propagate': True,
            'test_generation.coverage.env.append_pythonpath': [additional_path],
        }.get(key, default) or self._config_side_effect(key, default))
        
        with patch.dict(os.environ, {"PYTHONPATH": "/existing/path"}):
            env = self.engine._get_test_environment()
            
            assert env is not None
            expected_pythonpath = f"/existing/path{os.pathsep}{additional_path}"
            assert env["PYTHONPATH"] == expected_pythonpath
    
    def test_get_test_environment_pythonpath_no_existing(self):
        """Test environment with PYTHONPATH append when no existing PYTHONPATH."""
        additional_path = str(self.temp_dir / "additional")
        
        self.config.get = Mock(side_effect=lambda key, default=None: {
            'test_generation.coverage.env.propagate': True,
            'test_generation.coverage.env.append_pythonpath': [additional_path],
        }.get(key, default) or self._config_side_effect(key, default))
        
        # Ensure PYTHONPATH is not in environment
        with patch.dict(os.environ, {}, clear=True):
            env = self.engine._get_test_environment()
            
            assert env is not None
            assert env["PYTHONPATH"] == additional_path
    
    def test_get_test_environment_no_config(self):
        """Test environment when no config is provided."""
        engine = MutationTestingEngine(config=None)
        env = engine._get_test_environment()
        assert env is None


class TestMutationEngineBackwardCompatibility:
    """Test backward compatibility of MutationTestingEngine."""
    
    def test_initialization_backward_compatibility(self):
        """Test that old initialization style still works."""
        # Old style - should work without environment service
        engine = MutationTestingEngine(timeout=60)
        
        assert engine.timeout == 60
        assert engine.environment_service is None
        assert engine.config is None
        assert len(engine.operators) > 0  # Default operators should be loaded
    
    def test_old_initialization_with_operators(self):
        """Test old initialization style with custom operators."""
        from smart_test_generator.analysis.mutation_engine import ArithmeticOperatorMutator
        
        custom_operators = [ArithmeticOperatorMutator()]
        engine = MutationTestingEngine(operators=custom_operators, timeout=45)
        
        assert engine.timeout == 45
        assert len(engine.operators) == 1
        assert isinstance(engine.operators[0], ArithmeticOperatorMutator)
    
    def test_build_test_command_fallback_without_config(self):
        """Test command building falls back gracefully without config."""
        engine = MutationTestingEngine(environment_service=None, config=None)
        
        test_files = ["test_example.py"]
        cmd = engine._build_test_command(test_files, Path("/tmp"))
        
        # Should use default fallback
        expected = ["python", "-m", "pytest", "test_example.py", "--tb=short", "-q", "--no-header"]
        assert cmd == expected
    
    def test_get_test_environment_fallback_without_config(self):
        """Test environment variable handling falls back without config."""
        engine = MutationTestingEngine(config=None)
        
        env = engine._get_test_environment()
        assert env is None  # Should use subprocess default

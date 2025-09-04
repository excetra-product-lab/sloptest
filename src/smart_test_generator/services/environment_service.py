"""Environment detection and management service."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum

from smart_test_generator.services.base_service import BaseService


class EnvironmentManager(str, Enum):
    """Supported environment managers."""
    POETRY = "poetry"
    PIPENV = "pipenv"
    CONDA = "conda"
    UV = "uv"
    VENV = "venv"
    PYENV = "pyenv"
    SYSTEM = "system"


@dataclass
class EnvironmentInfo:
    """Information about the detected environment."""
    manager: EnvironmentManager
    python_executable: str
    python_version: str
    virtual_env_path: Optional[str] = None
    env_name: Optional[str] = None
    project_file: Optional[str] = None
    lockfile: Optional[str] = None
    is_virtual_env: bool = False
    detected_files: List[str] = None
    
    def __post_init__(self):
        if self.detected_files is None:
            self.detected_files = []


class EnvironmentService(BaseService):
    """Service for detecting and managing Python environments."""
    
    def __init__(self, project_root: Path, config, feedback=None):
        super().__init__(project_root, config, feedback)
        self._cache: Optional[EnvironmentInfo] = None
    
    def detect_current_environment(self, force_refresh: bool = False) -> EnvironmentInfo:
        """
        Detect the current Python environment.
        
        Args:
            force_refresh: If True, bypass cache and re-detect environment
            
        Returns:
            EnvironmentInfo with detected environment details
        """
        if self._cache is not None and not force_refresh:
            return self._cache
        
        self._log_debug("Starting environment detection")
        
        # Try detection methods in priority order
        detection_methods = [
            self._detect_poetry,
            self._detect_pipenv,
            self._detect_conda,
            self._detect_uv,
            self._detect_venv,
            self._detect_pyenv,
            self._detect_system,
        ]
        
        for method in detection_methods:
            env_info = method()
            if env_info is not None:
                self._cache = env_info
                self._log_debug(f"Detected environment: {env_info.manager.value}")
                return env_info
        
        # Fallback to system Python
        env_info = self._create_system_fallback()
        self._cache = env_info
        self._log_warning("Could not detect specific environment manager, using system Python")
        return env_info
    
    def _detect_poetry(self) -> Optional[EnvironmentInfo]:
        """Detect Poetry environment."""
        pyproject_path = self.project_root / "pyproject.toml"
        poetry_lock_path = self.project_root / "poetry.lock"
        
        if not pyproject_path.exists():
            return None
        
        try:
            import tomllib
        except ImportError:
            # Python < 3.11, try tomli
            try:
                import tomli as tomllib
            except ImportError:
                self._log_debug("tomllib/tomli not available, cannot parse pyproject.toml")
                return None
        
        try:
            with open(pyproject_path, "rb") as f:
                pyproject_data = tomllib.load(f)
            
            # Check if this is a Poetry project
            if "tool" not in pyproject_data or "poetry" not in pyproject_data["tool"]:
                return None
            
            detected_files = [str(pyproject_path)]
            if poetry_lock_path.exists():
                detected_files.append(str(poetry_lock_path))
            
            # Try to get Poetry's Python executable
            python_exec = self._get_poetry_python_executable()
            if python_exec is None:
                python_exec = sys.executable
            
            # Check if we're in a Poetry virtual environment
            virtual_env_path = self._get_poetry_venv_path()
            is_virtual_env = virtual_env_path is not None
            
            return EnvironmentInfo(
                manager=EnvironmentManager.POETRY,
                python_executable=python_exec,
                python_version=self._get_python_version(python_exec),
                virtual_env_path=virtual_env_path,
                project_file=str(pyproject_path),
                lockfile=str(poetry_lock_path) if poetry_lock_path.exists() else None,
                is_virtual_env=is_virtual_env,
                detected_files=detected_files
            )
            
        except Exception as e:
            self._log_debug(f"Error parsing pyproject.toml: {e}")
            return None
    
    def _detect_pipenv(self) -> Optional[EnvironmentInfo]:
        """Detect Pipenv environment."""
        pipfile_path = self.project_root / "Pipfile"
        pipfile_lock_path = self.project_root / "Pipfile.lock"
        
        if not pipfile_path.exists():
            return None
        
        detected_files = [str(pipfile_path)]
        if pipfile_lock_path.exists():
            detected_files.append(str(pipfile_lock_path))
        
        # Try to get Pipenv's Python executable
        python_exec = self._get_pipenv_python_executable()
        if python_exec is None:
            python_exec = sys.executable
        
        # Check if we're in a Pipenv virtual environment
        virtual_env_path = self._get_pipenv_venv_path()
        is_virtual_env = virtual_env_path is not None
        
        return EnvironmentInfo(
            manager=EnvironmentManager.PIPENV,
            python_executable=python_exec,
            python_version=self._get_python_version(python_exec),
            virtual_env_path=virtual_env_path,
            project_file=str(pipfile_path),
            lockfile=str(pipfile_lock_path) if pipfile_lock_path.exists() else None,
            is_virtual_env=is_virtual_env,
            detected_files=detected_files
        )
    
    def _detect_conda(self) -> Optional[EnvironmentInfo]:
        """Detect Conda environment."""
        # Check for environment.yml file
        env_yml_path = self.project_root / "environment.yml"
        detected_files = []
        
        if env_yml_path.exists():
            detected_files.append(str(env_yml_path))
        
        # Check for conda environment variables
        conda_default_env = os.environ.get("CONDA_DEFAULT_ENV")
        conda_prefix = os.environ.get("CONDA_PREFIX")
        
        # If neither environment file nor conda variables exist, not a conda environment
        if not env_yml_path.exists() and not conda_default_env and not conda_prefix:
            return None
        
        # Get conda environment information
        env_name = conda_default_env or "base"
        virtual_env_path = conda_prefix
        
        # Try to get conda's Python executable
        python_exec = self._get_conda_python_executable()
        if python_exec is None:
            python_exec = sys.executable
        
        is_virtual_env = conda_prefix is not None and conda_prefix != sys.base_prefix
        
        return EnvironmentInfo(
            manager=EnvironmentManager.CONDA,
            python_executable=python_exec,
            python_version=self._get_python_version(python_exec),
            virtual_env_path=virtual_env_path,
            env_name=env_name,
            project_file=str(env_yml_path) if env_yml_path.exists() else None,
            is_virtual_env=is_virtual_env,
            detected_files=detected_files
        )
    
    def _detect_uv(self) -> Optional[EnvironmentInfo]:
        """Detect uv environment."""
        uv_lock_path = self.project_root / "uv.lock"
        pyproject_path = self.project_root / "pyproject.toml"
        
        # uv requires uv.lock file
        if not uv_lock_path.exists():
            return None
        
        detected_files = [str(uv_lock_path)]
        
        # Check if pyproject.toml has uv configuration
        has_uv_config = False
        if pyproject_path.exists():
            detected_files.append(str(pyproject_path))
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError:
                    self._log_debug("tomllib/tomli not available, cannot parse pyproject.toml")
                    # Continue anyway since we have uv.lock
                    tomllib = None
            
            if tomllib is not None:
                try:
                    with open(pyproject_path, "rb") as f:
                        pyproject_data = tomllib.load(f)
                    
                    # Check for uv-related configuration
                    has_uv_config = (
                        "tool" in pyproject_data and 
                        ("uv" in pyproject_data["tool"] or 
                         "project" in pyproject_data)  # uv can use standard project metadata
                    )
                except Exception as e:
                    self._log_debug(f"Error parsing pyproject.toml for uv: {e}")
        
        # Try to get uv's Python executable
        python_exec = self._get_uv_python_executable()
        if python_exec is None:
            python_exec = sys.executable
        
        # Check if we're in a uv virtual environment
        virtual_env_path = self._get_uv_venv_path()
        is_virtual_env = virtual_env_path is not None
        
        return EnvironmentInfo(
            manager=EnvironmentManager.UV,
            python_executable=python_exec,
            python_version=self._get_python_version(python_exec),
            virtual_env_path=virtual_env_path,
            project_file=str(pyproject_path) if pyproject_path.exists() else None,
            lockfile=str(uv_lock_path),
            is_virtual_env=is_virtual_env,
            detected_files=detected_files
        )
    
    def _detect_venv(self) -> Optional[EnvironmentInfo]:
        """Detect venv/virtualenv environment."""
        # Check if we're currently in a virtual environment
        virtual_env = os.environ.get("VIRTUAL_ENV")
        
        if virtual_env:
            # We're in an activated virtual environment
            virtual_env_path = Path(virtual_env)
            pyvenv_cfg = virtual_env_path / "pyvenv.cfg"
            
            detected_files = []
            if pyvenv_cfg.exists():
                detected_files.append(str(pyvenv_cfg))
            
            # Get Python executable from virtual environment
            python_exec = self._get_venv_python_executable(virtual_env_path)
            if python_exec is None:
                python_exec = sys.executable
            
            return EnvironmentInfo(
                manager=EnvironmentManager.VENV,
                python_executable=python_exec,
                python_version=self._get_python_version(python_exec),
                virtual_env_path=str(virtual_env_path),
                is_virtual_env=True,
                detected_files=detected_files
            )
        
        # Check for common venv directories in project
        common_venv_names = [".venv", "venv", "env", "virtualenv"]
        for venv_name in common_venv_names:
            venv_path = self.project_root / venv_name
            if venv_path.exists():
                pyvenv_cfg = venv_path / "pyvenv.cfg"
                if pyvenv_cfg.exists():
                    # Found a virtual environment directory with pyvenv.cfg
                    detected_files = [str(pyvenv_cfg)]
                    
                    python_exec = self._get_venv_python_executable(venv_path)
                    if python_exec is None:
                        python_exec = sys.executable
                    
                    return EnvironmentInfo(
                        manager=EnvironmentManager.VENV,
                        python_executable=python_exec,
                        python_version=self._get_python_version(python_exec),
                        virtual_env_path=str(venv_path),
                        is_virtual_env=True,
                        detected_files=detected_files
                    )
        
        return None
    
    def _detect_pyenv(self) -> Optional[EnvironmentInfo]:
        """Detect pyenv environment."""
        # Look for .python-version file
        python_version_files = []
        
        # Check in project root and parent directories
        current_path = self.project_root
        for _ in range(5):  # Check up to 5 levels up
            version_file = current_path / ".python-version"
            if version_file.exists():
                python_version_files.append(version_file)
                break
            parent = current_path.parent
            if parent == current_path:  # Reached filesystem root
                break
            current_path = parent
        
        if not python_version_files:
            return None
        
        version_file = python_version_files[0]
        detected_files = [str(version_file)]
        
        try:
            python_version = version_file.read_text().strip()
        except Exception:
            return None
        
        # Try to get pyenv's Python executable
        python_exec = self._get_pyenv_python_executable(python_version)
        if python_exec is None:
            python_exec = sys.executable
        
        # Pyenv environments are typically virtual environments
        is_virtual_env = python_exec != sys.base_prefix
        
        return EnvironmentInfo(
            manager=EnvironmentManager.PYENV,
            python_executable=python_exec,
            python_version=python_version,
            is_virtual_env=is_virtual_env,
            detected_files=detected_files
        )
    
    def _detect_system(self) -> Optional[EnvironmentInfo]:
        """Detect system Python installation."""
        # This is the fallback - always succeeds
        return EnvironmentInfo(
            manager=EnvironmentManager.SYSTEM,
            python_executable=sys.executable,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            is_virtual_env=hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix),
            detected_files=[]
        )
    
    def _create_system_fallback(self) -> EnvironmentInfo:
        """Create fallback environment info using system Python."""
        return EnvironmentInfo(
            manager=EnvironmentManager.SYSTEM,
            python_executable=sys.executable,
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            is_virtual_env=False
        )
    
    def get_current_python_executable(self) -> str:
        """Get the Python executable for the current environment."""
        env_info = self.detect_current_environment()
        return env_info.python_executable
    
    def get_virtual_env_path(self) -> Optional[str]:
        """Get the path to the current virtual environment, if any."""
        env_info = self.detect_current_environment()
        return env_info.virtual_env_path
    
    def has_virtual_env(self) -> bool:
        """Check if currently running in a virtual environment."""
        env_info = self.detect_current_environment()
        return env_info.is_virtual_env
    
    def get_current_env_recommendations(self) -> List[str]:
        """Get recommendations for the current environment setup."""
        recommendations = []
        env_info = self.detect_current_environment()
        
        if env_info.manager == EnvironmentManager.SYSTEM:
            recommendations.append("Consider using a virtual environment for better dependency isolation")
            recommendations.append("Try: python -m venv venv && source venv/bin/activate")
        
        # More recommendations will be added as detection methods are implemented
        return recommendations
    
    def _get_poetry_python_executable(self) -> Optional[str]:
        """Get the Python executable used by Poetry."""
        import subprocess
        import shutil
        
        # Check if poetry is available
        if not shutil.which("poetry"):
            return None
        
        try:
            result = subprocess.run(
                ["poetry", "env", "info", "--executable"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.project_root
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        return None
    
    def _get_poetry_venv_path(self) -> Optional[str]:
        """Get the Poetry virtual environment path."""
        import subprocess
        import shutil
        
        # Check if poetry is available
        if not shutil.which("poetry"):
            return None
        
        try:
            result = subprocess.run(
                ["poetry", "env", "info", "--path"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.project_root
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        return None
    
    def _get_python_version(self, python_executable: str) -> str:
        """Get Python version for a given executable."""
        import subprocess
        
        try:
            result = subprocess.run(
                [python_executable, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Output is like "Python 3.9.18"
                version_str = result.stdout.strip()
                if version_str.startswith("Python "):
                    return version_str[7:]  # Remove "Python " prefix
                return version_str
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        # Fallback to current Python version
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    def _get_pipenv_python_executable(self) -> Optional[str]:
        """Get the Python executable used by Pipenv."""
        import subprocess
        import shutil
        
        # Check if pipenv is available
        if not shutil.which("pipenv"):
            return None
        
        try:
            result = subprocess.run(
                ["pipenv", "--py"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.project_root
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        return None
    
    def _get_pipenv_venv_path(self) -> Optional[str]:
        """Get the Pipenv virtual environment path."""
        import subprocess
        import shutil
        
        # Check if pipenv is available
        if not shutil.which("pipenv"):
            return None
        
        try:
            result = subprocess.run(
                ["pipenv", "--venv"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.project_root
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        return None
    
    def _get_conda_python_executable(self) -> Optional[str]:
        """Get the Python executable used by Conda."""
        import subprocess
        import shutil
        import json
        
        # Check if conda is available
        if not shutil.which("conda"):
            return None
        
        try:
            result = subprocess.run(
                ["conda", "info", "--json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                info = json.loads(result.stdout)
                # Get the active environment's python executable
                active_env = info.get("active_prefix")
                if active_env:
                    # Try common paths for python executable
                    possible_paths = [
                        Path(active_env) / "bin" / "python",
                        Path(active_env) / "python.exe",
                        Path(active_env) / "Scripts" / "python.exe"
                    ]
                    for path in possible_paths:
                        if path.exists():
                            return str(path)
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError):
            pass
        
        return None
    
    def _get_uv_python_executable(self) -> Optional[str]:
        """Get the Python executable used by uv."""
        import subprocess
        import shutil
        
        # Check if uv is available
        if not shutil.which("uv"):
            return None
        
        try:
            result = subprocess.run(
                ["uv", "python", "find"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.project_root
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        return None
    
    def _get_uv_venv_path(self) -> Optional[str]:
        """Get the uv virtual environment path."""
        import subprocess
        import shutil
        
        # Check if uv is available
        if not shutil.which("uv"):
            return None
        
        try:
            # Try to get environment info from uv
            result = subprocess.run(
                ["uv", "venv", "--help"],  # Check if venv command exists
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Check if there's a .venv directory (common uv pattern)
                venv_path = self.project_root / ".venv"
                if venv_path.exists():
                    return str(venv_path)
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        return None
    
    def _get_venv_python_executable(self, venv_path: Path) -> Optional[str]:
        """Get the Python executable from a venv/virtualenv directory."""
        # Try common paths for Python executable in virtual environment
        possible_paths = [
            venv_path / "bin" / "python",           # Unix/Linux/macOS
            venv_path / "bin" / "python3",          # Unix/Linux/macOS
            venv_path / "Scripts" / "python.exe",   # Windows
            venv_path / "Scripts" / "python3.exe",  # Windows
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def _get_pyenv_python_executable(self, python_version: str) -> Optional[str]:
        """Get the Python executable for a specific pyenv version."""
        import subprocess
        import shutil
        
        # Check if pyenv is available
        if not shutil.which("pyenv"):
            return None
        
        try:
            result = subprocess.run(
                ["pyenv", "which", "python"],
                capture_output=True,
                text=True,
                timeout=10,
                env={**os.environ, "PYENV_VERSION": python_version}
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        # Alternative: try to construct path directly
        try:
            result = subprocess.run(
                ["pyenv", "root"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                pyenv_root = result.stdout.strip()
                # Try common paths
                possible_paths = [
                    Path(pyenv_root) / "versions" / python_version / "bin" / "python",
                    Path(pyenv_root) / "versions" / python_version / "bin" / "python3",
                ]
                for path in possible_paths:
                    if path.exists():
                        return str(path)
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        
        return None

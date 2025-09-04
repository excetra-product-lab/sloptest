"""Dependency validation and management service."""

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import subprocess

from smart_test_generator.services.base_service import BaseService


@dataclass
class DependencyInfo:
    """Information about a Python package dependency."""
    name: str
    version: Optional[str] = None
    is_installed: bool = False
    install_location: Optional[str] = None
    conflicts: List[str] = None
    
    def __post_init__(self):
        if self.conflicts is None:
            self.conflicts = []


class DependencyService(BaseService):
    """Service for validating and managing Python dependencies."""
    
    def __init__(self, project_root: Path, config, feedback=None):
        super().__init__(project_root, config, feedback)
        self._cache: Dict[str, DependencyInfo] = {}
    
    def is_installed(self, package_name: str) -> bool:
        """
        Check if a package is installed in the current environment.
        
        Args:
            package_name: Name of the package to check
            
        Returns:
            True if package is installed, False otherwise
        """
        if package_name in self._cache:
            return self._cache[package_name].is_installed
        
        dependency_info = self._get_dependency_info(package_name)
        return dependency_info.is_installed
    
    def get_dependency_info(self, package_name: str) -> DependencyInfo:
        """
        Get detailed information about a dependency.
        
        Args:
            package_name: Name of the package
            
        Returns:
            DependencyInfo with package details
        """
        if package_name in self._cache:
            return self._cache[package_name]
        
        return self._get_dependency_info(package_name)
    
    def check_missing_deps(self, required_packages: List[str] = None) -> List[str]:
        """
        Check for missing dependencies.
        
        Args:
            required_packages: List of required package names. 
                             If None, uses default test dependencies.
        
        Returns:
            List of missing package names
        """
        if required_packages is None:
            required_packages = ["pytest", "pytest-cov"]
        
        missing = []
        for package in required_packages:
            if not self.is_installed(package):
                missing.append(package)
        
        return missing
    
    def check_conflicts(self) -> List[str]:
        """
        Check for version conflicts in installed packages.
        
        Returns:
            List of conflict descriptions
        """
        conflicts = []
        
        # Check common testing package conflicts
        test_packages = ["pytest", "pytest-cov", "coverage"]
        
        for package in test_packages:
            if self.is_installed(package):
                dep_info = self.get_dependency_info(package)
                conflicts.extend(dep_info.conflicts)
        
        return conflicts
    
    def get_installation_command(self, package_name: str, 
                                env_manager: str = None) -> List[str]:
        """
        Get the installation command for a package.
        
        Args:
            package_name: Name of the package to install
            env_manager: Environment manager (poetry, pipenv, conda, uv, pip)
            
        Returns:
            Command as list of strings
        """
        if env_manager == "poetry":
            return ["poetry", "add", "--group", "dev", package_name]
        elif env_manager == "pipenv":
            return ["pipenv", "install", "--dev", package_name]
        elif env_manager == "conda":
            return ["conda", "install", package_name]
        elif env_manager == "uv":
            return ["uv", "add", "--dev", package_name]
        else:
            # Default to pip
            return [sys.executable, "-m", "pip", "install", package_name]
    
    def _get_dependency_info(self, package_name: str) -> DependencyInfo:
        """Get detailed dependency information and cache it."""
        try:
            # Try to import the package
            module = importlib.import_module(package_name)
            
            # Get version if available
            version = None
            if hasattr(module, "__version__"):
                version = module.__version__
            elif hasattr(module, "version"):
                version = module.version
            else:
                # Try to get version from package metadata
                version = self._get_version_from_metadata(package_name)
            
            # Get install location
            install_location = None
            if hasattr(module, "__file__") and module.__file__:
                install_location = str(Path(module.__file__).parent)
            
            dep_info = DependencyInfo(
                name=package_name,
                version=version,
                is_installed=True,
                install_location=install_location,
                conflicts=[]
            )
            
        except ImportError:
            dep_info = DependencyInfo(
                name=package_name,
                is_installed=False
            )
        except Exception as e:
            self._log_debug(f"Error getting dependency info for {package_name}: {e}")
            dep_info = DependencyInfo(
                name=package_name,
                is_installed=False
            )
        
        # Cache the result
        self._cache[package_name] = dep_info
        return dep_info
    
    def _get_version_from_metadata(self, package_name: str) -> Optional[str]:
        """Try to get package version from metadata."""
        try:
            # Python 3.8+
            from importlib.metadata import version
            return version(package_name)
        except ImportError:
            try:
                # Fallback for older Python versions
                import pkg_resources
                return pkg_resources.get_distribution(package_name).version
            except Exception:
                pass
        except Exception:
            pass
        
        return None

"""Tests for DependencyService validation methods."""

import sys
import tempfile
import unittest.mock as mock
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from smart_test_generator.services.dependency_service import (
    DependencyService,
    DependencyInfo
)


class TestDependencyService:
    """Test suite for DependencyService validation methods."""
    
    def setup_method(self, method):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Mock()
        self.feedback = Mock()
        self.service = DependencyService(
            project_root=self.temp_dir,
            config=self.config,
            feedback=self.feedback
        )
    
    def teardown_method(self, method):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_is_installed_package_exists(self):
        """Test checking if an installed package exists."""
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.__version__ = "1.0.0"
            mock_import.return_value = mock_module
            
            is_installed = self.service.is_installed("pytest")
            assert is_installed is True
            mock_import.assert_called_with("pytest")
    
    def test_is_installed_package_missing(self):
        """Test checking if a missing package exists."""
        with patch('importlib.import_module', side_effect=ImportError("No module named 'nonexistent'")):
            is_installed = self.service.is_installed("nonexistent-package")
            assert is_installed is False
    
    def test_is_installed_uses_cache(self):
        """Test that dependency checking uses cache."""
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.__version__ = "1.0.0"
            mock_import.return_value = mock_module
            
            # First call should import
            is_installed1 = self.service.is_installed("pytest")
            assert mock_import.call_count == 1
            
            # Second call should use cache
            is_installed2 = self.service.is_installed("pytest")
            assert mock_import.call_count == 1  # Should not import again
            assert is_installed1 is True
            assert is_installed2 is True
    
    def test_get_dependency_info_with_version_attribute(self):
        """Test getting dependency info for package with __version__ attribute."""
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.__version__ = "7.1.2"
            mock_module.__file__ = "/path/to/pytest/__init__.py"
            mock_import.return_value = mock_module
            
            dep_info = self.service.get_dependency_info("pytest")
            
            assert dep_info.name == "pytest"
            assert dep_info.version == "7.1.2"
            assert dep_info.is_installed is True
            assert dep_info.install_location == "/path/to/pytest"
            assert dep_info.conflicts == []
    
    def test_get_dependency_info_with_version_attribute_fallback(self):
        """Test getting dependency info for package with version attribute."""
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.version = "1.0.0"  # version instead of __version__
            mock_module.__file__ = "/path/to/package/__init__.py"
            del mock_module.__version__  # Ensure __version__ doesn't exist
            mock_import.return_value = mock_module
            
            dep_info = self.service.get_dependency_info("package")
            
            assert dep_info.version == "1.0.0"
    
    def test_get_dependency_info_with_metadata_version(self):
        """Test getting dependency info using importlib.metadata."""
        with patch('importlib.import_module') as mock_import, \
             patch.object(self.service, '_get_version_from_metadata') as mock_metadata:
            
            mock_module = Mock()
            del mock_module.__version__  # No __version__ attribute
            del mock_module.version      # No version attribute
            mock_module.__file__ = "/path/to/package/__init__.py"
            mock_import.return_value = mock_module
            mock_metadata.return_value = "2.3.4"
            
            dep_info = self.service.get_dependency_info("package")
            
            assert dep_info.version == "2.3.4"
            mock_metadata.assert_called_with("package")
    
    def test_get_dependency_info_no_file_attribute(self):
        """Test getting dependency info for package without __file__ attribute."""
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.__version__ = "1.0.0"
            del mock_module.__file__  # No __file__ attribute (e.g., built-in module)
            mock_import.return_value = mock_module
            
            dep_info = self.service.get_dependency_info("builtin")
            
            assert dep_info.install_location is None
    
    def test_get_dependency_info_import_error(self):
        """Test getting dependency info for non-existent package."""
        with patch('importlib.import_module', side_effect=ImportError("No module named 'missing'")):
            dep_info = self.service.get_dependency_info("missing")
            
            assert dep_info.name == "missing"
            assert dep_info.version is None
            assert dep_info.is_installed is False
            assert dep_info.install_location is None
            assert dep_info.conflicts == []
    
    def test_get_dependency_info_generic_exception(self):
        """Test getting dependency info handles generic exceptions."""
        with patch('importlib.import_module', side_effect=AttributeError("Some attribute error")):
            dep_info = self.service.get_dependency_info("broken")
            
            assert dep_info.name == "broken"
            assert dep_info.is_installed is False
    
    def test_check_missing_deps_default_packages(self):
        """Test checking missing dependencies with default packages."""
        with patch.object(self.service, 'is_installed') as mock_installed:
            mock_installed.side_effect = lambda pkg: pkg == "pytest"  # Only pytest is installed
            
            missing = self.service.check_missing_deps()
            
            assert "pytest-cov" in missing
            assert "pytest" not in missing
    
    def test_check_missing_deps_custom_packages(self):
        """Test checking missing dependencies with custom packages."""
        custom_packages = ["requests", "numpy", "pandas"]
        
        with patch.object(self.service, 'is_installed') as mock_installed:
            mock_installed.side_effect = lambda pkg: pkg in ["requests", "numpy"]
            
            missing = self.service.check_missing_deps(custom_packages)
            
            assert missing == ["pandas"]
            assert "requests" not in missing
            assert "numpy" not in missing
    
    def test_check_missing_deps_all_installed(self):
        """Test checking missing dependencies when all are installed."""
        with patch.object(self.service, 'is_installed', return_value=True):
            missing = self.service.check_missing_deps()
            assert missing == []
    
    def test_check_conflicts_finds_conflicts(self):
        """Test checking conflicts finds dependency conflicts."""
        with patch.object(self.service, 'is_installed', return_value=True), \
             patch.object(self.service, 'get_dependency_info') as mock_get_info:
            
            # Mock conflicting dependency info
            mock_dep_info = DependencyInfo(
                name="pytest",
                version="6.0.0",
                is_installed=True,
                conflicts=["Incompatible with coverage < 5.0", "Version conflict with pytest-cov"]
            )
            mock_get_info.return_value = mock_dep_info
            
            conflicts = self.service.check_conflicts()
            
            assert len(conflicts) > 0
            assert "Incompatible with coverage < 5.0" in conflicts
    
    def test_check_conflicts_no_conflicts(self):
        """Test checking conflicts when no conflicts exist."""
        with patch.object(self.service, 'is_installed', return_value=True), \
             patch.object(self.service, 'get_dependency_info') as mock_get_info:
            
            mock_dep_info = DependencyInfo(
                name="pytest",
                version="7.0.0",
                is_installed=True,
                conflicts=[]
            )
            mock_get_info.return_value = mock_dep_info
            
            conflicts = self.service.check_conflicts()
            assert conflicts == []
    
    def test_check_conflicts_package_not_installed(self):
        """Test checking conflicts skips uninstalled packages."""
        with patch.object(self.service, 'is_installed', return_value=False):
            conflicts = self.service.check_conflicts()
            assert conflicts == []
    
    def test_get_installation_command_poetry(self):
        """Test getting installation command for Poetry."""
        cmd = self.service.get_installation_command("pytest", "poetry")
        expected = ["poetry", "add", "--group", "dev", "pytest"]
        assert cmd == expected
    
    def test_get_installation_command_pipenv(self):
        """Test getting installation command for Pipenv."""
        cmd = self.service.get_installation_command("pytest", "pipenv")
        expected = ["pipenv", "install", "--dev", "pytest"]
        assert cmd == expected
    
    def test_get_installation_command_conda(self):
        """Test getting installation command for Conda."""
        cmd = self.service.get_installation_command("pytest", "conda")
        expected = ["conda", "install", "pytest"]
        assert cmd == expected
    
    def test_get_installation_command_uv(self):
        """Test getting installation command for uv."""
        cmd = self.service.get_installation_command("pytest", "uv")
        expected = ["uv", "add", "--dev", "pytest"]
        assert cmd == expected
    
    def test_get_installation_command_pip_default(self):
        """Test getting installation command defaults to pip."""
        cmd = self.service.get_installation_command("pytest")
        expected = [sys.executable, "-m", "pip", "install", "pytest"]
        assert cmd == expected
    
    def test_get_installation_command_pip_explicit(self):
        """Test getting installation command for pip explicitly."""
        cmd = self.service.get_installation_command("pytest", "pip")
        expected = [sys.executable, "-m", "pip", "install", "pytest"]
        assert cmd == expected
    
    def test_get_installation_command_unknown_manager(self):
        """Test getting installation command for unknown environment manager."""
        cmd = self.service.get_installation_command("pytest", "unknown")
        expected = [sys.executable, "-m", "pip", "install", "pytest"]
        assert cmd == expected


class TestDependencyServiceVersionDetection:
    """Test version detection methods."""
    
    def setup_method(self, method):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Mock()
        self.feedback = Mock()
        self.service = DependencyService(
            project_root=self.temp_dir,
            config=self.config,
            feedback=self.feedback
        )
    
    def teardown_method(self, method):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_version_from_metadata_importlib_metadata(self):
        """Test getting version from importlib.metadata (Python 3.8+)."""
        with patch('importlib.metadata.version') as mock_version:
            mock_version.return_value = "1.2.3"
            
            version = self.service._get_version_from_metadata("test-package")
            assert version == "1.2.3"
            mock_version.assert_called_with("test-package")
    
    def test_get_version_from_metadata_pkg_resources_fallback(self):
        """Test fallback to pkg_resources when importlib.metadata not available."""
        with patch('importlib.metadata.version', side_effect=ImportError()):
            # Mock the entire pkg_resources module since it may not be available
            with patch.dict('sys.modules', {'pkg_resources': Mock()}):
                import sys
                mock_dist = Mock()
                mock_distribution = Mock()
                mock_distribution.version = "2.3.4"
                mock_dist.return_value = mock_distribution
                sys.modules['pkg_resources'].get_distribution = mock_dist
                
                version = self.service._get_version_from_metadata("test-package")
                assert version == "2.3.4"
    
    def test_get_version_from_metadata_all_fail(self):
        """Test version detection when all methods fail."""
        with patch('importlib.metadata.version', side_effect=ImportError()):
            # Mock the entire pkg_resources module since it may not be available
            with patch.dict('sys.modules', {'pkg_resources': Mock()}):
                import sys
                sys.modules['pkg_resources'].get_distribution = Mock(side_effect=ImportError())
                
                version = self.service._get_version_from_metadata("test-package")
                assert version is None
    
    def test_get_version_from_metadata_package_not_found(self):
        """Test version detection for non-existent package."""
        with patch('importlib.metadata.version', side_effect=Exception("Package not found")):
            version = self.service._get_version_from_metadata("nonexistent")
            assert version is None


class TestDependencyServiceCaching:
    """Test caching behavior of DependencyService."""
    
    def setup_method(self, method):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Mock()
        self.feedback = Mock()
        self.service = DependencyService(
            project_root=self.temp_dir,
            config=self.config,
            feedback=self.feedback
        )
    
    def teardown_method(self, method):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dependency_info_cached_after_first_call(self):
        """Test that dependency info is cached after first call."""
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.__version__ = "1.0.0"
            mock_module.__file__ = "/path/to/package/__init__.py"
            mock_import.return_value = mock_module
            
            # First call should import and cache
            dep_info1 = self.service.get_dependency_info("test-package")
            assert mock_import.call_count == 1
            
            # Second call should use cache
            dep_info2 = self.service.get_dependency_info("test-package")
            assert mock_import.call_count == 1  # Should not import again
            assert dep_info1 is dep_info2  # Same object from cache
    
    def test_is_installed_uses_cached_dependency_info(self):
        """Test that is_installed uses cached dependency info."""
        # Pre-populate cache
        cached_info = DependencyInfo(name="cached-package", is_installed=True)
        self.service._cache["cached-package"] = cached_info
        
        with patch('importlib.import_module') as mock_import:
            # Should not call import_module since info is cached
            is_installed = self.service.is_installed("cached-package")
            assert is_installed is True
            mock_import.assert_not_called()
    
    def test_cache_separate_for_different_packages(self):
        """Test that cache maintains separate entries for different packages."""
        with patch('importlib.import_module') as mock_import:
            mock_module1 = Mock()
            mock_module1.__version__ = "1.0.0"
            mock_module2 = Mock()
            mock_module2.__version__ = "2.0.0"
            
            mock_import.side_effect = [mock_module1, mock_module2]
            
            dep_info1 = self.service.get_dependency_info("package1")
            dep_info2 = self.service.get_dependency_info("package2")
            
            assert dep_info1.version == "1.0.0"
            assert dep_info2.version == "2.0.0"
            assert mock_import.call_count == 2


class TestDependencyServiceEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self, method):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Mock()
        self.feedback = Mock()
        self.service = DependencyService(
            project_root=self.temp_dir,
            config=self.config,
            feedback=self.feedback
        )
    
    def teardown_method(self, method):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_check_missing_deps_empty_list(self):
        """Test checking missing dependencies with empty list."""
        missing = self.service.check_missing_deps([])
        assert missing == []
    
    def test_check_missing_deps_none_input(self):
        """Test checking missing dependencies with None input uses defaults."""
        with patch.object(self.service, 'is_installed', return_value=False):
            missing = self.service.check_missing_deps(None)
            assert "pytest" in missing
            assert "pytest-cov" in missing
    
    def test_get_dependency_info_handles_mock_objects(self):
        """Test that get_dependency_info handles mock objects in tests."""
        with patch('importlib.import_module') as mock_import:
            # Create a mock that behaves unusually
            mock_module = Mock()
            mock_module.__version__ = Mock()  # __version__ is a Mock, not a string
            mock_module.__version__.__str__ = Mock(return_value="1.0.0")
            mock_import.return_value = mock_module
            
            # Should handle this gracefully
            dep_info = self.service.get_dependency_info("unusual-package")
            assert dep_info.name == "unusual-package"
            assert dep_info.is_installed is True
    
    def test_conflict_checking_with_mixed_installation_states(self):
        """Test conflict checking with mixed package installation states."""
        # Mock some packages as installed, others as not
        install_states = {"pytest": True, "pytest-cov": False, "coverage": True}
        
        with patch.object(self.service, 'is_installed') as mock_installed, \
             patch.object(self.service, 'get_dependency_info') as mock_get_info:
            
            mock_installed.side_effect = lambda pkg: install_states.get(pkg, False)
            
            # Only installed packages should be checked
            mock_get_info.return_value = DependencyInfo("test", conflicts=[])
            
            conflicts = self.service.check_conflicts()
            
            # Should only check installed packages
            expected_calls = ["pytest", "coverage"]  # pytest-cov not installed, so skipped
            actual_calls = [call[0][0] for call in mock_get_info.call_args_list]
            assert set(actual_calls) == set(expected_calls)
    
    def test_installation_command_with_special_characters(self):
        """Test installation command generation with special package names."""
        special_packages = [
            "package-with-dashes",
            "package_with_underscores",
            "Package.With.Dots",
            "package123"
        ]
        
        for package in special_packages:
            cmd = self.service.get_installation_command(package, "pip")
            assert package in cmd
            assert cmd[-1] == package
    
    def test_version_detection_with_namespace_packages(self):
        """Test version detection for namespace packages."""
        with patch('importlib.import_module') as mock_import:
            # Simulate namespace package (no __file__ attribute)
            mock_module = Mock()
            del mock_module.__file__
            mock_module.__version__ = "1.0.0"
            mock_import.return_value = mock_module
            
            dep_info = self.service.get_dependency_info("namespace-package")
            
            assert dep_info.is_installed is True
            assert dep_info.version == "1.0.0"
            assert dep_info.install_location is None

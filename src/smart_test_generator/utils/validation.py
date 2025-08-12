"""Input validation utilities for test generation."""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path
from typing import Union, List

from smart_test_generator.exceptions import ValidationError, ProjectStructureError, DependencyError

logger = __import__('logging').getLogger(__name__)


class SystemValidator:
    """System-level validation utilities."""
    
    @staticmethod
    def check_python_version(min_version: tuple = (3, 8)) -> None:
        """Check if Python version meets requirements."""
        current_version = sys.version_info[:2]
        if current_version < min_version:
            raise ValidationError(
                f"Python {min_version[0]}.{min_version[1]}+ required, found {current_version[0]}.{current_version[1]}",
                suggestion=f"Please upgrade to Python {min_version[0]}.{min_version[1]} or higher."
            )


class Validator:
    """Comprehensive input validation with helpful error messages."""
    
    @staticmethod
    def find_project_root(start_dir: Union[str, Path]) -> Path:
        """
        Find the actual project root by looking for project indicators.
        
        Args:
            start_dir: Directory to start searching from
            
        Returns:
            Path to the project root
        """
        # Project indicators in order of preference
        project_indicators = [
            'pyproject.toml',
            'setup.py', 
            'setup.cfg',
            'requirements.txt',
            'Pipfile',
            'poetry.lock',
            '.git',
            '.testgen.yml',
            '.testgen_state.json'
        ]
        
        current_dir = Path(start_dir).resolve()
        
        # Traverse up the directory tree
        for parent in [current_dir] + list(current_dir.parents):
            # Check if this directory has any project indicators
            for indicator in project_indicators:
                if (parent / indicator).exists():
                    logger.debug(f"Found project root at {parent} (indicator: {indicator})")
                    return parent
        
        # If no indicators found, use the start directory
        logger.warning(f"No project indicators found, using start directory: {current_dir}")
        return current_dir
    
    @staticmethod
    def validate_directory_only(directory: Union[str, Path], must_exist: bool = True, must_be_writable: bool = False) -> Path:
        """
        Validate directory path without finding project root.
        
        Args:
            directory: Directory path to validate
            must_exist: Whether directory must already exist
            must_be_writable: Whether directory must be writable
            
        Returns:
            Validated Path object (as-is, no project root finding)
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            dir_path = Path(directory).resolve()
        except Exception as e:
            raise ValidationError(
                f"Invalid directory path: {directory}",
                suggestion="Please provide a valid directory path."
            )
        
        if must_exist and not dir_path.exists():
            raise ValidationError(
                f"Directory does not exist: {dir_path}",
                suggestion=f"Create the directory with: mkdir -p {dir_path}"
            )
        
        if dir_path.exists() and not dir_path.is_dir():
            raise ValidationError(
                f"Path exists but is not a directory: {dir_path}",
                suggestion="Please provide a path to a directory, not a file."
            )
        
        if must_be_writable:
            if dir_path.exists() and not os.access(dir_path, os.W_OK):
                raise ValidationError(
                    f"Directory is not writable: {dir_path}",
                    suggestion="Please check permissions and ensure you have write access."
                )
        
        return dir_path

    @staticmethod
    def validate_directory(directory: Union[str, Path], must_exist: bool = True, must_be_writable: bool = False) -> Path:
        """
        Validate directory path and find project root.
        
        Args:
            directory: Directory path to validate
            must_exist: Whether directory must already exist
            must_be_writable: Whether directory must be writable
            
        Returns:
            Validated Path object (project root)
            
        Raises:
            ValidationError: If validation fails
        """
        # First validate the directory itself
        dir_path = Validator.validate_directory_only(directory, must_exist, must_be_writable)
        
        # Find the actual project root
        project_root = Validator.find_project_root(dir_path)
        return project_root

    @staticmethod
    def validate_python_project(directory: Path) -> None:
        """
        Validate that directory contains a Python project.
        
        Args:
            directory: Project directory to validate
            
        Raises:
            ProjectStructureError: If not a valid Python project
        """
        # Check for common Python project indicators
        python_indicators = [
            'setup.py',
            'pyproject.toml',
            'requirements.txt',
            'Pipfile',
            'poetry.lock',
            'setup.cfg'
        ]
        
        # Look for Python files
        python_files = list(directory.rglob('*.py'))
        
        # Look for project indicators
        has_indicators = any((directory / indicator).exists() for indicator in python_indicators)
        
        if not python_files and not has_indicators:
            raise ProjectStructureError(
                f"No Python project found in: {directory}",
                suggestion="Ensure you're in the correct directory and it contains Python files or project configuration."
            )
        
        if not python_files:
            raise ProjectStructureError(
                f"No Python files found in: {directory}",
                suggestion="Add some Python files to your project directory."
            )

    @staticmethod
    def validate_file_path(filepath: Union[str, Path], must_exist: bool = True, 
                          must_be_readable: bool = True) -> Path:
        """
        Validate file path.
        
        Args:
            filepath: File path to validate
            must_exist: Whether file must already exist
            must_be_readable: Whether file must be readable
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            file_path = Path(filepath).resolve()
        except Exception as e:
            raise ValidationError(
                f"Invalid file path: {filepath}",
                suggestion="Please provide a valid file path."
            )
        
        if must_exist and not file_path.exists():
            raise ValidationError(
                f"File does not exist: {file_path}",
                suggestion="Ensure the file exists and the path is correct."
            )
        
        if file_path.exists() and not file_path.is_file():
            raise ValidationError(
                f"Path exists but is not a file: {file_path}",
                suggestion="Please provide a path to a file, not a directory."
            )
        
        if must_be_readable and file_path.exists() and not os.access(file_path, os.R_OK):
            raise ValidationError(
                f"File is not readable: {file_path}",
                suggestion="Please check permissions and ensure you have read access."
            )
        
        return file_path

    @staticmethod  
    def validate_files_exist(filepaths: List[Union[str, Path]]) -> List[Path]:
        """
        Validate that multiple files exist.
        
        Args:
            filepaths: List of file paths to check
            
        Returns:
            List of validated Path objects
            
        Raises:
            ValidationError: If any file doesn't exist
        """
        validated_paths = []
        missing_files = []
        
        for filepath in filepaths:
            file_path = Path(filepath)
            if file_path.exists():
                validated_paths.append(file_path.resolve())
            else:
                missing_files.append(str(filepath))
        
        if missing_files:
            raise ValidationError(
                f"Missing files: {', '.join(missing_files)}",
                suggestion="Ensure all required files exist."
            )
        
        return validated_paths

    @staticmethod
    def check_dependencies(packages: List[str]) -> None:
        """
        Check if required Python packages are available.
        
        Args:
            packages: List of package names to check
            
        Raises:
            DependencyError: If any package is missing
        """
        missing_packages = []
        
        for package in packages:
            if importlib.util.find_spec(package) is None:
                missing_packages.append(package)
        
        if missing_packages:
            suggestion = f"Install missing packages with: pip install {' '.join(missing_packages)}"
            raise DependencyError(
                f"Missing required packages: {', '.join(missing_packages)}",
                suggestion=suggestion
            )

    @staticmethod
    def validate_api_credentials(claude_api_key: str = None, azure_endpoint: str = None, 
                                azure_api_key: str = None, azure_deployment: str = None,
                                bedrock_model_id: str = None) -> None:
        """
        Validate API credentials.
        
        Args:
            claude_api_key: Claude API key
            azure_endpoint: Azure OpenAI endpoint
            azure_api_key: Azure OpenAI API key  
            azure_deployment: Azure OpenAI deployment name
            
        Raises:
            ValidationError: If credentials are invalid
        """
        has_claude = claude_api_key is not None
        has_azure = all([azure_endpoint, azure_api_key, azure_deployment])
        has_bedrock = bedrock_model_id is not None
        
        if not has_claude and not has_azure and not has_bedrock:
            raise ValidationError(
                "No valid API credentials provided",
                suggestion="Provide either Claude API key (--claude-api-key), Azure OpenAI credentials (--endpoint, --api-key, --deployment), or AWS Bedrock model ID (--bedrock-model-id)."
            )
        
        # Validate Claude API key format
        if has_claude and not claude_api_key.startswith('sk-ant-'):
            raise ValidationError(
                "Invalid Claude API key format",
                suggestion="Claude API keys should start with 'sk-ant-'. Check your API key."
            )
        
        # Validate Azure endpoint format
        if has_azure and not azure_endpoint.startswith('https://'):
            raise ValidationError(
                "Invalid Azure endpoint format",
                suggestion="Azure endpoints should start with 'https://'. Example: https://your-resource.openai.azure.com/"
            )

    @staticmethod
    def validate_api_key(api_key: str, provider: str) -> str:
        """
        Validate API key format for specific provider.
        
        Args:
            api_key: API key to validate
            provider: Provider name (e.g., "Claude", "Azure OpenAI")
            
        Returns:
            Validated API key
            
        Raises:
            ValidationError: If API key format is invalid
        """
        if not api_key or not api_key.strip():
            raise ValidationError(
                f"Empty {provider} API key provided",
                suggestion=f"Provide a valid {provider} API key."
            )
        
        api_key = api_key.strip()
        
        # Validate Claude API key format
        if provider.lower() == "claude" and not api_key.startswith('sk-ant-'):
            raise ValidationError(
                f"Invalid {provider} API key format",
                suggestion="Claude API keys should start with 'sk-ant-'. Check your API key."
            )
        
        # Basic validation for Azure OpenAI (they don't have a specific format requirement)
        if provider.lower() in ["azure openai", "azure"] and len(api_key) < 10:
            raise ValidationError(
                f"Invalid {provider} API key format",
                suggestion="Azure OpenAI API keys should be longer. Check your API key."
            )
        
        return api_key

    @staticmethod
    def validate_model_name(model: str, available_models: List[str]) -> str:
        """
        Validate model name against available models.
        
        Args:
            model: Model name to validate
            available_models: List of available model names
            
        Returns:
            Validated model name
            
        Raises:
            ValidationError: If model is not available
        """
        if not model or not model.strip():
            raise ValidationError(
                "Empty model name provided",
                suggestion=f"Provide one of the available models: {', '.join(available_models)}"
            )
        
        model = model.strip()
        
        if model not in available_models:
            raise ValidationError(
                f"Invalid model: {model}",
                suggestion=f"Available models: {', '.join(available_models)}"
            )
        
        return model

    @staticmethod
    def validate_batch_size(batch_size: int) -> int:
        """
        Validate batch size parameter.
        
        Args:
            batch_size: Batch size to validate
            
        Returns:
            Validated batch size
            
        Raises:
            ValidationError: If batch size is invalid
        """
        if batch_size < 1:
            raise ValidationError(
                f"Batch size must be at least 1, got: {batch_size}",
                suggestion="Use a positive integer for batch size."
            )
        
        if batch_size > 50:
            raise ValidationError(
                f"Batch size too large: {batch_size}",
                suggestion="Use a smaller batch size (1-50) to avoid API rate limits and timeout issues."
            )
        
        return batch_size

    @staticmethod  
    def validate_config_file(config_file: Union[str, Path]) -> Path:
        """
        Validate configuration file.
        
        Args:
            config_file: Path to config file
            
        Returns:
            Validated config file path
            
        Raises:
            ValidationError: If config file is invalid
        """
        config_path = Path(config_file)
        
        if config_path.exists():
            if not config_path.is_file():
                raise ValidationError(
                    f"Config path is not a file: {config_path}",
                    suggestion="Provide a path to a configuration file."
                )
            
            if not config_path.suffix in ['.yml', '.yaml', '.json']:
                raise ValidationError(
                    f"Unsupported config file format: {config_path.suffix}",
                    suggestion="Use a .yml, .yaml, or .json configuration file."
                )
        
        return config_path.resolve()


class GitValidator:
    """Git-related validation utilities."""
    
    @staticmethod
    def is_git_repository(directory: Path) -> bool:
        """Check if directory is a git repository."""
        return (directory / '.git').exists()
    
    @staticmethod
    def check_git_available() -> bool:
        """Check if git command is available."""
        try:
            subprocess.run(['git', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    @staticmethod  
    def validate_git_repository(directory: Path) -> None:
        """
        Validate git repository setup.
        
        Args:
            directory: Directory to check
            
        Raises:
            ValidationError: If git setup is invalid
        """
        if not GitValidator.check_git_available():
            raise ValidationError(
                "Git is not available",
                suggestion="Install git to enable version control features."
            )
        
        if not GitValidator.is_git_repository(directory):
            raise ValidationError(
                f"Not a git repository: {directory}",
                suggestion="Initialize git repository with: git init"
            )


class EnvironmentValidator:
    """Environment and system validation utilities."""
    
    @staticmethod
    def check_disk_space(directory: Path, required_mb: int = 100) -> None:
        """
        Check available disk space.
        
        Args:
            directory: Directory to check
            required_mb: Required space in MB
            
        Raises:
            ValidationError: If insufficient disk space
        """
        try:
            stat = os.statvfs(directory)
            available_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
            
            if available_mb < required_mb:
                raise ValidationError(
                    f"Insufficient disk space: {available_mb:.1f}MB available, {required_mb}MB required",
                    suggestion="Free up disk space or use a different directory."
                )
        except (OSError, AttributeError):
            # Skip disk space check on unsupported systems
            pass
    
    @staticmethod
    def check_memory_available(required_mb: int = 512) -> None:
        """
        Check available memory.
        
        Args:
            required_mb: Required memory in MB
            
        Raises:
            ValidationError: If insufficient memory
        """
        try:
            import psutil
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            
            if available_mb < required_mb:
                raise ValidationError(
                    f"Insufficient memory: {available_mb:.1f}MB available, {required_mb}MB required",
                    suggestion="Close other applications to free up memory."
                )
        except ImportError:
            # Skip memory check if psutil not available
            pass
        except Exception:
            # Skip memory check on error
            pass 
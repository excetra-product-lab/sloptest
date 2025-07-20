"""Custom exception classes for smart test generator."""

from typing import Optional, List


class SmartTestGeneratorError(Exception):
    """Base exception for all smart test generator errors."""
    
    def __init__(self, message: str, suggestion: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion
    
    def __str__(self):
        result = self.message
        if self.suggestion:
            result += f"\n\nSuggestion: {self.suggestion}"
        return result


class ConfigurationError(SmartTestGeneratorError):
    """Raised when there are configuration-related issues."""
    pass


class ValidationError(SmartTestGeneratorError):
    """Raised when input validation fails."""
    pass


class FileOperationError(SmartTestGeneratorError):
    """Raised when file operations fail."""
    
    def __init__(self, message: str, filepath: Optional[str] = None, suggestion: Optional[str] = None):
        super().__init__(message, suggestion)
        self.filepath = filepath


class LLMClientError(SmartTestGeneratorError):
    """Raised when LLM client operations fail."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, suggestion: Optional[str] = None):
        super().__init__(message, suggestion)
        self.status_code = status_code


class CoverageAnalysisError(SmartTestGeneratorError):
    """Raised when coverage analysis fails."""
    pass


class TestGenerationError(SmartTestGeneratorError):
    """Raised when test generation fails."""
    
    def __init__(self, message: str, failed_files: Optional[List[str]] = None, suggestion: Optional[str] = None):
        super().__init__(message, suggestion)
        self.failed_files = failed_files or []


class DependencyError(SmartTestGeneratorError):
    """Raised when required dependencies are missing."""
    
    def __init__(self, message: str, missing_packages: Optional[List[str]] = None, suggestion: Optional[str] = None):
        super().__init__(message, suggestion)
        self.missing_packages = missing_packages or []


class AuthenticationError(SmartTestGeneratorError):
    """Raised when authentication fails with LLM services."""
    pass


class ProjectStructureError(SmartTestGeneratorError):
    """Raised when there are issues with project structure."""
    pass 
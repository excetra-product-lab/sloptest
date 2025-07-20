import pytest
from smart_test_generator.exceptions import (
    SmartTestGeneratorError,
    FileOperationError,
    LLMClientError,
    TestGenerationError,
    DependencyError
)


class TestSmartTestGeneratorError:
    """Test SmartTestGeneratorError class."""
    
    def test_init_with_message_only(self):
        """Test SmartTestGeneratorError initialization with message only."""
        # Arrange
        message = "Test error message"
        
        # Act
        error = SmartTestGeneratorError(message)
        
        # Assert
        assert error.message == message
        assert error.suggestion is None
        assert str(error) == message
    
    def test_init_with_message_and_suggestion(self):
        """Test SmartTestGeneratorError initialization with message and suggestion."""
        # Arrange
        message = "Test error message"
        suggestion = "Try this solution"
        
        # Act
        error = SmartTestGeneratorError(message, suggestion)
        
        # Assert
        assert error.message == message
        assert error.suggestion == suggestion
    
    def test_init_with_none_suggestion(self):
        """Test SmartTestGeneratorError initialization with explicit None suggestion."""
        # Arrange
        message = "Test error message"
        
        # Act
        error = SmartTestGeneratorError(message, None)
        
        # Assert
        assert error.message == message
        assert error.suggestion is None
    
    def test_str_with_message_only(self):
        """Test __str__ method returns message when no suggestion provided."""
        # Arrange
        message = "Test error message"
        error = SmartTestGeneratorError(message)
        
        # Act
        result = str(error)
        
        # Assert
        assert result == message
    
    def test_str_with_message_and_suggestion(self):
        """Test __str__ method returns message with suggestion when provided."""
        # Arrange
        message = "Test error message"
        suggestion = "Try this solution"
        error = SmartTestGeneratorError(message, suggestion)
        expected = f"{message}\n\nSuggestion: {suggestion}"
        
        # Act
        result = str(error)
        
        # Assert
        assert result == expected
    
    def test_str_with_empty_suggestion(self):
        """Test __str__ method with empty string suggestion."""
        # Arrange
        message = "Test error message"
        suggestion = ""
        error = SmartTestGeneratorError(message, suggestion)
        
        # Act
        result = str(error)
        
        # Assert
        assert result == message  # Empty suggestion should not be included
    
    def test_str_with_whitespace_only_suggestion(self):
        """Test __str__ method with whitespace-only suggestion."""
        # Arrange
        message = "Test error message"
        suggestion = "   "
        error = SmartTestGeneratorError(message, suggestion)
        expected = f"{message}\n\nSuggestion: {suggestion}"
        
        # Act
        result = str(error)
        
        # Assert
        assert result == expected


class TestFileOperationError:
    """Test FileOperationError class."""
    
    def test_init_with_message_only(self):
        """Test FileOperationError initialization with message only."""
        # Arrange
        message = "File operation failed"
        
        # Act
        error = FileOperationError(message)
        
        # Assert
        assert error.message == message
        assert error.filepath is None
        assert error.suggestion is None
    
    def test_init_with_all_parameters(self):
        """Test FileOperationError initialization with all parameters."""
        # Arrange
        message = "File operation failed"
        filepath = "/path/to/file.py"
        suggestion = "Check file permissions"
        
        # Act
        error = FileOperationError(message, filepath, suggestion)
        
        # Assert
        assert error.message == message
        assert error.filepath == filepath
        assert error.suggestion == suggestion
    
    def test_init_with_filepath_only(self):
        """Test FileOperationError initialization with filepath but no suggestion."""
        # Arrange
        message = "File operation failed"
        filepath = "/path/to/file.py"
        
        # Act
        error = FileOperationError(message, filepath)
        
        # Assert
        assert error.message == message
        assert error.filepath == filepath
        assert error.suggestion is None
    
    def test_init_with_suggestion_only(self):
        """Test FileOperationError initialization with suggestion but no filepath."""
        # Arrange
        message = "File operation failed"
        suggestion = "Check file permissions"
        
        # Act
        error = FileOperationError(message, suggestion=suggestion)
        
        # Assert
        assert error.message == message
        assert error.filepath is None
        assert error.suggestion == suggestion


class TestLLMClientError:
    """Test LLMClientError class."""
    
    def test_init_with_message_only(self):
        """Test LLMClientError initialization with message only."""
        # Arrange
        message = "LLM client error"
        
        # Act
        error = LLMClientError(message)
        
        # Assert
        assert error.message == message
        assert error.status_code is None
        assert error.suggestion is None
    
    def test_init_with_all_parameters(self):
        """Test LLMClientError initialization with all parameters."""
        # Arrange
        message = "LLM client error"
        status_code = 429
        suggestion = "Wait and retry"
        
        # Act
        error = LLMClientError(message, status_code, suggestion)
        
        # Assert
        assert error.message == message
        assert error.status_code == status_code
        assert error.suggestion == suggestion
    
    def test_init_with_status_code_only(self):
        """Test LLMClientError initialization with status code but no suggestion."""
        # Arrange
        message = "LLM client error"
        status_code = 500
        
        # Act
        error = LLMClientError(message, status_code)
        
        # Assert
        assert error.message == message
        assert error.status_code == status_code
        assert error.suggestion is None
    
    def test_init_with_zero_status_code(self):
        """Test LLMClientError initialization with zero status code."""
        # Arrange
        message = "LLM client error"
        status_code = 0
        
        # Act
        error = LLMClientError(message, status_code)
        
        # Assert
        assert error.message == message
        assert error.status_code == 0


class TestTestGenerationError:
    """Test TestGenerationError class."""
    
    def test_init_with_message_only(self):
        """Test TestGenerationError initialization with message only."""
        # Arrange
        message = "Test generation failed"
        
        # Act
        error = TestGenerationError(message)
        
        # Assert
        assert error.message == message
        assert error.failed_files == []
        assert error.suggestion is None
    
    def test_init_with_all_parameters(self):
        """Test TestGenerationError initialization with all parameters."""
        # Arrange
        message = "Test generation failed"
        failed_files = ["file1.py", "file2.py"]
        suggestion = "Check file syntax"
        
        # Act
        error = TestGenerationError(message, failed_files, suggestion)
        
        # Assert
        assert error.message == message
        assert error.failed_files == failed_files
        assert error.suggestion == suggestion
    
    def test_init_with_empty_failed_files_list(self):
        """Test TestGenerationError initialization with empty failed files list."""
        # Arrange
        message = "Test generation failed"
        failed_files = []
        
        # Act
        error = TestGenerationError(message, failed_files)
        
        # Assert
        assert error.message == message
        assert error.failed_files == []
        assert error.suggestion is None
    
    def test_init_with_none_failed_files(self):
        """Test TestGenerationError initialization with None failed files."""
        # Arrange
        message = "Test generation failed"
        
        # Act
        error = TestGenerationError(message, None)
        
        # Assert
        assert error.message == message
        assert error.failed_files == []
        assert error.suggestion is None
    
    def test_init_with_single_failed_file(self):
        """Test TestGenerationError initialization with single failed file."""
        # Arrange
        message = "Test generation failed"
        failed_files = ["single_file.py"]
        
        # Act
        error = TestGenerationError(message, failed_files)
        
        # Assert
        assert error.message == message
        assert error.failed_files == ["single_file.py"]


class TestDependencyError:
    """Test DependencyError class."""
    
    def test_init_with_message_only(self):
        """Test DependencyError initialization with message only."""
        # Arrange
        message = "Missing dependencies"
        
        # Act
        error = DependencyError(message)
        
        # Assert
        assert error.message == message
        assert error.missing_packages == []
        assert error.suggestion is None
    
    def test_init_with_all_parameters(self):
        """Test DependencyError initialization with all parameters."""
        # Arrange
        message = "Missing dependencies"
        missing_packages = ["pytest", "coverage"]
        suggestion = "Run pip install -r requirements.txt"
        
        # Act
        error = DependencyError(message, missing_packages, suggestion)
        
        # Assert
        assert error.message == message
        assert error.missing_packages == missing_packages
        assert error.suggestion == suggestion
    
    def test_init_with_empty_missing_packages_list(self):
        """Test DependencyError initialization with empty missing packages list."""
        # Arrange
        message = "Missing dependencies"
        missing_packages = []
        
        # Act
        error = DependencyError(message, missing_packages)
        
        # Assert
        assert error.message == message
        assert error.missing_packages == []
        assert error.suggestion is None
    
    def test_init_with_none_missing_packages(self):
        """Test DependencyError initialization with None missing packages."""
        # Arrange
        message = "Missing dependencies"
        
        # Act
        error = DependencyError(message, None)
        
        # Assert
        assert error.message == message
        assert error.missing_packages == []
        assert error.suggestion is None
    
    def test_init_with_single_missing_package(self):
        """Test DependencyError initialization with single missing package."""
        # Arrange
        message = "Missing dependencies"
        missing_packages = ["pytest"]
        
        # Act
        error = DependencyError(message, missing_packages)
        
        # Assert
        assert error.message == message
        assert error.missing_packages == ["pytest"]
    
    def test_init_with_missing_packages_only(self):
        """Test DependencyError initialization with missing packages but no suggestion."""
        # Arrange
        message = "Missing dependencies"
        missing_packages = ["pytest", "coverage"]
        
        # Act
        error = DependencyError(message, missing_packages)
        
        # Assert
        assert error.message == message
        assert error.missing_packages == missing_packages
        assert error.suggestion is None
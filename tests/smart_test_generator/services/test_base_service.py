import pytest
import logging
from pathlib import Path
from unittest.mock import Mock, patch

from smart_test_generator.services.base_service import BaseService
from smart_test_generator.config import Config
from smart_test_generator.utils.user_feedback import UserFeedback


class ConcreteService(BaseService):
    """Concrete implementation of BaseService for testing."""
    pass


class TestBaseServiceInit:
    """Test BaseService.__init__ method."""
    
    def test_init_with_all_parameters(self):
        """Test initialization with all parameters provided."""
        # Arrange
        project_root = Path("/test/project")
        config = Mock(spec=Config)
        feedback = Mock(spec=UserFeedback)
        
        # Act
        service = ConcreteService(project_root, config, feedback)
        
        # Assert
        assert service.project_root == project_root
        assert service.config == config
        assert service.feedback == feedback
        assert isinstance(service.logger, logging.Logger)
        assert service.logger.name == "ConcreteService"
    
    def test_init_without_feedback_creates_default(self):
        """Test initialization without feedback creates default UserFeedback."""
        # Arrange
        project_root = Path("/test/project")
        config = Mock(spec=Config)
        
        # Act
        with patch('smart_test_generator.services.base_service.UserFeedback') as mock_feedback_class:
            mock_feedback_instance = Mock(spec=UserFeedback)
            mock_feedback_class.return_value = mock_feedback_instance
            
            service = ConcreteService(project_root, config)
        
        # Assert
        assert service.project_root == project_root
        assert service.config == config
        assert service.feedback == mock_feedback_instance
        mock_feedback_class.assert_called_once_with()
    
    def test_init_with_none_feedback_creates_default(self):
        """Test initialization with None feedback creates default UserFeedback."""
        # Arrange
        project_root = Path("/test/project")
        config = Mock(spec=Config)
        
        # Act
        with patch('smart_test_generator.services.base_service.UserFeedback') as mock_feedback_class:
            mock_feedback_instance = Mock(spec=UserFeedback)
            mock_feedback_class.return_value = mock_feedback_instance
            
            service = ConcreteService(project_root, config, None)
        
        # Assert
        assert service.project_root == project_root
        assert service.config == config
        assert service.feedback == mock_feedback_instance
        mock_feedback_class.assert_called_once_with()
    
    def test_init_sets_logger_with_class_name(self):
        """Test that logger is set with the correct class name."""
        # Arrange
        project_root = Path("/test/project")
        config = Mock(spec=Config)
        
        # Act
        service = ConcreteService(project_root, config)
        
        # Assert
        assert service.logger.name == "ConcreteService"
        assert isinstance(service.logger, logging.Logger)
    
    def test_init_with_pathlib_path_object(self):
        """Test initialization with Path object for project_root."""
        # Arrange
        project_root = Path("/test/project/subdir")
        config = Mock(spec=Config)
        feedback = Mock(spec=UserFeedback)
        
        # Act
        service = ConcreteService(project_root, config, feedback)
        
        # Assert
        assert service.project_root == project_root
        assert isinstance(service.project_root, Path)
    
    def test_init_with_string_path_converts_to_path(self):
        """Test initialization with string path converts to Path object."""
        # Arrange
        project_root_str = "/test/project"
        config = Mock(spec=Config)
        feedback = Mock(spec=UserFeedback)
        
        # Act
        service = ConcreteService(Path(project_root_str), config, feedback)
        
        # Assert
        assert service.project_root == Path(project_root_str)
        assert isinstance(service.project_root, Path)
    
    def test_init_preserves_config_object(self):
        """Test that config object is preserved as-is."""
        # Arrange
        project_root = Path("/test/project")
        config = Mock(spec=Config)
        config.some_attribute = "test_value"
        
        # Act
        service = ConcreteService(project_root, config)
        
        # Assert
        assert service.config is config
        assert service.config.some_attribute == "test_value"
    
    def test_init_with_different_service_classes_have_different_loggers(self):
        """Test that different service classes get loggers with their own names."""
        # Arrange
        class AnotherService(BaseService):
            pass
        
        project_root = Path("/test/project")
        config = Mock(spec=Config)
        
        # Act
        service1 = ConcreteService(project_root, config)
        service2 = AnotherService(project_root, config)
        
        # Assert
        assert service1.logger.name == "ConcreteService"
        assert service2.logger.name == "AnotherService"
        assert service1.logger != service2.logger
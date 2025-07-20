"""Base service class for common functionality."""

import logging
from abc import ABC
from pathlib import Path
from typing import Optional

from smart_test_generator.config import Config
from smart_test_generator.utils.user_feedback import UserFeedback

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """Base class for all services providing common functionality."""
    
    def __init__(self, project_root: Path, config: Config, feedback: Optional[UserFeedback] = None):
        self.project_root = project_root
        self.config = config
        self.feedback = feedback or UserFeedback()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Determine if we should use Rich UI or standard logging
        self.use_rich_ui = feedback is not None
    
    def _log_info(self, message: str):
        """Log info message."""
        if self.use_rich_ui:
            # Use Rich UI only - avoid duplicate logging
            self.feedback.info(message)
        else:
            # Fallback to standard logging if no Rich UI
            self.logger.info(message)
    
    def _log_success(self, message: str):
        """Log success message."""
        if self.use_rich_ui:
            self.feedback.success(message)
        else:
            self.logger.info(f"SUCCESS: {message}")
    
    def _log_warning(self, message: str, suggestion: Optional[str] = None):
        """Log warning message."""
        if self.use_rich_ui:
            self.feedback.warning(message, suggestion)
        else:
            self.logger.warning(message)
            if suggestion:
                self.logger.warning(f"Suggestion: {suggestion}")
    
    def _log_error(self, message: str, suggestion: Optional[str] = None):
        """Log error message."""
        if self.use_rich_ui:
            self.feedback.error(message, suggestion)
        else:
            self.logger.error(message)
            if suggestion:
                self.logger.error(f"Suggestion: {suggestion}")
                
    def _log_debug(self, message: str):
        """Log debug message."""
        if self.use_rich_ui and self.feedback.verbose:
            self.feedback.debug(message)
        else:
            self.logger.debug(message) 
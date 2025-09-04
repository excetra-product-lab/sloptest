import pytest
import os
from unittest.mock import Mock, patch, MagicMock

from smart_test_generator.core.llm_factory import LLMClientFactory
from smart_test_generator.generation.llm_clients import ClaudeAPIClient, AzureOpenAIClient
from smart_test_generator.utils.user_feedback import UserFeedback
from smart_test_generator.exceptions import AuthenticationError, ValidationError


class TestLLMClientFactory:
    """Test suite for LLMClientFactory."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_feedback = Mock(spec=UserFeedback)
        self.mock_cost_manager = Mock()
        
    @patch('smart_test_generator.core.llm_factory.Validator')
    @patch('smart_test_generator.core.llm_factory.ClaudeAPIClient')
    def test_create_client_with_claude_api_key_parameter(self, mock_claude_client, mock_validator):
        """Test creating Claude client with API key parameter."""
        # Arrange
        api_key = "test-claude-key"
        model = "claude-sonnet-4-20250514"
        mock_validator.validate_api_key.return_value = api_key
        mock_client_instance = Mock()
        mock_claude_client.return_value = mock_client_instance
        
        # Act
        result = LLMClientFactory.create_client(
            claude_api_key=api_key,
            claude_model=model,
            feedback=self.mock_feedback,
            cost_manager=self.mock_cost_manager
        )
        
        # Assert
        mock_validator.validate_api_key.assert_called_once_with(api_key, "Claude")
        mock_validator.validate_model_name.assert_called_once()
        mock_claude_client.assert_called_once_with(api_key, model, extended_thinking=False, thinking_budget=4096, cost_manager=self.mock_cost_manager, feedback=self.mock_feedback, config=None)
        self.mock_feedback.info.assert_called_once_with(f"Using Claude API with model: {model}")
        assert result == mock_client_instance
        
    @patch.dict(os.environ, {'CLAUDE_API_KEY': 'env-claude-key'})
    @patch('smart_test_generator.core.llm_factory.Validator')
    @patch('smart_test_generator.core.llm_factory.ClaudeAPIClient')
    def test_create_client_with_claude_api_key_from_environment(self, mock_claude_client, mock_validator):
        """Test creating Claude client with API key from environment variable."""
        # Arrange
        env_key = "env-claude-key"
        model = "claude-sonnet-4-20250514"
        mock_validator.validate_api_key.return_value = env_key
        mock_client_instance = Mock()
        mock_claude_client.return_value = mock_client_instance
        
        # Act
        result = LLMClientFactory.create_client(
            claude_model=model,
            feedback=self.mock_feedback,
            cost_manager=self.mock_cost_manager
        )
        
        # Assert
        mock_validator.validate_api_key.assert_called_once_with(env_key, "Claude")
        mock_claude_client.assert_called_once_with(env_key, model, extended_thinking=False, thinking_budget=4096, cost_manager=self.mock_cost_manager, feedback=self.mock_feedback, config=None)
        assert result == mock_client_instance
        
    @patch('smart_test_generator.core.llm_factory.Validator')
    @patch('smart_test_generator.core.llm_factory.AzureOpenAIClient')
    def test_create_client_with_azure_credentials(self, mock_azure_client, mock_validator):
        """Test creating Azure OpenAI client with valid credentials."""
        # Arrange
        endpoint = "https://test.openai.azure.com/"
        api_key = "test-azure-key"
        deployment = "test-deployment"
        mock_validator.validate_api_key.return_value = api_key
        mock_client_instance = Mock()
        mock_azure_client.return_value = mock_client_instance
        
        # Act
        result = LLMClientFactory.create_client(
            azure_endpoint=endpoint,
            azure_api_key=api_key,
            azure_deployment=deployment,
            feedback=self.mock_feedback,
            cost_manager=self.mock_cost_manager
        )
        
        # Assert
        mock_validator.validate_api_key.assert_called_once_with(api_key, "Azure OpenAI")
        mock_azure_client.assert_called_once_with(endpoint, api_key, deployment, extended_thinking=False, cost_manager=self.mock_cost_manager, feedback=self.mock_feedback, config=None)
        self.mock_feedback.info.assert_called_once_with("Using Azure OpenAI")
        assert result == mock_client_instance
        
    def test_create_client_with_no_credentials_raises_authentication_error(self):
        """Test that missing credentials raises AuthenticationError."""
        # Act & Assert
        with pytest.raises(AuthenticationError) as exc_info:
            LLMClientFactory.create_client(feedback=self.mock_feedback)
            
        assert "No LLM API credentials provided" in str(exc_info.value)
        assert "Claude API key" in exc_info.value.suggestion
        assert "Azure OpenAI credentials" in exc_info.value.suggestion
        
    def test_create_client_with_incomplete_azure_credentials_raises_authentication_error(self):
        """Test that incomplete Azure credentials raise AuthenticationError."""
        # Test missing deployment
        with pytest.raises(AuthenticationError):
            LLMClientFactory.create_client(
                azure_endpoint="https://test.openai.azure.com/",
                azure_api_key="test-key",
                feedback=self.mock_feedback
            )
            
        # Test missing API key
        with pytest.raises(AuthenticationError):
            LLMClientFactory.create_client(
                azure_endpoint="https://test.openai.azure.com/",
                azure_deployment="test-deployment",
                feedback=self.mock_feedback
            )
            
        # Test missing endpoint
        with pytest.raises(AuthenticationError):
            LLMClientFactory.create_client(
                azure_api_key="test-key",
                azure_deployment="test-deployment",
                feedback=self.mock_feedback
            )
            
    @patch('smart_test_generator.core.llm_factory.UserFeedback')
    @patch('smart_test_generator.core.llm_factory.Validator')
    @patch('smart_test_generator.core.llm_factory.ClaudeAPIClient')
    def test_create_client_creates_default_feedback_when_none_provided(self, mock_claude_client, mock_validator, mock_feedback_class):
        """Test that default UserFeedback is created when none provided."""
        # Arrange
        api_key = "test-claude-key"
        mock_validator.validate_api_key.return_value = api_key
        mock_client_instance = Mock()
        mock_claude_client.return_value = mock_client_instance
        mock_feedback_instance = Mock()
        mock_feedback_class.return_value = mock_feedback_instance
        
        # Act
        result = LLMClientFactory.create_client(claude_api_key=api_key)
        
        # Assert
        mock_feedback_class.assert_called_once_with()
        mock_feedback_instance.info.assert_called_once()
        
    @patch('smart_test_generator.core.llm_factory.Validator')
    def test_create_client_propagates_validation_error(self, mock_validator):
        """Test that ValidationError from validator is propagated."""
        # Arrange
        api_key = "invalid-key"
        mock_validator.validate_api_key.side_effect = ValidationError("Invalid API key format")
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            LLMClientFactory.create_client(
                claude_api_key=api_key,
                feedback=self.mock_feedback
            )
            
        assert "Invalid API key format" in str(exc_info.value)
        
    @patch('smart_test_generator.core.llm_factory.Validator')
    def test_create_client_propagates_authentication_error(self, mock_validator):
        """Test that AuthenticationError from validator is propagated."""
        # Arrange
        api_key = "test-key"
        mock_validator.validate_api_key.side_effect = AuthenticationError("API key authentication failed")
        
        # Act & Assert
        with pytest.raises(AuthenticationError) as exc_info:
            LLMClientFactory.create_client(
                claude_api_key=api_key,
                feedback=self.mock_feedback
            )
            
        assert "API key authentication failed" in str(exc_info.value)
        
    @patch('smart_test_generator.core.llm_factory.Validator')
    @patch('smart_test_generator.core.llm_factory.ClaudeAPIClient')
    def test_create_client_wraps_unexpected_exceptions_in_authentication_error(self, mock_claude_client, mock_validator):
        """Test that unexpected exceptions are wrapped in AuthenticationError."""
        # Arrange
        api_key = "test-key"
        mock_validator.validate_api_key.return_value = api_key
        mock_claude_client.side_effect = RuntimeError("Unexpected error")
        
        # Act & Assert
        with pytest.raises(AuthenticationError) as exc_info:
            LLMClientFactory.create_client(
                claude_api_key=api_key,
                feedback=self.mock_feedback
            )
            
        assert "Failed to initialize LLM client: Unexpected error" in str(exc_info.value)
        assert "Check your API credentials" in exc_info.value.suggestion
        
    @patch('smart_test_generator.core.llm_factory.Validator')
    @patch('smart_test_generator.core.llm_factory.ClaudeAPIClient')
    def test_create_client_with_custom_claude_model(self, mock_claude_client, mock_validator):
        """Test creating Claude client with custom model."""
        # Arrange
        api_key = "test-claude-key"
        custom_model = "claude-3-5-haiku-20241022"
        mock_validator.validate_api_key.return_value = api_key
        mock_client_instance = Mock()
        mock_claude_client.return_value = mock_client_instance
        
        # Act
        result = LLMClientFactory.create_client(
            claude_api_key=api_key,
            claude_model=custom_model,
            feedback=self.mock_feedback
        )
        
        # Assert
        mock_claude_client.assert_called_once_with(api_key, custom_model, extended_thinking=False, thinking_budget=4096, cost_manager=None, feedback=self.mock_feedback, config=None)
        self.mock_feedback.info.assert_called_once_with(f"Using Claude API with model: {custom_model}")
        
    @patch('smart_test_generator.core.llm_factory.Validator')
    def test_create_client_validates_azure_endpoint_format(self, mock_validator):
        """Test that Azure endpoint format is validated."""
        # Arrange
        invalid_endpoint = "invalid-endpoint"
        api_key = "test-key"
        deployment = "test-deployment"
        mock_validator.validate_api_key.return_value = api_key
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            LLMClientFactory.create_client(
                azure_endpoint=invalid_endpoint,
                azure_api_key=api_key,
                azure_deployment=deployment,
                feedback=self.mock_feedback
            )
            
        assert f"Invalid Azure OpenAI endpoint: {invalid_endpoint}" in str(exc_info.value)
        assert "should start with 'https://'" in exc_info.value.suggestion
        
    @patch.dict(os.environ, {}, clear=True)
    def test_create_client_with_no_environment_claude_key(self):
        """Test behavior when CLAUDE_API_KEY environment variable is not set."""
        # Act & Assert
        with pytest.raises(AuthenticationError):
            LLMClientFactory.create_client(feedback=self.mock_feedback)
            
    @patch('smart_test_generator.core.llm_factory.Validator')
    @patch('smart_test_generator.core.llm_factory.ClaudeAPIClient')
    def test_create_client_passes_none_cost_manager_when_not_provided(self, mock_claude_client, mock_validator):
        """Test that None is passed as cost_manager when not provided."""
        # Arrange
        api_key = "test-claude-key"
        mock_validator.validate_api_key.return_value = api_key
        mock_client_instance = Mock()
        mock_claude_client.return_value = mock_client_instance

        # Act
        LLMClientFactory.create_client(
            claude_api_key=api_key,
            feedback=self.mock_feedback
        )

        # Assert
        mock_claude_client.assert_called_once_with(api_key, "claude-sonnet-4-20250514", extended_thinking=False, thinking_budget=4096, cost_manager=None, feedback=self.mock_feedback, config=None)

    @patch('smart_test_generator.core.llm_factory.Validator')
    @patch('smart_test_generator.core.llm_factory.ClaudeAPIClient')
    def test_create_client_with_extended_thinking_enabled(self, mock_claude_client, mock_validator):
        """Test creating Claude client with extended thinking enabled."""
        # Arrange
        api_key = "test-claude-key"
        model = "claude-sonnet-4-20250514"
        thinking_budget = 8192
        mock_validator.validate_api_key.return_value = api_key
        mock_client_instance = Mock()
        mock_claude_client.return_value = mock_client_instance

        # Act
        result = LLMClientFactory.create_client(
            claude_api_key=api_key,
            claude_model=model,
            claude_extended_thinking=True,
            claude_thinking_budget=thinking_budget,
            feedback=self.mock_feedback
        )

        # Assert
        mock_claude_client.assert_called_once_with(api_key, model, extended_thinking=True, thinking_budget=thinking_budget, cost_manager=None, feedback=self.mock_feedback, config=None)
        self.mock_feedback.info.assert_called_once_with(f"Using Claude API with model: {model} (extended thinking enabled, budget: {thinking_budget} tokens)")
        assert result == mock_client_instance

    @patch('smart_test_generator.core.llm_factory.Validator')
    def test_create_client_extended_thinking_unsupported_model(self, mock_validator):
        """Test that extended thinking raises error for unsupported models."""
        # Arrange
        api_key = "test-claude-key"
        unsupported_model = "claude-3-5-haiku-20241022"
        mock_validator.validate_api_key.return_value = api_key

        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            LLMClientFactory.create_client(
                claude_api_key=api_key,
                claude_model=unsupported_model,
                claude_extended_thinking=True,
                feedback=self.mock_feedback
            )

        assert f"Extended thinking is not supported for model: {unsupported_model}" in str(exc_info.value)

    @patch('smart_test_generator.core.llm_factory.Validator')
    def test_create_client_extended_thinking_invalid_budget(self, mock_validator):
        """Test that invalid thinking budget raises error."""
        # Arrange
        api_key = "test-claude-key"
        model = "claude-sonnet-4-20250514"
        invalid_budget = 50000  # Too high
        mock_validator.validate_api_key.return_value = api_key

        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            LLMClientFactory.create_client(
                claude_api_key=api_key,
                claude_model=model,
                claude_extended_thinking=True,
                claude_thinking_budget=invalid_budget,
                feedback=self.mock_feedback
            )

        assert "Thinking budget must be between 1024 and 32000 tokens" in str(exc_info.value)

    @patch('smart_test_generator.core.llm_factory.Validator')
    @patch('smart_test_generator.core.llm_factory.ClaudeAPIClient')
    def test_create_client_extended_thinking_default_budget(self, mock_claude_client, mock_validator):
        """Test that default thinking budget is used when not specified."""
        # Arrange
        api_key = "test-claude-key"
        model = "claude-sonnet-4-20250514"
        mock_validator.validate_api_key.return_value = api_key
        mock_client_instance = Mock()
        mock_claude_client.return_value = mock_client_instance

        # Act
        result = LLMClientFactory.create_client(
            claude_api_key=api_key,
            claude_model=model,
            claude_extended_thinking=True,
            feedback=self.mock_feedback
        )

        # Assert
        mock_claude_client.assert_called_once_with(api_key, model, extended_thinking=True, thinking_budget=4096, cost_manager=None, feedback=self.mock_feedback, config=None)
        assert result == mock_client_instance
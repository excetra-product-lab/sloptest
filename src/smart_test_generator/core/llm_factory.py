"""Factory for creating LLM clients."""

import os
import subprocess
from pathlib import Path
from typing import Optional

from smart_test_generator.generation.llm_clients import LLMClient, ClaudeAPIClient, AzureOpenAIClient, BedrockClient
from smart_test_generator.utils.validation import Validator
from smart_test_generator.utils.user_feedback import UserFeedback
from smart_test_generator.exceptions import AuthenticationError, ValidationError


class LLMClientFactory:
    """Factory for creating appropriate LLM client instances."""
    
    @staticmethod
    def create_client(claude_api_key: Optional[str] = None,
                     claude_model: str = "claude-sonnet-4-20250514",
                     azure_endpoint: Optional[str] = None,
                     azure_api_key: Optional[str] = None,
                     azure_deployment: Optional[str] = None,
                     bedrock_role_arn: Optional[str] = None,
                     bedrock_inference_profile: Optional[str] = None,
                     bedrock_region: Optional[str] = None,
                     feedback: Optional[UserFeedback] = None,
                     cost_manager=None) -> LLMClient:
        """Create an LLM client based on provided credentials."""
        
        # Try to get Claude API key from environment if not provided
        if not claude_api_key:
            claude_api_key = os.environ.get("CLAUDE_API_KEY")
        
        if feedback is None:
            feedback = UserFeedback()
        
        try:
            if claude_api_key:
                return LLMClientFactory._create_claude_client(claude_api_key, claude_model, feedback, cost_manager)
            elif azure_endpoint and azure_api_key and azure_deployment:
                return LLMClientFactory._create_azure_client(azure_endpoint, azure_api_key, azure_deployment, feedback, cost_manager)
            elif bedrock_role_arn and bedrock_inference_profile:
                return LLMClientFactory._create_bedrock_client(
                    role_arn=bedrock_role_arn,
                    inference_profile=bedrock_inference_profile,
                    region=bedrock_region or "us-east-1",
                    feedback=feedback,
                    cost_manager=cost_manager,
                )
            else:
                raise AuthenticationError(
                    "No LLM API credentials provided",
                    suggestion="Provide either:\n" +
                    "  1. Claude API key: --claude-api-key or set CLAUDE_API_KEY environment variable\n" +
                    "  2. Azure OpenAI credentials: --endpoint, --api-key, and --deployment\n" +
                    "  3. AWS Bedrock: --bedrock-role-arn and --bedrock-inference-profile (optional --bedrock-region)"
                )
        except Exception as e:
            if isinstance(e, (AuthenticationError, ValidationError)):
                raise
            raise AuthenticationError(
                f"Failed to initialize LLM client: {e}",
                suggestion="Check your API credentials and try again."
            )
    
    @staticmethod
    def _create_claude_client(api_key: str, model: str, feedback: UserFeedback, cost_manager=None) -> ClaudeAPIClient:
        """Create a Claude API client."""
        # Validate Claude API key
        validated_key = Validator.validate_api_key(api_key, "Claude")
        
        # Validate model name if specified
        available_claude_models = [
            "claude-opus-4-20250514", 
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022", 
            "claude-3-5-haiku-20241022",
            "claude-3-haiku-20240307"
        ]
        if model:
            Validator.validate_model_name(model, available_claude_models)
        
        feedback.info(f"Using Claude API with model: {model}")
        return ClaudeAPIClient(validated_key, model, cost_manager, feedback)
    
    @staticmethod
    def _create_azure_client(endpoint: str, api_key: str, deployment: str, feedback: UserFeedback, cost_manager=None) -> AzureOpenAIClient:
        """Create an Azure OpenAI client."""
        # Validate Azure OpenAI credentials
        validated_key = Validator.validate_api_key(api_key, "Azure OpenAI")
        
        if not endpoint.startswith(('http://', 'https://')):
            raise ValidationError(
                f"Invalid Azure OpenAI endpoint: {endpoint}",
                suggestion="Endpoint should start with 'https://' (e.g., https://your-resource.openai.azure.com/)"
            )
        
        feedback.info("Using Azure OpenAI")
        return AzureOpenAIClient(endpoint, validated_key, deployment, cost_manager, feedback) 

    @staticmethod
    def _create_bedrock_client(*, role_arn: str, inference_profile: str, region: str, feedback: UserFeedback, cost_manager=None) -> BedrockClient:
        feedback.info(f"Using AWS Bedrock (inference profile: {inference_profile}, region: {region})")
        return BedrockClient(role_arn=role_arn, inference_profile=inference_profile, region=region, cost_manager=cost_manager, feedback=feedback)
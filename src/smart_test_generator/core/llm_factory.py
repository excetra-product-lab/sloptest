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
                     bedrock_model_id: Optional[str] = None,
                     bedrock_region: Optional[str] = None,
                     bedrock_profile: Optional[str] = None,
                     bedrock_require_credentials_path: Optional[str] = None,
                     bedrock_pcl_command: Optional[str] = None,
                     bedrock_pcl_inputs: Optional[dict] = None,
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
            elif bedrock_model_id:
                # Ensure credentials exist via PCL if path specified
                if bedrock_require_credentials_path:
                    LLMClientFactory._ensure_aws_credentials(
                        credentials_path=bedrock_require_credentials_path,
                        pcl_command=bedrock_pcl_command,
                        pcl_inputs=bedrock_pcl_inputs or {},
                        feedback=feedback,
                    )
                return LLMClientFactory._create_bedrock_client(bedrock_model_id, bedrock_region or "us-east-1", bedrock_profile, feedback, cost_manager)
            else:
                raise AuthenticationError(
                    "No LLM API credentials provided",
                    suggestion="Provide either:\n" +
                    "  1. Claude API key: --claude-api-key or set CLAUDE_API_KEY environment variable\n" +
                    "  2. Azure OpenAI credentials: --endpoint, --api-key, and --deployment\n" +
                    "  3. AWS Bedrock: --bedrock-model-id (and optional --bedrock-region/--bedrock-profile)"
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
        return ClaudeAPIClient(validated_key, model, cost_manager)
    
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
        return AzureOpenAIClient(endpoint, validated_key, deployment, cost_manager) 

    @staticmethod
    def _create_bedrock_client(model_id: str, region: str, profile: Optional[str], feedback: UserFeedback, cost_manager=None) -> BedrockClient:
        feedback.info(f"Using AWS Bedrock model: {model_id} ({region})")
        return BedrockClient(model_id=model_id, region=region, profile=profile, cost_manager=cost_manager)

    @staticmethod
    def _ensure_aws_credentials(credentials_path: str, pcl_command: Optional[str], pcl_inputs: dict, feedback: UserFeedback) -> None:
        """Ensure AWS credentials exist at a given path, else run PCL command to set them.

        credentials_path can be a file or directory. If it is a directory, '~/.aws/credentials' semantics apply.
        PCL command is executed with provided inputs as env vars appended to the command environment.
        """
        path = Path(os.path.expanduser(credentials_path)).resolve()
        # Determine if credentials exist: either file exists or default AWS resolution works
        exists = path.exists() and path.stat().st_size > 0
        if exists:
            return
        if not pcl_command:
            raise AuthenticationError(
                f"AWS credentials not found at {path}",
                suggestion="Provide --bedrock-pcl-command to bootstrap credentials or create them manually."
            )

        # Run the PCL command with inputs as environment variables
        env = os.environ.copy()
        inputs = dict(pcl_inputs or {})
        # If no inputs provided, prompt user to enter key=value pairs interactively
        if not inputs:
            if feedback:
                feedback.info("Credentials not found. Preparing to run PCL. Enter KEY=VALUE pairs for variables (blank line to finish):")
            try:
                while True:
                    line = input().strip()
                    if not line:
                        break
                    if '=' in line:
                        k, v = line.split('=', 1)
                        inputs[k.strip()] = v.strip()
            except EOFError:
                # Non-interactive environment; proceed without extra inputs
                pass
        for k, v in inputs.items():
            if v is not None:
                env[str(k)] = str(v)

        try:
            if feedback:
                feedback.info("Running credentials bootstrap command (PCL)")
            subprocess.run(pcl_command, shell=True, check=True, env=env)
        except subprocess.CalledProcessError as e:
            raise AuthenticationError(
                f"Failed to run credentials bootstrap command: {e}",
                suggestion="Verify the command and inputs."
            )
        # Re-check existence
        if not (path.exists() and path.stat().st_size > 0):
            raise AuthenticationError(
                f"AWS credentials still not found at {path} after running bootstrap",
                suggestion="Ensure the bootstrap actually writes credentials to the specified path."
            )
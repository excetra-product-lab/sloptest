"""Factory for creating LLM clients."""

import os
import subprocess
from pathlib import Path
from typing import Optional

from smart_test_generator.generation.llm_clients import LLMClient  # legacy ABC
from smart_test_generator.generation.orchestrator_client import OrchestratedLLMClient
from smart_test_generator.generation.clients import (
    OpenAITransport,
    AzureOpenAITransport,
    ClaudeTransport,
    BedrockTransport,
)
from smart_test_generator.utils.validation import Validator
from smart_test_generator.utils.user_feedback import UserFeedback
from smart_test_generator.exceptions import AuthenticationError, ValidationError


class LLMClientFactory:
    """Factory for creating appropriate LLM client instances."""
    
    @staticmethod
    def create_client(claude_api_key: Optional[str] = None,
                     claude_model: str = "claude-sonnet-4-20250514",
                     claude_extended_thinking: bool = True,
                     claude_thinking_budget: Optional[int] = None,
                     azure_endpoint: Optional[str] = None,
                     azure_api_key: Optional[str] = None,
                     azure_deployment: Optional[str] = None,
                     bedrock_role_arn: Optional[str] = None,
                     bedrock_inference_profile: Optional[str] = None,
                     bedrock_region: Optional[str] = None,
                     openai_api_key: Optional[str] = None,
                     openai_model: str = "gpt-4.1",
                     openai_extended_thinking: bool = False,
                     feedback: Optional[UserFeedback] = None,
                     cost_manager=None,
                     config=None) -> LLMClient:
        """Create an LLM client based on provided credentials."""
        
        # Try to get API keys from environment if not provided
        if not claude_api_key:
            claude_api_key = os.environ.get("CLAUDE_API_KEY")
        
        if not openai_api_key:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        if feedback is None:
            feedback = UserFeedback()
        
        try:
            if claude_api_key:
                return LLMClientFactory._create_claude_client(
                    claude_api_key, claude_model, claude_extended_thinking, claude_thinking_budget, feedback, cost_manager, config
                )
            elif openai_api_key:
                return LLMClientFactory._create_openai_client(
                    openai_api_key, openai_model, openai_extended_thinking, feedback, cost_manager, config
                )
            elif azure_endpoint and azure_api_key and azure_deployment:
                # For Azure OpenAI, be conservative with extended thinking to avoid API issues
                return LLMClientFactory._create_azure_client(azure_endpoint, azure_api_key, azure_deployment, 
                                                            claude_extended_thinking, feedback, cost_manager, config)
            elif bedrock_role_arn and bedrock_inference_profile:
                # For Bedrock, be conservative with extended thinking to avoid API issues  
                return LLMClientFactory._create_bedrock_client(
                    role_arn=bedrock_role_arn,
                    inference_profile=bedrock_inference_profile,
                    region=bedrock_region or "us-east-1",
                    extended_thinking=claude_extended_thinking,
                    feedback=feedback,
                    cost_manager=cost_manager,
                    config=config,
                )
            else:
                raise AuthenticationError(
                    "No LLM API credentials provided",
                    suggestion="Provide either:\n" +
                    "  1. Claude API key: --claude-api-key or set CLAUDE_API_KEY environment variable\n" +
                    "  2. OpenAI API key: --openai-api-key or set OPENAI_API_KEY environment variable\n" +
                    "  3. Azure OpenAI credentials: --endpoint, --api-key, and --deployment\n" +
                    "  4. AWS Bedrock: --bedrock-role-arn and --bedrock-inference-profile (optional --bedrock-region)"
                )
        except Exception as e:
            if isinstance(e, (AuthenticationError, ValidationError)):
                raise
            raise AuthenticationError(
                f"Failed to initialize LLM client: {e}",
                suggestion="Check your API credentials and try again."
            )
    
    @staticmethod
    def _create_claude_client(api_key: str, model: str, extended_thinking: bool, thinking_budget: Optional[int],
                            feedback: UserFeedback, cost_manager=None, config=None) -> LLMClient:
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

        # Validate extended thinking configuration
        if extended_thinking:
            # Extended thinking is available for these models (per Anthropic docs)
            extended_thinking_supported_models = [
                "claude-opus-4-1-20250805",
                "claude-opus-4-20250514",
                "claude-sonnet-4-20250514", 
                "claude-3-7-sonnet-20250219"
            ]
            if model not in extended_thinking_supported_models:
                feedback.warning(f"Extended thinking is not supported for model: {model}")
                feedback.warning(f"Falling back to standard mode. Supported models: {', '.join(extended_thinking_supported_models)}")
                extended_thinking = False
            else:
                # Validate thinking budget if provided (only for supported models)
                if thinking_budget is not None:
                    if not (1024 <= thinking_budget <= 32000):
                        raise ValidationError(
                            f"Thinking budget must be between 1024 and 32000 tokens, got: {thinking_budget}",
                            suggestion="Use a value between 1024 and 32000 tokens"
                        )
                else:
                    # Set default thinking budget (increased for better performance)
                    thinking_budget = 8192

        if extended_thinking:
            feedback.info(f"Using Claude API with model: {model} (extended thinking enabled, budget: {thinking_budget} tokens)")
        else:
            feedback.info(f"Using Claude API with model: {model}")

        transport = ClaudeTransport(api_key=validated_key, model=model)
        return OrchestratedLLMClient(
            transport=transport,
            model_name=model,
            cost_manager=cost_manager,
            feedback=feedback,
            config=config,
            extended_thinking=extended_thinking,
        )
    
    @staticmethod
    def _create_azure_client(endpoint: str, api_key: str, deployment: str, extended_thinking: bool, 
                           feedback: UserFeedback, cost_manager=None, config=None) -> LLMClient:
        """Create an Azure OpenAI client."""
        # Validate Azure OpenAI credentials
        validated_key = Validator.validate_api_key(api_key, "Azure OpenAI")
        
        if not endpoint.startswith(('http://', 'https://')):
            raise ValidationError(
                f"Invalid Azure OpenAI endpoint: {endpoint}",
                suggestion="Endpoint should start with 'https://' (e.g., https://your-resource.openai.azure.com/)"
            )
        
        # Azure OpenAI uses prompt-based extended thinking (no native API support)
        if extended_thinking:
            feedback.info("Using Azure OpenAI (extended thinking enabled via enhanced prompts)")
        else:
            feedback.info("Using Azure OpenAI")
            
        transport = AzureOpenAITransport(endpoint=endpoint, api_key=validated_key, deployment_name=deployment)
        return OrchestratedLLMClient(
            transport=transport,
            model_name=f"Azure:{deployment}",
            cost_manager=cost_manager,
            feedback=feedback,
            config=config,
            extended_thinking=extended_thinking,
        )

    @staticmethod
    def _create_bedrock_client(*, role_arn: str, inference_profile: str, region: str, extended_thinking: bool, 
                             feedback: UserFeedback, cost_manager=None, config=None) -> LLMClient:
        # Bedrock uses prompt-based extended thinking (Claude models on Bedrock support this)
        if extended_thinking:
            feedback.info(f"Using AWS Bedrock (inference profile: {inference_profile}, region: {region}, extended thinking enabled via enhanced prompts)")
        else:
            feedback.info(f"Using AWS Bedrock (inference profile: {inference_profile}, region: {region})")
            
        # Build ChatBedrock instance similar to previous implementation
        try:
            from langchain_aws import ChatBedrock
            import boto3
            import botocore.config as botocore_config
        except Exception as e:  # pragma: no cover
            raise AuthenticationError(
                f"AWS Bedrock initialization failed: {e}",
                suggestion="Install and configure AWS SDKs (boto3, langchain-aws).",
            )

        # Assume role and create runtime client
        base_session = boto3.Session(region_name=region)
        sts = base_session.client("sts", region_name=region)
        assumed = sts.assume_role(RoleArn=role_arn, RoleSessionName="bedrock")
        creds = assumed["Credentials"]
        cfg = botocore_config.Config(read_timeout=300, retries={"max_attempts": 3, "mode": "standard"})
        session = boto3.Session(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
            region_name=region,
        )
        bedrock_runtime = session.client("bedrock-runtime", config=cfg)
        chat = ChatBedrock(model_id=inference_profile, region_name=region, provider="anthropic", config=cfg, client=bedrock_runtime)
        transport = BedrockTransport(chat_bedrock=chat)
        return OrchestratedLLMClient(
            transport=transport,
            model_name=f"Bedrock:{inference_profile}",
            cost_manager=cost_manager,
            feedback=feedback,
            config=config,
            extended_thinking=extended_thinking,
        )
    
    @staticmethod
    def _create_openai_client(api_key: str, model: str, extended_thinking: bool, 
                            feedback: UserFeedback, cost_manager=None, config=None) -> LLMClient:
        """Create an OpenAI API client."""
        # Validate OpenAI API key
        validated_key = Validator.validate_api_key(api_key, "OpenAI")
        
        # Validate model name if specified - GPT-4.1 models only
        available_openai_models = [
            "gpt-4.1"
        ]
        if model:
            Validator.validate_model_name(model, available_openai_models)

        # OpenAI GPT-4.1 doesn't have native extended thinking like Claude, but we can use enhanced prompts
        if extended_thinking:
            feedback.info(f"Using OpenAI GPT-4.1 with model: {model} (extended thinking enabled via enhanced prompts)")
        else:
            feedback.info(f"Using OpenAI GPT-4.1 with model: {model}")

        transport = OpenAITransport(api_key=validated_key, model=model)
        return OrchestratedLLMClient(
            transport=transport,
            model_name=model,
            cost_manager=cost_manager,
            feedback=feedback,
            config=config,
            extended_thinking=extended_thinking,
        )
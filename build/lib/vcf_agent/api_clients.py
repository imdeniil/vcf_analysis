"""
API Clients for VCF Agent

This module contains client implementations for connecting to various LLM APIs:
- OpenAI API
- Cerebras API

The clients handle authentication, request formatting, and response parsing.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union, cast
from pathlib import Path

import openai
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
load_dotenv()

class APIClientError(Exception):
    """Base exception for API client errors."""
    pass

class CredentialManager:
    """Manages API credentials securely."""
    
    def __init__(self, credentials_file: Optional[str] = None):
        """
        Initialize the credential manager.
        
        Args:
            credentials_file: Optional path to JSON credentials file.
                              If None, will use environment variables.
        """
        self.credentials_file = credentials_file
        self.credentials = {}
        self._load_credentials()
    
    def _load_credentials(self) -> None:
        """Load credentials from file or environment variables."""
        # First try to load from credentials file if provided
        if self.credentials_file and Path(self.credentials_file).exists():
            try:
                with open(self.credentials_file, 'r') as f:
                    file_creds = json.load(f)
                    self.credentials.update(file_creds)
                    logger.info(f"Loaded credentials from file: {self.credentials_file}")
                    # If we loaded credentials from file, return early
                    return
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load credentials from file: {e}")
                raise APIClientError(f"Failed to load credentials: {e}")
        
        # If no file or file loading failed, try environment variables
        openai_key = os.environ.get("OPENAI_API_KEY")
        cerebras_key = os.environ.get("CEREBRAS_API_KEY")
        
        if openai_key:
            self.credentials["openai"] = {"api_key": openai_key}
            logger.info("Loaded OpenAI credentials from environment")
        
        if cerebras_key:
            self.credentials["cerebras"] = {"api_key": cerebras_key}
            logger.info("Loaded Cerebras credentials from environment")
    
    def get_credential(self, service: str, key: str = "api_key") -> str:
        """
        Get a credential for a specific service.
        
        Args:
            service: The service name (e.g., 'openai', 'cerebras')
            key: The credential key (default: 'api_key')
            
        Returns:
            The credential value
            
        Raises:
            APIClientError: If credential is not found
        """
        if service not in self.credentials:
            raise APIClientError(f"No credentials found for service: {service}")
        
        if key not in self.credentials[service]:
            raise APIClientError(f"No '{key}' found in {service} credentials")
        
        return self.credentials[service][key]


class OpenAIClient:
    """Client for OpenAI API."""
    
    def __init__(self, credential_manager: Optional[CredentialManager] = None):
        """
        Initialize the OpenAI client.
        
        Args:
            credential_manager: Optional credential manager.
                                If None, a new one will be created.
        """
        self.credential_manager = credential_manager or CredentialManager()
        self.client: Optional[OpenAI] = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the OpenAI client with API key."""
        try:
            api_key = self.credential_manager.get_credential("openai")
            self.client = OpenAI(api_key=api_key)
        except APIClientError as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test connection to OpenAI API.
        
        Returns:
            True if connection is successful, raises exception otherwise
        """
        try:
            if not self.client:
                raise APIClientError("OpenAI client not initialized")
            # Make a simple models list call to test the connection
            models = self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            raise APIClientError(f"Failed to connect to OpenAI API: {e}")
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Send a chat completion request to OpenAI.
        
        Args:
            messages: List of message objects with role and content
            model: Model to use (default: gpt-4o)
            temperature: Temperature for sampling (default: 0.7)
            max_tokens: Maximum tokens to generate (optional)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The API response
        """
        try:
            if not self.client:
                raise APIClientError("OpenAI client not initialized")
            
            # Cast to Any to avoid type issues
            client_any = cast(Any, self.client)
            
            response = client_any.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response
        except Exception as e:
            logger.error(f"OpenAI chat completion failed: {e}")
            raise APIClientError(f"OpenAI chat completion failed: {e}")


class CerebrasClient:
    """Client for Cerebras API."""
    
    def __init__(self, credential_manager: Optional[CredentialManager] = None):
        """
        Initialize the Cerebras client.
        
        Args:
            credential_manager: Optional credential manager.
                                If None, a new one will be created.
        """
        self.credential_manager = credential_manager or CredentialManager()
        self.api_key: Optional[str] = None
        self.base_url = os.environ.get(
            "CEREBRAS_API_URL", 
            "https://api.cerebras.ai/v1"
        )
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Cerebras client with API key."""
        try:
            self.api_key = self.credential_manager.get_credential("cerebras")
        except APIClientError as e:
            logger.error(f"Failed to initialize Cerebras client: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test connection to Cerebras API.
        
        Returns:
            True if connection is successful, raises exception otherwise
        
        Note:
            This is a mock implementation until actual Cerebras API details are available.
        """
        # This is a placeholder until actual Cerebras API details are available
        # In a real implementation, we would make a simple API call to test the connection
        logger.info("Testing connection to Cerebras API (mock implementation)")
        if not self.api_key:
            raise APIClientError("No API key for Cerebras")
        return True
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "cerebras-gpt",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to Cerebras.
        
        Args:
            messages: List of message objects with role and content
            model: Model to use (default: cerebras-gpt)
            temperature: Temperature for sampling (default: 0.7)
            max_tokens: Maximum tokens to generate (optional)
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The API response as a dictionary
        
        Note:
            This is a mock implementation until actual Cerebras API details are available.
        """
        # This is a placeholder until actual Cerebras API details are available
        logger.info(f"Making chat completion request to Cerebras API (mock): {model}")
        
        # Return a mock response
        mock_response = {
            "id": "mock-cerebras-response",
            "object": "chat.completion",
            "created": 1684936580,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"This is a mock response from Cerebras {model}."
                    }
                }
            ]
        }
        return mock_response 
"""Model-agnostic LLM client interface.

This module provides a unified interface for LLM providers (Gemini, OpenRouter)
with no vendor-specific logic leaking into other components.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import os
from dotenv import load_dotenv

load_dotenv()


class LLMClient(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Dict with 'content' (str) and 'raw_response' (Any) keys
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available and configured."""
        pass


class GeminiClient(LLMClient):
    """Google Gemini LLM client implementation."""

    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        """
        Initialize Gemini client.

        Args:
            model_name: Gemini model name (e.g., 'gemini-3-flash-preview', 'gemini-3-pro')
        """
        try:
            from google import genai
            self.genai = genai
        except ImportError:
            raise ImportError(
                "google-genai not installed. Install with: pip install google-genai"
            )

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        # Initialize client with API key
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate chat completion using Gemini."""
        # Convert messages to the format expected by google-genai
        # Build contents list with proper role mapping
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Map roles to google-genai format
            if role == "system":
                # System messages: prepend to first user message or handle separately
                if contents and contents[-1].get("role") == "user":
                    # Prepend system message to existing user message
                    system_text = f"System: {content}\n\n"
                    if "parts" in contents[-1] and contents[-1]["parts"]:
                        contents[-1]["parts"][0]["text"] = system_text + contents[-1]["parts"][0]["text"]
                    else:
                        contents[-1]["parts"] = [{"text": system_text + content}]
                else:
                    # Add as first user message with system prefix
                    contents.append({"role": "user", "parts": [{"text": f"System: {content}"}]})
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})

        # Configure generation parameters
        config = {
            "temperature": temperature,
        }
        if max_tokens:
            config["max_output_tokens"] = max_tokens

        try:
            # Use the new API format - try different possible API structures
            try:
                # Try: client.models.generate_content(model, contents, config)
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
            except (AttributeError, TypeError):
                # Fallback: try alternative API structure
                try:
                    response = self.client.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=config,
                    )
                except Exception:
                    # Last fallback: try with model object
                    model = self.client.models.get(self.model_name)
                    response = model.generate_content(
                        contents=contents,
                        config=config,
                    )

            # Extract text from response - handle different response structures
            content = ""
            if hasattr(response, 'text'):
                content = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                if hasattr(response.candidates[0], 'content'):
                    if hasattr(response.candidates[0].content, 'parts'):
                        content = response.candidates[0].content.parts[0].text
                    elif hasattr(response.candidates[0].content, 'text'):
                        content = response.candidates[0].content.text
            elif isinstance(response, dict):
                content = response.get("text", "") or response.get("content", "")
            
            if not content:
                content = str(response)

            return {
                "content": content if content else "",
                "raw_response": response,
            }
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")

    def is_available(self) -> bool:
        """Check if Gemini is configured."""
        return os.getenv("GEMINI_API_KEY") is not None


class OpenRouterClient(LLMClient):
    """OpenRouter LLM client implementation (OpenAI-compatible API)."""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize OpenRouter client.

        Args:
            model_name: Model name (defaults to OPENROUTER_MODEL env var)
        """
        try:
            from openai import OpenAI
            self.OpenAI = OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

        self.model_name = model_name or os.getenv(
            "OPENROUTER_MODEL", "openrouter/gpt-5-mini"
        )
        self.client = self.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate chat completion using OpenRouter."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content
            return {
                "content": content or "",
                "raw_response": response,
            }
        except Exception as e:
            raise RuntimeError(f"OpenRouter API error: {str(e)}")

    def is_available(self) -> bool:
        """Check if OpenRouter is configured."""
        return os.getenv("OPENROUTER_API_KEY") is not None


def get_llm_client(fallback: bool = True, model_name: Optional[str] = None) -> LLMClient:
    """
    Factory function to get an available LLM client.

    Args:
        fallback: If True, fallback to OpenRouter if Gemini is unavailable
        model_name: Optional model name (defaults: 'gemini-3-flash-preview' for Gemini, 'openrouter/gpt-5-mini' for OpenRouter)

    Returns:
        An initialized LLMClient instance

    Raises:
        RuntimeError: If no LLM client is available
    """
    # Try Gemini first
    try:
        client = GeminiClient(model_name=model_name or "gemini-3-flash-preview")
        if client.is_available():
            return client
    except (ValueError, ImportError) as e:
        if not fallback:
            raise RuntimeError(f"Gemini not available: {str(e)}")

    # Fallback to OpenRouter
    if fallback:
        try:
            client = OpenRouterClient(model_name=model_name or "openrouter/gpt-5-mini")
            if client.is_available():
                return client
        except (ValueError, ImportError) as e:
            pass

    raise RuntimeError(
        "No LLM client available. Please configure GEMINI_API_KEY or OPENROUTER_API_KEY."
    )

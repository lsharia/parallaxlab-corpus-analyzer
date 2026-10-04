"""Reusable client for the DeepSeek chat completions API."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_OUTPUT_TOKENS = 1024


class LLMError(RuntimeError):
    """Base class for LLM client failures."""


class MissingAPIKeyError(LLMError):
    """Raised when the required API key is not configured."""


class AuthenticationError(LLMError):
    """Raised when the provider rejects API authentication."""


class RateLimitError(LLMError):
    """Raised when the provider rate-limits a request."""


class RequestTimeoutError(LLMError):
    """Raised when a request times out."""


class LLMConnectionError(LLMError):
    """Raised when the provider cannot be reached."""


class APIError(LLMError):
    """Raised for other unsuccessful provider responses."""


class InvalidAPIResponseError(LLMError):
    """Raised when the provider response does not match the expected schema."""


class OutputLimitError(APIError):
    """Raised when the request or generated output exceeds a token limit."""


class LLMClient:
    """Call DeepSeek's OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_output_tokens: int | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        configured_key = api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY")
        if not configured_key or not configured_key.strip():
            raise MissingAPIKeyError("Set the DEEPSEEK_API_KEY environment variable.")

        self._api_key = configured_key
        self.model = model or os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = self._float_setting(
            timeout if timeout is not None else os.getenv("DEEPSEEK_TIMEOUT"),
            DEFAULT_TIMEOUT,
            "DEEPSEEK_TIMEOUT",
        )
        self.max_output_tokens = self._int_setting(
            max_output_tokens
            if max_output_tokens is not None
            else os.getenv("DEEPSEEK_MAX_OUTPUT_TOKENS"),
            DEFAULT_MAX_OUTPUT_TOKENS,
            "DEEPSEEK_MAX_OUTPUT_TOKENS",
        )
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero")

        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.Client(timeout=self.timeout)

    @staticmethod
    def _float_setting(value: Any, default: float, name: str) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a number") from error

    @staticmethod
    def _int_setting(value: Any, default: int, name: str) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be an integer") from error

    @staticmethod
    def _normalize_messages(
        prompt_or_messages: str | Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        if isinstance(prompt_or_messages, str):
            if not prompt_or_messages.strip():
                raise ValueError("prompt must not be empty")
            return [{"role": "user", "content": prompt_or_messages}]
        if not isinstance(prompt_or_messages, Sequence) or not prompt_or_messages:
            raise ValueError("messages must be a non-empty sequence")
        if any(not isinstance(message, Mapping) for message in prompt_or_messages):
            raise ValueError("each message must be a mapping")
        return list(prompt_or_messages)

    @staticmethod
    def _error_text(payload: Any) -> str:
        try:
            return json.dumps(payload).lower()
        except (TypeError, ValueError):
            return ""

    def generate(self, prompt_or_messages: str | Sequence[Mapping[str, Any]]) -> str:
        """Generate and return text for a prompt or chat messages."""
        messages = self._normalize_messages(prompt_or_messages)
        try:
            response = self._http_client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_output_tokens,
                },
                timeout=self.timeout,
            )
        except httpx.TimeoutException as error:
            raise RequestTimeoutError("The LLM request timed out.") from error
        except httpx.ConnectError as error:
            raise LLMConnectionError("Could not connect to the LLM provider.") from error
        except httpx.RequestError as error:
            raise LLMConnectionError("The LLM request failed to reach the provider.") from error

        try:
            payload = response.json()
        except ValueError as error:
            if response.status_code >= 400:
                payload = None
            else:
                raise InvalidAPIResponseError("The LLM provider returned invalid JSON.") from error

        if response.status_code >= 400:
            error_text = self._error_text(payload)
            if response.status_code in {401, 403}:
                raise AuthenticationError("The LLM provider rejected the API credentials.")
            if response.status_code == 429:
                raise RateLimitError("The LLM provider rate-limited the request.")
            if response.status_code in {408, 504}:
                raise RequestTimeoutError("The LLM provider timed out the request.")
            if response.status_code == 413 or self._is_output_limit_error(error_text):
                raise OutputLimitError("The LLM request exceeded a provider token or size limit.")
            raise APIError(f"The LLM provider returned HTTP {response.status_code}.")

        if not isinstance(payload, Mapping):
            raise InvalidAPIResponseError("The LLM provider returned an unexpected response.")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise InvalidAPIResponseError("The LLM provider response has no completion choice.")
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            raise OutputLimitError("The generated response reached the output token limit.")
        message = choice.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise InvalidAPIResponseError("The LLM provider response has no text content.")
        return message["content"]

    @staticmethod
    def _is_output_limit_error(error_text: str) -> bool:
        indicators = (
            "context_length",
            "context length",
            "token_limit",
            "token limit",
            "max_tokens",
            "maximum context",
            "too many tokens",
        )
        return any(indicator in error_text for indicator in indicators)

    def close(self) -> None:
        """Close the underlying HTTP client when this instance owns it."""
        if self._owns_http_client:
            self._http_client.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
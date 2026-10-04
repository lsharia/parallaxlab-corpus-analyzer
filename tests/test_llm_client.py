"""Mocked unit tests for the DeepSeek-compatible LLM client."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest
import src.llm_client as llm_client_module

from src.llm_client import (
    APIError,
    AuthenticationError,
    InvalidAPIResponseError,
    LLMClient,
    LLMConnectionError,
    MissingAPIKeyError,
    OutputLimitError,
    RateLimitError,
    RequestTimeoutError,
)


def make_client(
    handler: Callable[[httpx.Request], httpx.Response], **settings: Any
) -> tuple[LLMClient, httpx.Client]:
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    llm_client = LLMClient(api_key="test-secret", http_client=http_client, **settings)
    return llm_client, http_client


def success_response(text: str = "Generated answer.") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            ]
        },
    )


def test_generate_sends_prompt_and_returns_generated_text() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["body"] = request.read()
        return success_response("A generated answer.")

    client, transport = make_client(handler)
    try:
        assert client.generate("Answer this.") == "A generated answer."
    finally:
        transport.close()

    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["authorization"] == "Bearer test-secret"
    assert b'"role":"user"' in captured["body"]
    assert b'"content":"Answer this."' in captured["body"]


def test_generate_accepts_messages_and_configurable_model_and_timeout() -> None:
    captured: dict[str, Any] = {}

    class RecordingClient:
        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            captured.update(url=url, **kwargs)
            return success_response()

        def close(self) -> None:
            pass

    client = LLMClient(
        api_key="test-secret",
        model="deepseek-reasoner",
        timeout=7.5,
        max_output_tokens=222,
        http_client=RecordingClient(),  # type: ignore[arg-type]
    )

    assert client.generate([{"role": "system", "content": "Be concise."}]) == "Generated answer."
    assert captured["json"] == {
        "model": "deepseek-reasoner",
        "messages": [{"role": "system", "content": "Be concise."}],
        "max_tokens": 222,
    }
    assert captured["timeout"] == 7.5


def test_environment_configures_credentials_and_api_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class RecordingClient:
        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            captured.update(url=url, **kwargs)
            return success_response()

        def close(self) -> None:
            pass

    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-secret")
    monkeypatch.setenv("DEEPSEEK_MODEL", "environment-model")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("DEEPSEEK_TIMEOUT", "4.25")
    monkeypatch.setenv("DEEPSEEK_MAX_OUTPUT_TOKENS", "321")
    client = LLMClient(http_client=RecordingClient())  # type: ignore[arg-type]

    assert client.generate("prompt") == "Generated answer."
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer environment-secret"
    assert captured["json"]["model"] == "environment-model"
    assert captured["json"]["max_tokens"] == 321
    assert captured["timeout"] == 4.25


def test_project_dotenv_is_loaded_before_reading_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class RecordingClient:
        def post(self, url: str, **kwargs: Any) -> httpx.Response:
            captured.update(url=url, **kwargs)
            return success_response()

        def close(self) -> None:
            pass

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    def fake_load_dotenv(path: Any) -> None:
        captured["dotenv_path"] = path
        monkeypatch.setenv("DEEPSEEK_API_KEY", "dotenv-test-key")

    monkeypatch.setattr(llm_client_module, "load_dotenv", fake_load_dotenv)
    client = LLMClient(http_client=RecordingClient())  # type: ignore[arg-type]

    assert client.generate("prompt") == "Generated answer."
    assert captured["dotenv_path"] == llm_client_module.Path(__file__).resolve().parents[1] / ".env"
    assert captured["headers"]["Authorization"] == "Bearer dotenv-test-key"


def test_missing_api_key_is_reported_without_exposing_a_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(llm_client_module, "load_dotenv", lambda path: False)

    with pytest.raises(MissingAPIKeyError, match="DEEPSEEK_API_KEY") as error:
        LLMClient()

    assert "test-secret" not in str(error.value)


@pytest.mark.parametrize("status_code", [401, 403])
def test_authentication_errors_are_classified(status_code: int) -> None:
    client, transport = make_client(lambda request: httpx.Response(status_code, json={}))
    try:
        with pytest.raises(AuthenticationError):
            client.generate("prompt")
    finally:
        transport.close()


def test_rate_limit_is_classified() -> None:
    client, transport = make_client(lambda request: httpx.Response(429, json={}))
    try:
        with pytest.raises(RateLimitError):
            client.generate("prompt")
    finally:
        transport.close()


def test_timeout_is_classified() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    client, transport = make_client(handler)
    try:
        with pytest.raises(RequestTimeoutError):
            client.generate("prompt")
    finally:
        transport.close()


def test_connection_failure_is_classified() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    client, transport = make_client(handler)
    try:
        with pytest.raises(LLMConnectionError):
            client.generate("prompt")
    finally:
        transport.close()


def test_other_api_errors_do_not_expose_response_body() -> None:
    client, transport = make_client(
        lambda request: httpx.Response(500, json={"error": "private provider detail"})
    )
    try:
        with pytest.raises(APIError, match="HTTP 500") as error:
            client.generate("prompt")
    finally:
        transport.close()

    assert "private provider detail" not in str(error.value)
    assert "test-secret" not in str(error.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"unexpected": True}),
        httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
        httpx.Response(200, content=b"not-json"),
    ],
)
def test_malformed_responses_are_classified(response: httpx.Response) -> None:
    client, transport = make_client(lambda request: response)
    try:
        with pytest.raises(InvalidAPIResponseError):
            client.generate("prompt")
    finally:
        transport.close()


def test_output_token_limit_is_classified_from_finish_reason() -> None:
    response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]},
    )
    client, transport = make_client(lambda request: response)
    try:
        with pytest.raises(OutputLimitError):
            client.generate("prompt")
    finally:
        transport.close()


def test_provider_token_limit_error_is_classified() -> None:
    client, transport = make_client(
        lambda request: httpx.Response(
            400, json={"error": {"code": "context_length_exceeded"}}
        )
    )
    try:
        with pytest.raises(OutputLimitError):
            client.generate("prompt")
    finally:
        transport.close()
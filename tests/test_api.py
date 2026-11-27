from unittest.mock import AsyncMock
import pytest
from fastapi.testclient import TestClient
from httpx2 import Request as SDKRequest
from openai import APIConnectionError, APIError, APITimeoutError
from app import main as api
from app.config import Settings
from app.llm import LLMResponseError
from app.schemas import EvaluationResult

# искусственная строка для проверки отсутствия утечек
PRIVATE_MARKER = "SYNTHETIC_PRIVATE_DETAIL"
# используется только для создания объектов ошибок SDK
SDK_REQUEST = SDKRequest(
    "POST",
    "http://localhost:11434/v1/chat/completions",
)

@pytest.fixture
def payload():
    return {
        "question": "Во сколько открывается тестовый сервис?",
        "answer": "В 09:00",
        "reference_context": [
            "Тестовый сервис открывается в 09:00"
        ],
        "context": [
            "Тестовый сервис открывается в 09:00"
        ],
        "temperature": 0,
    }

@pytest.fixture
def api_client(monkeypatch):
    # не читаем локальный .env: настройки теста заданы явно
    settings = Settings(
        _env_file=None,
        base_url="http://localhost:11434/v1",
        model="test-model",
        api_key="test-placeholder",
        temperature=0,
        timeout_seconds=1,
    )
    llm_mock = AsyncMock(
        return_value=EvaluationResult(
            score=1,
            tags=[],
            explanation="Ответ подтверждается контекстом.",
        )
    )
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(api, "evaluate_answer", llm_mock)
    with TestClient(api.app) as client:
        yield client, llm_mock

def test_evaluate_returns_result(api_client, payload):
    client, llm_mock = api_client
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    assert response.json() == {
        "score": 1,
        "tags": [],
        "explanation": "Ответ подтверждается контекстом.",
    }
    llm_mock.assert_awaited_once()
    forwarded_request = llm_mock.await_args.kwargs["request"]
    assert forwarded_request.model_dump() == payload

def test_invalid_request_does_not_call_llm(api_client, payload):
    client, llm_mock = api_client
    payload["temperature"] = 3
    payload["answer"] = PRIVATE_MARKER
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_request"
    assert response.json()["detail"]["fields"] == [
        {
            "field": "body.temperature",
            "type": "less_than_equal",
        }
    ]
    assert PRIVATE_MARKER not in response.text
    llm_mock.assert_not_awaited()

@pytest.mark.parametrize(
    "error,expected_status,expected_code",
    [
        (
            APITimeoutError(request=SDK_REQUEST),
            504,
            "llm_timeout",
        ),
        (
            APIConnectionError(
                message=PRIVATE_MARKER,
                request=SDK_REQUEST,
            ),
            503,
            "llm_unavailable",
        ),
        (
            LLMResponseError(PRIVATE_MARKER),
            502,
            "invalid_llm_response",
        ),
        (
            APIError(
                message=PRIVATE_MARKER,
                request=SDK_REQUEST,
                body=None,
            ),
            502,
            "llm_api_error",
        ),
    ],
    ids=[
        "timeout",
        "connection-error",
        "invalid-llm-response",
        "provider-error",
    ],
)
def test_llm_errors_are_safe(
    api_client,
    payload,
    error,
    expected_status,
    expected_code,
):
    client, llm_mock = api_client
    llm_mock.side_effect = error
    response = client.post("/evaluate", json=payload)
    assert response.status_code == expected_status
    assert response.json()["detail"]["code"] == expected_code
    assert response.headers["content-type"].startswith(
        "application/json"
    )
    assert PRIVATE_MARKER not in response.text
    llm_mock.assert_awaited_once()
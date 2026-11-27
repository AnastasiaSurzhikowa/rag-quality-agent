from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from openai import APIConnectionError, APIError, APITimeoutError
from app.config import get_settings
from app.llm import LLMResponseError, create_llm_client, evaluate_answer
from app.schemas import EvaluationRequest, EvaluationResult

@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    application.state.settings = settings
    async with create_llm_client(settings) as client:
        application.state.llm_client = client
        yield

app = FastAPI(
    title="RAG Quality Agent",
    description="Прототип API оценки одного ответа RAG-системы.",
    version="0.1.0",
    lifespan=lifespan,
)

@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    fields = [
        {
            "field": ".".join(str(part) for part in item["loc"]),
            "type": item["type"],
        }
        for item in error.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "code": "invalid_request",
                "message": "Проверьте поля запроса.",
                "fields": fields,
            }
        },
    )

@app.post(
    "/evaluate",
    response_model=EvaluationResult,
    summary="Оценить один ответ RAG-системы",
    responses={
        422: {"description": "Некорректные входные данные."},
        502: {"description": "Ошибка сервера LLM или невалидный ответ."},
        503: {"description": "Не удалось подключиться к LLM."},
        504: {"description": "Превышено время ожидания LLM."},
    },
)
async def evaluate(
    payload: EvaluationRequest,
    request: Request,
) -> EvaluationResult:
    try:
        return await evaluate_answer(
            request=payload,
            client=request.app.state.llm_client,
            settings=request.app.state.settings,
        )
    except APITimeoutError:
        raise HTTPException(
            status_code=504,
            detail={
                "code": "llm_timeout",
                "message": "Превышено время ожидания LLM.",
            },
        ) from None
    except APIConnectionError:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "llm_unavailable",
                "message": "Не удалось подключиться к LLM.",
            },
        ) from None
    except LLMResponseError:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "invalid_llm_response",
                "message": "LLM не вернула корректный результат оценки.",
            },
        ) from None
    except APIError:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "llm_api_error",
                "message": "Сервер LLM вернул ошибку.",
            },
        ) from None
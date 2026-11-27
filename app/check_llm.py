import asyncio
from time import perf_counter
from openai import APIConnectionError, APIError, APITimeoutError
from app.config import get_settings
from app.llm import LLMResponseError, create_llm_client, evaluate_answer
from app.schemas import EvaluationRequest

async def main() -> None:
    settings = get_settings()
    request = EvaluationRequest(
        question="Во сколько открывается тестовый сервис?",
        answer="Тестовый сервис открывается в 09:00",
        reference_context=[
            "Тестовый сервис работает ежедневно с 09:00 до 18:00"
        ],
        context=[
            "Тестовый сервис работает ежедневно с 09:00 до 18:00"
        ],
        temperature=0.0,
    )
    started = perf_counter()
    try:
        async with create_llm_client(settings) as client:
            result = await evaluate_answer(request, client, settings)
    except APITimeoutError:
        raise SystemExit("Ошибка: превышено время ожидания LLM") from None
    except APIConnectionError:
        raise SystemExit("Ошибка: не удалось подключиться к Ollama") from None
    except APIError:
        raise SystemExit("Ошибка: сервер LLM отклонил запрос") from None
    except LLMResponseError as error:
        raise SystemExit(f"Ошибка: {error}") from None
    elapsed = perf_counter() - started
    print(f"Model: {settings.model}")
    print(f"Temperature: {request.temperature}")
    print(f"Elapsed: {elapsed:.2f} s")
    print(result.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())
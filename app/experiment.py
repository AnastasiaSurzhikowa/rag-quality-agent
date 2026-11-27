import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from openai import APIError
from app.config import get_settings
from app.llm import (
    SYSTEM_PROMPT,
    LLMResponseError,
    create_llm_client,
    evaluate_answer,
)
from app.schemas import EvaluationRequest

QUESTION = "Во сколько открывается и закрывается тестовый сервис?"
CONTEXT = [
    "Тестовый сервис работает ежедневно с 09:00 до 18:00"
]
CASES = [
    (
        "correct",
        "Тестовый сервис открывается в 09:00 и закрывается в 18:00",
        1,
    ),
    (
        "distortion",
        "Тестовый сервис открывается в 10:00 и закрывается в 18:00",
        0,
    ),
    (
        "incomplete",
        "Тестовый сервис открывается в 09:00",
        0,
    ),
]
TEMPERATURES = [0.0, 0.5, 1.0]
REPEATS = 2

async def main() -> None:
    settings = get_settings()
    output_dir = (
        Path(__file__).resolve().parent.parent
        / "artifacts"
        / "structured-output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = output_dir / f"experiment_{timestamp}.json"
    artifact = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "model": settings.model,
        "timeout_seconds": settings.timeout_seconds,
        "prompt_sha256": hashlib.sha256(
            SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "question": QUESTION,
        "reference_context": CONTEXT,
        "context": CONTEXT,
        "temperatures": TEMPERATURES,
        "repeats": REPEATS,
        "runs": [],
    }
    total = len(CASES) * len(TEMPERATURES) * REPEATS
    async with create_llm_client(settings) as client:
        for case_id, answer, expected_score in CASES:
            for temperature in TEMPERATURES:
                for repeat in range(1, REPEATS + 1):
                    request = EvaluationRequest(
                        question=QUESTION,
                        answer=answer,
                        reference_context=CONTEXT,
                        context=CONTEXT,
                        temperature=temperature,
                    )
                    record = {
                        "case_id": case_id,
                        "answer": answer,
                        "expected_score": expected_score,
                        "temperature": temperature,
                        "repeat": repeat,
                    }
                    started = perf_counter()
                    try:
                        result = await evaluate_answer(
                            request=request,
                            client=client,
                            settings=settings,
                        )
                    except (APIError, LLMResponseError) as error:
                        record["status"] = "error"
                        record["error_type"] = type(error).__name__
                    else:
                        record["status"] = "valid"
                        record["result"] = result.model_dump()
                        record["score_matches_expected"] = (
                            result.score == expected_score
                        )
                    record["elapsed_seconds"] = round(
                        perf_counter() - started,
                        3,
                    )
                    artifact["runs"].append(record)
                    # cохраняем прогресс после каждой попытки
                    output_path.write_text(
                        json.dumps(
                            artifact,
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    print(
                        f"{len(artifact['runs'])}/{total}: "
                        f"{case_id}, T={temperature}, "
                        f"repeat={repeat}, "
                        f"status={record['status']}, "
                        f"time={record['elapsed_seconds']} s"
                    )
    print("\nРезультаты по температурам:")
    for temperature in TEMPERATURES:
        rows = [
            row
            for row in artifact["runs"]
            if row["temperature"] == temperature
        ]
        valid = [row for row in rows if row["status"] == "valid"]
        matches = sum(
            row.get("score_matches_expected", False)
            for row in rows
        )
        average_time = (
            f"{sum(row['elapsed_seconds'] for row in valid) / len(valid):.2f} s"
            if valid
            else "нет успешных ответов"
        )
        print(
            f"T={temperature}: "
            f"валидных {len(valid)}/{len(rows)}, "
            f"совпадений оценки {matches}/{len(rows)}, "
            f"среднее время успешных ответов: {average_time}"
        )
    print(f"\nФайл результатов: {output_path.name}")
    print("Папка: artifacts/structured-output")

if __name__ == "__main__":
    asyncio.run(main())
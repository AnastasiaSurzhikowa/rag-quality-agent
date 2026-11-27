import pytest
from pydantic import ValidationError
from app.schemas import EvaluationRequest, EvaluationResult

def test_request_accepts_empty_answer_and_contexts():
    request = EvaluationRequest(
        question="  Когда открывается сервис?  ",
        answer="",
        reference_context=[],
        context=[],
    )
    assert request.question == "Когда открывается сервис?"
    assert request.answer == ""
    assert request.reference_context == []
    assert request.context == []
    assert request.temperature == 0.0

@pytest.mark.parametrize(
    "changes",
    [
        {"question": "   "},
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"temperature": float("nan")},
        {"context": "Чанк вместо списка"},
        {"reference_context": [123]},
        {"reference_answer": "Устаревшее поле"},
    ],
    ids=[
        "blank-question",
        "negative-temperature",
        "temperature-above-limit",
        "nan-temperature",
        "context-is-not-list",
        "chunk-is-not-string",
        "unexpected-field",
    ],
)
def test_request_rejects_invalid_data(changes):
    data = {
        "question": "Когда открывается сервис?",
        "answer": "В 09:00",
        "reference_context": ["Сервис открывается в 09:00"],
        "context": ["Сервис открывается в 09:00"],
    }
    data.update(changes)
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(data)

@pytest.mark.parametrize(
    "score,tags",
    [
        (1, []),
        (0, ["not_full", "hallucination"]),
    ],
    ids=["correct-answer", "answer-with-errors"],
)
def test_valid_result_can_be_serialized_and_read(score, tags):
    result = EvaluationResult(
        score=score,
        tags=tags,
        explanation="Краткое обоснование оценки",
    )
    restored = EvaluationResult.model_validate_json(
        result.model_dump_json()
    )
    assert restored == result

@pytest.mark.parametrize(
    "score",
    [2, -1, "1", True, 1.0],
    ids=["too-large", "negative", "string", "boolean", "float"],
)
def test_result_rejects_invalid_score(score):
    with pytest.raises(ValidationError):
        EvaluationResult(
            score=score,
            tags=[],
            explanation="Краткое обоснование оценки",
        )

@pytest.mark.parametrize(
    "score,tags",
    [
        (1, ["not_full"]),
        (0, []),
        (0, ["unknown_error"]),
        (0, ["not_full", "not_full"]),
    ],
    ids=[
        "tags-for-correct-answer",
        "missing-tags-for-error",
        "unknown-tag",
        "duplicate-tags",
    ],
)
def test_result_rejects_invalid_tags(score, tags):
    with pytest.raises(ValidationError):
        EvaluationResult(
            score=score,
            tags=tags,
            explanation="Краткое обоснование оценки",
        )

@pytest.mark.parametrize("explanation", ["", "   "])
def test_result_rejects_blank_explanation(explanation):
    with pytest.raises(ValidationError):
        EvaluationResult(
            score=1,
            tags=[],
            explanation=explanation,
        )
from typing import Literal, Self
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# теги ошибок
ErrorTag = Literal[
    "wrong_chunks",
    "not_full",
    "hallucination",
    "fact_distortion",
    "verbose",
]

class EvaluationRequest(BaseModel):
    """входные данные для оценки одного ответа rag-системы"""
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )
    question: str = Field(
        min_length=1,
        description="Вопрос пользователя.",
    )
    answer: str = Field(
        description="Ответ анализируемой RAG-системы.",
    )
    reference_context: list[str] = Field(
        description="Тексты эталонных чанков из разметки.",
    )
    context: list[str] = Field(
        description="Тексты чанков, найденных RAG-системой.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0,
        le=2,
        allow_inf_nan=False,
        description="Температура генерации от 0 до 2.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Вопрос не должен быть пустым.")
        return cleaned

class EvaluationResult(BaseModel):
    """структурированный результат оценки ответа"""
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )
    score: int = Field(
        ge=0,
        le=1,
        description="1 - ответ корректен; 0 - обнаружена ошибка.",
    )
    tags: list[ErrorTag] = Field(
        description="Теги ошибок без повторов.",
    )
    explanation: str = Field(
        min_length=1,
        description="Краткое обоснование оценки.",
    )

    @field_validator("tags")
    @classmethod
    def validate_unique_tags(
        cls,
        value: list[ErrorTag],
    ) -> list[ErrorTag]:
        if len(value) != len(set(value)):
            raise ValueError("Теги ошибок не должны повторяться.")
        return value

    @field_validator("explanation")
    @classmethod
    def validate_explanation(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Обоснование не должно быть пустым.")
        return cleaned

    @model_validator(mode="after")
    def validate_score_and_tags(self) -> Self:
        if self.score == 1 and self.tags:
            raise ValueError(
                "При score=1 список tags должен быть пустым."
            )
        if self.score == 0 and not self.tags:
            raise ValueError(
                "При score=0 нужен хотя бы один тег ошибки."
            )
        return self
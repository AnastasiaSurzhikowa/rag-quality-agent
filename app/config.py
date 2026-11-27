from functools import lru_cache
from pathlib import Path
from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """настройки подключения к LLM"""
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8-sig",
        env_prefix="LLM_",
        extra="forbid",
        hide_input_in_errors=True,
    )
    base_url: AnyHttpUrl
    model: str = Field(min_length=1)
    api_key: SecretStr
    temperature: float = Field(
        default=0.0,
        ge=0,
        le=2,
        allow_inf_nan=False,
    )
    timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        allow_inf_nan=False,
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("название модели не должно быть пустым")

        return cleaned

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
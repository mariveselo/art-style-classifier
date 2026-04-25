from __future__ import annotations

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str = Field(..., description="Название предсказанного класса.")
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Вероятность, возвращённая моделью.",
    )


class PredictionResponse(BaseModel):
    predictions: list[Prediction] = Field(
        ..., description="Топ предсказаний модели для загруженного изображения."
    )
    model_name: str = Field(..., description="Имя загруженного артефакта модели.")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Описание ошибки.")

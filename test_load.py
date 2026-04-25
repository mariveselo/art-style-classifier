"""Проверочный тест сервиса классификации.

Проверяет:
1. Модель загружается без ошибок.
2. Инференс на синтетическом изображении возвращает корректно
   отформатированный результат (список из top_k пар метка/вероятность).
3. Пустой ввод поднимает InvalidImageError.
4. Невалидные байты поднимают InvalidImageError.

Запуск:
    python test_load.py          # из папки art_classifier_service с активным .venv
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _make_test_image_bytes(width: int = 100, height: int = 100) -> bytes:
    """Создаёт минимальное синтетическое JPEG-изображение в памяти."""
    pixels = np.full((height, width, 3), 128, dtype=np.uint8)
    img = Image.fromarray(pixels)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _check(condition: bool, message: str) -> None:
    if not condition:
        print(f"ОШИБКА: {message}")
        sys.exit(1)


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from app.inference import InvalidImageError, get_model_error, is_model_ready, predict_image

    # 1. Модель должна быть загружена
    print("1. Проверка загрузки модели...")
    _check(is_model_ready(), f"Модель не загружена: {get_model_error()}")
    print("   Успешно")

    # 2. Инференс на синтетическом изображении
    print("2. Инференс на синтетическом изображении...")
    results = predict_image(_make_test_image_bytes(), top_k=3)
    _check(isinstance(results, list), "predict_image должна возвращать список")
    _check(len(results) == 3, f"Ожидалось 3 результата, получено {len(results)}")
    for label, prob in results:
        _check(isinstance(label, str) and label, "Метка класса должна быть непустой строкой")
        _check(0.0 <= prob <= 1.0, f"Вероятность {prob} вне диапазона [0, 1]")
    print("   Успешно. Топ-3 предсказания:")
    for rank, (label, prob) in enumerate(results, start=1):
        print(f"      {rank}. {label}: {prob:.4f}")

    # 3. Пустой ввод
    print("3. Пустой ввод должен поднимать InvalidImageError...")
    try:
        predict_image(b"", top_k=3)
        _check(False, "Пустой ввод не поднял исключение")
    except InvalidImageError:
        print("   Успешно")

    # 4. Невалидные байты
    print("4. Невалидные байты должны поднимать InvalidImageError...")
    try:
        predict_image(b"not an image", top_k=3)
        _check(False, "Невалидные байты не подняли исключение")
    except InvalidImageError:
        print("   Успешно")

    print("\nВсе проверки пройдены.")


if __name__ == "__main__":
    main()

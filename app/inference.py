import importlib
import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    import tensorflow as tf
except Exception as exc:  # pragma: no cover
    tf = None
    _TF_IMPORT_ERROR = exc
else:
    _TF_IMPORT_ERROR = None


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "Xception_production.keras"
META_PATH = MODEL_DIR / "model_meta.json"

_MODEL: Any = None
_CLASS_NAMES: list[str] = []
_IMG_SIZE: tuple[int, int] = (1, 1)
_MODEL_NAME = MODEL_PATH.stem
_PREPROCESS_INPUT: Any = None
_MODEL_LOAD_ERROR: Exception | None = None

_RESAMPLE_LANCZOS = Image.Resampling.LANCZOS


class ModelNotReadyError(RuntimeError):
    """Модель не загружена или её файлы недоступны."""


class InvalidImageError(ValueError):
    """Загруженный файл не является корректным изображением."""


def _import_callable(import_path: str):
    # model_meta.json может хранить путь как tf.keras.… — заменяем префикс
    path = import_path
    if path.startswith("tf."):
        path = "tensorflow." + path[3:]
    module_name, attr_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, attr_name, None)
    if fn is None or not callable(fn):
        raise ModelNotReadyError(
            f"Не удалось импортировать функцию предобработки: {import_path}."
        )
    return fn


def _parse_img_size(raw_size: Any) -> tuple[int, int]:
    """Принимает целое число или список из двух чисел, возвращает (ш, в)."""
    if isinstance(raw_size, int):
        return (raw_size, raw_size)
    if (
        isinstance(raw_size, (list, tuple))
        and len(raw_size) == 2
        and all(isinstance(item, int) for item in raw_size)
    ):
        return (int(raw_size[0]), int(raw_size[1]))
    raise ModelNotReadyError(
        "В model_meta.json поле img_size должно быть числом или списком из двух чисел."
    )


def _load_metadata() -> dict[str, Any]:
    if not META_PATH.exists():
        raise ModelNotReadyError(
            f"Файл метаданных не найден: {META_PATH}. "
            "Поместите model_meta.json в папку model/."
        )

    with META_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    class_names = metadata.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ModelNotReadyError(
            "В model_meta.json отсутствует непустой список class_names."
        )

    if not all(isinstance(item, str) and item.strip() for item in class_names):
        raise ModelNotReadyError("Список class_names должен состоять из строк.")

    return metadata


def _initialize_model() -> None:
    """Загружает модель при старте. Ошибки сохраняются, сервис не падает."""
    global _MODEL, _CLASS_NAMES, _IMG_SIZE, _MODEL_NAME, _PREPROCESS_INPUT, _MODEL_LOAD_ERROR

    try:
        if _TF_IMPORT_ERROR is not None or tf is None:
            raise ModelNotReadyError(
                "TensorFlow не импортируется. Установите зависимости из requirements.txt."
            ) from _TF_IMPORT_ERROR

        if not MODEL_PATH.exists():
            raise ModelNotReadyError(
                f"Файл модели не найден: {MODEL_PATH}. "
                "Поместите Xception_production.keras в папку model/."
            )

        metadata = _load_metadata()
        _CLASS_NAMES = metadata["class_names"]

        raw_img_size = metadata.get("img_size")
        if raw_img_size is None:
            raise ModelNotReadyError("В model_meta.json отсутствует поле img_size.")
        _IMG_SIZE = _parse_img_size(raw_img_size)

        _MODEL_NAME = str(metadata.get("model_name") or MODEL_PATH.stem)

        raw_preprocess = metadata.get("preprocess_input")
        if not raw_preprocess:
            raise ModelNotReadyError("В model_meta.json отсутствует поле preprocess_input.")
        _PREPROCESS_INPUT = _import_callable(str(raw_preprocess))

        _MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
        _MODEL_LOAD_ERROR = None
        logger.info("Модель '%s' загружена (%d классов).", _MODEL_NAME, len(_CLASS_NAMES))

    except Exception as exc:
        _MODEL_LOAD_ERROR = exc if isinstance(exc, ModelNotReadyError) else ModelNotReadyError(
            f"Не удалось загрузить модель: {exc}"
        )
        logger.warning("Инициализация модели пропущена: %s", _MODEL_LOAD_ERROR)


def _ensure_model_ready() -> None:
    if _MODEL_LOAD_ERROR is not None or _MODEL is None or _PREPROCESS_INPUT is None:
        if _MODEL_LOAD_ERROR is not None:
            raise ModelNotReadyError(str(_MODEL_LOAD_ERROR)) from _MODEL_LOAD_ERROR
        raise ModelNotReadyError("Модель недоступна для инференса.")


def _prepare_image(image_bytes: bytes) -> np.ndarray:
    """Декодирует байты, исправляет EXIF-ориентацию, ресайзит и применяет preprocess."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb_image = ImageOps.exif_transpose(image).convert("RGB")
    except UnidentifiedImageError as exc:
        logger.warning("Загруженный файл не является корректным изображением.")
        raise InvalidImageError("Не удалось распознать файл как изображение.") from exc
    except OSError as exc:
        logger.warning("Не удалось открыть изображение.")
        raise InvalidImageError("Изображение повреждено или имеет неверный формат.") from exc

    resized = rgb_image.resize(_IMG_SIZE, _RESAMPLE_LANCZOS)
    array = np.asarray(resized, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)
    return _PREPROCESS_INPUT(batch)


def predict_image(image_bytes: bytes, top_k: int = 3) -> list[tuple[str, float]]:
    """Возвращает топ-K предсказаний для изображения в виде (класс, вероятность)."""
    _ensure_model_ready()
    batch = _prepare_image(image_bytes)

    try:
        raw_predictions = _MODEL(batch, training=False).numpy()
    except Exception as exc:
        logger.exception("Предсказание модели завершилось ошибкой.")
        raise RuntimeError("Во время инференса произошла внутренняя ошибка.") from exc

    probabilities = np.asarray(raw_predictions).squeeze()
    if probabilities.ndim != 1:
        raise RuntimeError("Модель вернула выход неожиданной размерности.")

    requested_top_k = max(1, min(top_k, len(_CLASS_NAMES)))
    top_indices = np.argsort(probabilities)[::-1][:requested_top_k]

    return [(_CLASS_NAMES[i], float(probabilities[i])) for i in top_indices]


def is_model_ready() -> bool:
    return _MODEL_LOAD_ERROR is None and _MODEL is not None and _PREPROCESS_INPUT is not None


def get_model_name() -> str:
    return _MODEL_NAME


def get_model_error() -> str | None:
    return None if _MODEL_LOAD_ERROR is None else str(_MODEL_LOAD_ERROR)


_initialize_model()

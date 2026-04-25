import asyncio
import logging
import os
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.inference import (
    InvalidImageError,
    ModelNotReadyError,
    get_model_error,
    get_model_name,
    is_model_ready,
    predict_image,
)
from app.schemas import ErrorResponse, Prediction, PredictionResponse


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 МБ
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
LOCAL_ORIGIN_REGEX = r"https?://(localhost|127\.0\.0\.1)(:\d+)?$"


def _parse_allowed_origins() -> list[str]:
    raw = os.getenv("ART_CLASSIFIER_ALLOWED_ORIGINS", "")
    return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]


app = FastAPI(
    title="Art Classifier Service",
    description="FastAPI-сервис для классификации художественного стиля картин.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_allowed_origins(),
    allow_origin_regex=LOCAL_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started_at = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Необработанная ошибка при выполнении %s %s (%.2f ms)",
            request.method,
            request.url.path,
            (perf_counter() - started_at) * 1000,
        )
        raise
    logger.info(
        "%s %s -> %s (%.2f ms)",
        request.method,
        request.url.path,
        response.status_code,
        (perf_counter() - started_at) * 1000,
    )
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/", include_in_schema=False)
async def serve_index() -> RedirectResponse:
    return RedirectResponse(url="/static/index.html")


@app.get("/health", include_in_schema=False)
async def healthcheck() -> dict[str, str]:
    if not is_model_ready():
        raise HTTPException(
            status_code=503,
            detail=get_model_error() or "Модель ещё не готова к работе.",
        )
    return {"status": "ok", "model_name": get_model_name()}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Классифицировать загруженное изображение картины.",
)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    content_type = (file.content_type or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Разрешены только файлы изображений с content-type image/*.",
        )

    try:
        image_bytes = await file.read(MAX_FILE_SIZE + 1)
    finally:
        await file.close()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Загруженный файл пуст.")

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Размер файла превышает 10 МБ.")

    try:
        predictions = await asyncio.to_thread(predict_image, image_bytes, top_k=3)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Инференс завершился ошибкой.")
        raise HTTPException(
            status_code=500,
            detail="Не удалось выполнить инференс модели.",
        ) from exc

    return PredictionResponse(
        predictions=[
            Prediction(label=label, probability=probability)
            for label, probability in predictions
        ],
        model_name=get_model_name(),
    )

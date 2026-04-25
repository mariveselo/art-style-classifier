"""Создание Grad-CAM-карт для папки с изображениями."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "model" / "Xception_production.keras"
DEFAULT_META_PATH = PROJECT_DIR / "model" / "model_meta.json"
IMG_SIZE = (299, 299)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Создать Grad-CAM-карты для изображений из указанной папки."
    )
    parser.add_argument("input_dir", type=Path, help="Папка с исходными изображениями.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Путь к файлу модели .keras.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_META_PATH,
        help="Путь к model_meta.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "research" / "gradcam_output",
        help="Папка для отдельных Grad-CAM-карт.",
    )
    parser.add_argument(
        "--sheet",
        type=Path,
        default=PROJECT_DIR / "research" / "images" / "gradcam_grid.png",
        help="Путь к итоговому контактному листу.",
    )
    return parser.parse_args()


def jet(values: np.ndarray) -> np.ndarray:
    red = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    return np.stack([red, green, blue], axis=-1)


def build_gradcam_models(model, tf_module):
    backbone = next(
        layer for layer in model.layers if isinstance(layer, tf_module.keras.Model)
    )
    last_conv = next(
        layer.name for layer in reversed(backbone.layers) if len(layer.output.shape) == 4
    )

    conv_model = tf_module.keras.Model(
        backbone.input, backbone.get_layer(last_conv).output
    )
    head_input = tf_module.keras.Input(shape=conv_model.output.shape[1:])
    output = head_input
    after_backbone = False

    for layer in model.layers:
        if layer is backbone:
            after_backbone = True
            continue
        if after_backbone:
            output = layer(output)

    head_model = tf_module.keras.Model(head_input, output)
    return backbone.name, last_conv, conv_model, head_model


def make_gradcam(
    batch, conv_model, head_model, tf_module, class_index: int | None = None
):
    with tf_module.GradientTape() as tape:
        conv_output = conv_model(batch)
        tape.watch(conv_output)
        predictions = head_model(conv_output, training=False)
        if class_index is None:
            class_index = int(tf_module.argmax(predictions[0]))
        score = predictions[:, class_index]

    gradients = tape.gradient(score, conv_output)
    weights = tf_module.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf_module.nn.relu(
        tf_module.reduce_sum(conv_output[0] * weights, axis=-1)
    )
    heatmap = (heatmap / (tf_module.reduce_max(heatmap) + 1e-8)).numpy()
    return heatmap, predictions.numpy()[0]


def save_contact_sheet(rows, sheet_path: Path) -> None:
    width, height = IMG_SIZE
    padding, label_height, columns = 8, 20, 2
    cell_width = width * 2 + padding
    row_count = (len(rows) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (
            cell_width * columns + padding * (columns + 1),
            (height + label_height + padding) * row_count + padding,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    for index, (name, prediction, probability, original, overlay) in enumerate(rows):
        row, column = divmod(index, columns)
        x = padding + column * (cell_width + padding)
        y = padding + row * (height + label_height + padding)
        draw.text(
            (x, y),
            f"{name[:30]}  ->  {prediction} {probability * 100:.0f}%",
            fill="black",
        )
        canvas.paste(Image.fromarray(original), (x, y + label_height))
        canvas.paste(Image.fromarray(overlay), (x + width + padding, y + label_height))

    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(sheet_path)


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"Папка с изображениями не найдена: {args.input_dir}")
    if not args.model.is_file():
        raise SystemExit(f"Модель не найдена: {args.model}")
    if not args.metadata.is_file():
        raise SystemExit(f"Метаданные не найдены: {args.metadata}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    from tensorflow.keras.applications.xception import preprocess_input

    print("Загрузка модели...")
    model = tf.keras.models.load_model(args.model, compile=False)
    class_names = json.loads(args.metadata.read_text(encoding="utf-8"))["class_names"]
    backbone_name, last_conv, conv_model, head_model = build_gradcam_models(model, tf)
    print("Backbone:", backbone_name, "| целевой слой:", last_conv)

    files = sorted(
        path for path in args.input_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise SystemExit(f"В папке нет поддерживаемых изображений: {args.input_dir}")

    rows = []
    print(f"\nНайдено изображений: {len(files)}\n")
    for path in files:
        with Image.open(path) as image:
            original = np.asarray(image.convert("RGB").resize(IMG_SIZE))

        batch = preprocess_input(np.expand_dims(original.astype(np.float32), axis=0))
        heatmap, probabilities = make_gradcam(batch, conv_model, head_model, tf)
        prediction_index = int(np.argmax(probabilities))
        top_indices = np.argsort(probabilities)[::-1][:3]
        prediction = class_names[prediction_index]

        top3 = ", ".join(
            f"{class_names[index]} {probabilities[index] * 100:.0f}%"
            for index in top_indices
        )
        print(
            f"{path.name[:48]:48} -> {prediction:22} "
            f"{probabilities[prediction_index] * 100:5.1f}% | top3: {top3}"
        )

        resized_heatmap = Image.fromarray(np.uint8(255 * heatmap)).resize(
            IMG_SIZE, Image.Resampling.BILINEAR
        )
        heatmap_array = np.asarray(resized_heatmap) / 255.0
        overlay = np.uint8(original * 0.6 + jet(heatmap_array) * 255 * 0.4)
        Image.fromarray(overlay).save(args.output_dir / f"{path.stem}__gradcam.png")
        rows.append(
            (
                path.name,
                prediction,
                probabilities[prediction_index],
                original,
                overlay,
            )
        )

    save_contact_sheet(rows, args.sheet)
    print("\nОтдельные карты:", args.output_dir)
    print("Общий лист:", args.sheet)


if __name__ == "__main__":
    main()

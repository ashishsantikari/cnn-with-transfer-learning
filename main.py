from pathlib import Path
from typing import Any
import base64
import json
import os

import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename
from model_cache import ModelCacheManager

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

MODELS_ROOT = Path(os.getenv("MODELS_ROOT", "models"))
DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
CACHE_TTL_SECONDS = int(os.getenv("MODEL_CACHE_TTL_SECONDS", "300"))
PREDICTION_IMAGE_SIZE = 224

cache_manager = ModelCacheManager(ttl_seconds=CACHE_TTL_SECONDS)


def _slugify(value: str) -> str:
    safe_chars = "abcdefghijklmnopqrstuvwxyz0123456789-"
    lowered = value.lower().replace(" ", "-").replace("_", "-")
    slug = "".join(ch for ch in lowered if ch in safe_chars)
    return slug or "model"


def discover_models() -> list[dict[str, str]]:
    models: list[dict[str, str]] = []
    if not MODELS_ROOT.exists():
        return models

    # Hugging Face-style model folders (config + safetensors)
    for config_path in sorted(MODELS_ROOT.glob("**/config.json")):
        model_dir = config_path.parent
        weights_path = model_dir / "model.safetensors"
        if not weights_path.exists():
            continue
        rel = model_dir.relative_to(MODELS_ROOT)
        model_id = _slugify(f"hf-{rel.as_posix()}")
        models.append(
            {
                "id": model_id,
                "name": f"{model_dir.name} (ViT)",
                "kind": "transformers",
                "path": str(model_dir),
            }
        )

    # Keras models: support both .h5 and .keras files
    keras_candidates = sorted(MODELS_ROOT.glob("**/*.h5")) + sorted(MODELS_ROOT.glob("**/*.keras"))
    for keras_path in keras_candidates:
        resolved_keras = keras_path

        # Some exports create a directory ending with .h5 containing the real file.
        if keras_path.is_dir() and keras_path.suffix == ".h5":
            nested = keras_path / keras_path.name
            if nested.exists() and nested.is_file() and nested.suffix == ".h5":
                resolved_keras = nested
            else:
                # Ignore .h5 directories that do not contain a valid model file.
                continue

        # Keep only actual files after resolution.
        if not resolved_keras.is_file():
            continue

        rel = resolved_keras.relative_to(MODELS_ROOT)
        model_id = _slugify(f"keras-{rel.as_posix()}")
        display_name = f"{resolved_keras.stem} (Keras)"
        if resolved_keras.parent.name.lower() == "convnext" and resolved_keras.stem.lower() == "model":
            display_name = "ConvNext"
        models.append(
            {
                "id": model_id,
                "name": display_name,
                "kind": "keras",
                "path": str(resolved_keras),
            }
        )

    # Deduplicate in case of path collisions.
    dedup: dict[str, dict[str, str]] = {}
    for item in models:
        dedup[item["id"]] = item
    return list(dedup.values())


def get_selected_model(model_id: str | None = None) -> tuple[dict[str, str], list[dict[str, str]]]:
    models = discover_models()
    if not models:
        raise FileNotFoundError(
            f"No models found under '{MODELS_ROOT}'. Add a ViT folder, .h5 model, or .keras model file."
        )

    selected_id = model_id or DEFAULT_MODEL_ID
    if selected_id:
        for item in models:
            if item["id"] == selected_id:
                return item, models

    return models[0], models


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_class_names_for_transformer(model_dir: Path) -> list[str]:
    model_config_path = model_dir / "config.json"
    if not model_config_path.exists():
        return []

    with model_config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    id2label = cfg.get("id2label") or {}
    if not id2label:
        return []

    sorted_items = sorted(id2label.items(), key=lambda x: int(x[0]))
    return [label for _, label in sorted_items]


def get_transformer_bundle(model_meta: dict[str, str]) -> dict[str, Any]:
    cache_manager.purge_stale_model_cache()
    cache_key = model_meta["id"]
    cached = cache_manager.get_cached_bundle(cache_key)
    if cached is not None:
        return cached

    model_dir = Path(model_meta["path"])
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found at '{model_dir}'.")

    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'transformers'. Install with: pip install transformers torch"
        ) from exc

    model = AutoModelForImageClassification.from_pretrained(str(model_dir))
    processor = AutoImageProcessor.from_pretrained(str(model_dir))
    model.eval()

    bundle = {
        "kind": "transformers",
        "model": model,
        "processor": processor,
        "class_names": load_class_names_for_transformer(model_dir),
    }
    return cache_manager.store_cached_bundle(cache_key, bundle)


def _load_keras_class_names(h5_path: Path, output_size: int) -> list[str]:
    # ConvNeXt export convention: model folder contains label_map.json.
    label_map_file = h5_path.parent / "label_map.json"
    if label_map_file.exists():
        with label_map_file.open("r", encoding="utf-8") as f:
            label_map = json.load(f)

        index_to_label = label_map.get("index_to_label")
        if isinstance(index_to_label, dict) and index_to_label:
            try:
                ordered = sorted(index_to_label.items(), key=lambda x: int(x[0]))
                labels = [str(v) for _, v in ordered]
                if labels:
                    return labels
            except Exception:
                pass

    labels_file = h5_path.with_suffix(".labels.json")
    if labels_file.exists():
        with labels_file.open("r", encoding="utf-8") as f:
            labels = json.load(f)
        if isinstance(labels, list) and labels:
            return [str(x) for x in labels]
        if isinstance(labels, dict) and labels:
            try:
                ordered = sorted(labels.items(), key=lambda x: int(x[0]))
                return [str(v) for _, v in ordered]
            except Exception:
                pass

    # Alternate convention: a generic labels file in the model folder.
    alt_labels_file = h5_path.parent / "labels.json"
    if alt_labels_file.exists():
        with alt_labels_file.open("r", encoding="utf-8") as f:
            labels = json.load(f)
        if isinstance(labels, list) and labels:
            return [str(x) for x in labels]
        if isinstance(labels, dict) and labels:
            try:
                ordered = sorted(labels.items(), key=lambda x: int(x[0]))
                return [str(v) for _, v in ordered]
            except Exception:
                pass

    # Fallback to neutral class names when no explicit labels are available.
    return [f"Class_{idx}" for idx in range(output_size)]


def get_keras_bundle(model_meta: dict[str, str]) -> dict[str, Any]:
    cache_manager.purge_stale_model_cache()
    cache_key = model_meta["id"]
    cached = cache_manager.get_cached_bundle(cache_key)
    if cached is not None:
        return cached

    h5_path = Path(model_meta["path"])
    if not h5_path.exists():
        raise FileNotFoundError(f"Keras model file not found at '{h5_path}'.")

    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'tensorflow'. Install with: pip install tensorflow"
        ) from exc

    model = load_model(h5_path)
    output_shape = model.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    output_size = int(output_shape[-1]) if output_shape and output_shape[-1] else 1

    bundle = {
        "kind": "keras",
        "model": model,
        "class_names": _load_keras_class_names(h5_path, max(output_size, 1)),
    }
    return cache_manager.store_cached_bundle(cache_key, bundle)


def get_model_bundle(model_meta: dict[str, str]) -> dict[str, Any]:
    if model_meta["kind"] == "transformers":
        return get_transformer_bundle(model_meta)
    if model_meta["kind"] == "keras":
        return get_keras_bundle(model_meta)
    raise ValueError(f"Unsupported model kind: {model_meta['kind']}")


def preprocess_image(file_storage):
    img = Image.open(file_storage.stream).convert("RGB")
    return img.resize((PREDICTION_IMAGE_SIZE, PREDICTION_IMAGE_SIZE))


def build_image_preview(file_storage) -> str | None:
    file_storage.stream.seek(0)
    raw_bytes = file_storage.read()
    file_storage.stream.seek(0)
    if not raw_bytes:
        return None

    mime_type = file_storage.mimetype or "image/jpeg"
    encoded = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _ensure_probability_vector(raw_output: np.ndarray) -> np.ndarray:
    arr = np.array(raw_output, dtype=np.float32).flatten()
    if arr.size == 0:
        raise RuntimeError("Model returned empty prediction output.")

    if arr.size == 1:
        p1 = float(np.clip(arr[0], 0.0, 1.0))
        return np.array([1.0 - p1, p1], dtype=np.float32)

    if np.min(arr) < 0 or np.max(arr) > 1.0:
        exps = np.exp(arr - np.max(arr))
        arr = exps / np.sum(exps)
    return arr

def _predict_transformer(bundle: dict[str, Any], image: Image.Image) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'torch'. Install with: pip install torch") from exc

    model = bundle["model"]
    processor = bundle["processor"]
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        return torch.softmax(logits, dim=-1)[0].cpu().numpy()


def _predict_keras(bundle: dict[str, Any], image: Image.Image) -> np.ndarray:
    model = bundle["model"]
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    raw = model.predict(arr, verbose=0)[0]
    return _ensure_probability_vector(raw)


def run_prediction(file_storage, model_meta: dict[str, str]):
    bundle = get_model_bundle(model_meta)
    class_names: list[str] = bundle.get("class_names") or []
    image = preprocess_image(file_storage)

    if bundle["kind"] == "transformers":
        probabilities = _predict_transformer(bundle, image)
    else:
        probabilities = _predict_keras(bundle, image)

    top_idx = int(np.argmax(probabilities))
    top_confidence = float(probabilities[top_idx])
    raw_top_label = class_names[top_idx] if top_idx < len(class_names) else f"Class {top_idx}"
    top_label = raw_top_label

    top3_idx = np.argsort(probabilities)[-3:][::-1]
    top3 = []
    for idx in top3_idx:
        raw_class_name = class_names[int(idx)] if int(idx) < len(class_names) else f"Class {int(idx)}"
        # class_name = to_meaningful_label_text(raw_class_name)
        top3.append({"label": raw_class_name, "confidence": round(float(probabilities[int(idx)]) * 100, 2)})

    return {
        "label": top_label,
        "confidence": round(top_confidence * 100, 2),
        "top3": top3,
    }


def get_feedback_mood(confidence: float) -> str:
    if confidence >= 80:
        return "win"
    if confidence >= 50:
        return "thinking"
    return "oops"


@app.route("/", methods=["GET"])
def home():
    cache_manager.purge_stale_model_cache()
    selected = request.args.get("model_id")
    selected_model, models = get_selected_model(selected)
    return render_template("index.html", models=models, selected_model_id=selected_model["id"])


@app.route("/predict", methods=["POST"])
def predict():
    cache_manager.purge_stale_model_cache()
    result = None
    error = None
    uploaded_name = None
    image_preview = None
    selected_model = None
    models: list[dict[str, str]] = []

    file = request.files.get("image")
    selected_model_id = request.form.get("model_id", "")

    try:
        selected_model, models = get_selected_model(selected_model_id)
    except Exception as exc:
        error = str(exc)

    if error:
        pass
    elif not file or file.filename == "":
        error = "Please select an image file."
    elif not allowed_file(file.filename):
        error = "Unsupported file type. Use png, jpg, jpeg, or webp."
    else:
        try:
            uploaded_name = secure_filename(file.filename)
            image_preview = build_image_preview(file)
            result = run_prediction(file, selected_model)
        except Exception as exc:
            error = f"Prediction failed: {exc}"

    mood = "oops"
    if result:
        mood = get_feedback_mood(result["confidence"])

    return render_template(
        "result.html",
        result=result,
        error=error,
        mood=mood,
        filename=uploaded_name,
        model_name=selected_model["name"] if selected_model else "Unknown",
        model_id=selected_model["id"] if selected_model else "",
        image_preview=image_preview,
        models=models,
    )


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5001, debug=debug_mode)

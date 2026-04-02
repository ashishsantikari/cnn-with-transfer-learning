# CNN With Transfer Learning

This repository contains a Flask web app for image classification inference using local transfer-learning models. Upload an image, pick a model, and view the predicted class with confidence and top-3 results.

The app is built for practical local experimentation: no training pipeline, just model discovery + inference + UI.

## What The App Supports

- Multiple models in one UI (selectable from a dropdown)
- Hugging Face Vision Transformer exports from local folders
- Keras `.h5` models from local files
- Automatic class labels from model metadata
- Uploaded image preview on result page
- Docker and Docker Compose for containerized runs

## Supported Model Formats

### 1. Hugging Face-style vision model directory

The app discovers folders under `MODELS_ROOT` that include:

- `config.json`
- `model.safetensors`

These models are loaded with:

- `AutoImageProcessor`
- `AutoModelForImageClassification`

If `id2label` exists in `config.json`, those labels are used automatically.

### 2. Keras `.h5` model files

The app also discovers `.h5` files recursively under `MODELS_ROOT`.

Optional label file:

- `<model_name>.labels.json`

If label JSON is missing, fallback labels are generated (`Class_0`, `Class_1`, etc).

## Inference Flow

1. App scans `MODELS_ROOT` and builds the model dropdown.
2. User uploads an image (`png`, `jpg`, `jpeg`, `webp`).
3. User-selected model is loaded lazily and cached in memory.
4. Prediction runs with backend-specific preprocessing:
   - Transformers: image processor + softmax logits
   - Keras: resize to model input size + normalize to `0..1`
5. Result page shows:
   - top label
   - confidence
   - top-3 predictions
   - uploaded file preview

## Upload And Safety Limits

- Allowed file extensions: `png`, `jpg`, `jpeg`, `webp`
- Max upload size: `8 MB`
- Filename sanitization via `secure_filename`

## Project Structure

```text
.
├── main.py
├── templates/
│   ├── index.html
│   └── result.html
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Local Run

### Prerequisites

- Python 3.11+
- pip
- Local model artifacts placed under `models/` (or custom `MODELS_ROOT`)

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Add Models

Example layout:

```text
models/
├── vit-model/
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
├── animals10_efficientnetv2s.h5
└── animals10_efficientnetv2s.labels.json
```

### Start App

```bash
python main.py
```

Open in browser:

```text
http://127.0.0.1:5000
```

## Environment Variables

- `MODELS_ROOT`: root path scanned for models (default: `models`)
- `MODEL_ID`: optional default selected model id in dropdown
- `FLASK_DEBUG`: set `1` to enable Flask debug mode

Example:

```bash
FLASK_DEBUG=1 MODELS_ROOT=models MODEL_ID=hf-vit-model python main.py
```

## Run With Docker Compose

Build and run:

```bash
docker compose up --build
```

Open:

```text
http://127.0.0.1:5000
```

Current Compose setup mounts multiple model paths into `/app/model-store` and sets:

- `MODELS_ROOT=/app/model-store`
- `MODEL_ID=hf-vit-model`

Volume mappings include:

- `./models/vit model -> /app/model-store/vit-model`
- `./models/animals10_efficientnetv2s.h5 -> /app/model-store/animals10_efficientnetv2s.h5`
- `./models/experiment-a -> /app/model-store/experiment-a`
- `./models/experiment-b -> /app/model-store/experiment-b`

To add models, mount additional files/folders under `/app/model-store`.

## Main Components

- `main.py`: routes, model discovery, model loading, prediction logic
- `templates/index.html`: model selector + upload form + loading animation
- `templates/result.html`: prediction output + confidence + top-3 + error state

## Notes And Limitations

- Models are cached in-memory after first load.
- App is for classification inference only.
- No automated test suite is included yet.
- Installing TensorFlow + PyTorch together can be heavy on small machines.

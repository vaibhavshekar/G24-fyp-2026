# SITS Forecasting UI

Streamlit-based UI for Satellite Image Time Series (SITS) forecasting with explainability (XAI). It loads trained checkpoints, runs predictions on monthly raster sequences, visualizes outputs, and generates attribution analyses (Saliency, Integrated Gradients, Occlusion). Optional: generate an LLM-written XAI report using the OpenAI API.

## Features
- Load trained model checkpoints (`.pth`)
- Browse dataset zones and preview input sequences
- Configure look-back, forecast horizon, patch size, and channels
- Run predictions and view per-month metrics
- XAI visualizations + temporal/spectral importance plots
- Optional LLM report for XAI summaries

## Project Structure
- `app.py`: Streamlit UI entry point
- `src/`: Model, dataset, XAI, and utilities
- `checkpoints/`: Trained model checkpoints (`.pth`)
- `data/`: Dataset root (zones with `normalized/` folders)
- `requirements.txt`: Python dependencies
- `run_ui.sh`: Bash helper to install deps and run the UI

## Requirements
- Python 3.9+ (recommended)
- CUDA GPU optional (CPU works, slower)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

### Option A: Streamlit directly

```bash
streamlit run app.py --server.port 8501 --server.address localhost
```

### Option B: Bash helper

```bash
bash run_ui.sh
```

Open in your browser:

```
http://localhost:8501
```

## Dataset Layout

The UI expects dataset zones that contain a `normalized/` folder with monthly `.tif` files:

```
<data_root>/
  Zone_001/
    normalized/
      2018_01.tif
      2018_02.tif
      ...
```

Set the dataset root in the sidebar (default `./data`). The UI will discover all `normalized/` folders automatically.

## Checkpoints

Place trained checkpoints in `./checkpoints` (or any folder you select in the UI). The UI will scan for `.pth` files recursively.

## Optional: LLM XAI Report

The XAI tab can generate a report using the OpenAI API.

1. Install optional deps (already in `requirements.txt`):

```bash
pip install openai python-dotenv pydantic
```

2. Provide an API key in the sidebar field **OpenAI API Key**.

The report is generated per XAI method and uses both the attribution visualization and quantitative summary.

## Troubleshooting
- If no checkpoints are found, verify the folder path and file extension (`.pth`).
- If no dataset folders are found, ensure each zone contains a `normalized/` directory with `.tif` files.
- If you see CUDA errors, the app will fall back to CPU. Use smaller patch sizes to reduce memory usage.

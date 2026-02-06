# Explainable Time Series-based Satellite Image Forecasting

Final Year Project — Group 24

**Guide**: Dr. Bagyammal T

**Contributors**
- Ananthkrishnan Chakalathalam Kannan — `CB.EN.U4CSE22308`
- Sarath Chandra — `CB.EN.U4CSE22437`
- Shaun Joseph Sunny — `CB.EN.U4CSE22443`
- Vaibhav V Shekar — `CB.EN.U4CSE22452`

**Overview**
This project forecasts multi-temporal satellite imagery and provides explainability (XAI) for the predictions. It includes a full preprocessing pipeline, a spatio‑temporal UNet‑based model (`UNet3DTemporal`), training checkpoints, and documentation detailing the approach, experiments, and results.

**Repository Layout**
- `Dataset/` — dataset folder (data + preprocessing outputs)
- `Code/` — runnable UI code and raw notebooks
- `Code/raw/preprocessing/preproccessing.ipynb` — preprocessing pipeline
- `Code/raw/pipeline/impl3_tune_4.75.ipynb` — final model architecture + training
- `Trained Models/` — trained checkpoints
- `README.md` — this file

**Dataset**
[Google Drive Folder](https://drive.google.com/drive/folders/1UQwECVDDoGTIR1EPGQD_swOtUK-lWJfi?usp=drive_link)

- Primary data lives at `Dataset/data` (junction to `D:\Amrita\Final Year Project\fyp-2026-ui\data`).
- A snapshot archive is available at `Dataset/data.zip`.

**Preprocessing**
The preprocessing notebook covers:
- Reflectance conversion and normalization
- Monthly compositing and aggregation
- Resampling (bilinear / bicubic)
- Global normalization statistics
- GeoTIFF read/write pipeline using `rasterio`

Configured paths inside `Code/raw/preprocessing/preproccessing.ipynb` now point to the dataset folder:
- `D:\Amrita\Final Year Project\fyp 2026\Dataset\flair_sen_tifs_preproccessed`
- `D:\Amrita\Final Year Project\fyp 2026\Dataset\flair2_preprocessed\<zone>\normalized\`

**Model**
Implemented model in `Code/raw/pipeline/impl3_tune_4.75.ipynb`:
- `UNet3DTemporal`
- Loss: `MAE + w*(1-SSIM)` with `SSIM_WEIGHT = 0.20`

The training data root in the notebook is configured to:
- `D:\Amrita\Final Year Project\fyp 2026\Dataset\data`

**Trained Models**
Checkpoints are under `Trained Models/checkpoints_mae_ssim` (junction to `D:\Amrita\Final Year Project\chaklu approach\checkpoints_mae_ssim`). Example runs:
- `RUN_20251227_155431/unet3d_best_mae_w_ssim.pth`
- `RUN_20260122_205632/unet3d_best_mae_w_ssim.pth`
- `RUN_20260122_211333/unet3d_best_mae_w_ssim.pth`

**Documentation**
- `Code/G24_P2_R2.pptx`
- `Code/G24_Documentation_Phase2 (2).pdf`

**Quick Start**
1. Open `Code/raw/preprocessing/preproccessing.ipynb` to generate normalized composites.
2. Open `Code/raw/pipeline/impl3_tune_4.75.ipynb` to train or evaluate the model.
3. Use checkpoints in `Trained Models/` for inference or evaluation.

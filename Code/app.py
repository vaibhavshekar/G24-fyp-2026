"""
SITS Forecasting UI - Satellite Image Time Series Forecasting with XAI

A Streamlit application for:
- Loading trained model checkpoints
- Selecting dataset sequences for prediction
- Configuring prediction parameters (lookback, months ahead)
- Running predictions and viewing visualizations
- Generating XAI (Explainable AI) outputs
"""

import os
import base64
import io
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.model import UNet3DTemporal, load_model, get_model_info
from src.dataset import SingleZoneFolderDataset
from src.loss import compute_metrics
from src.xai import XAIConfig
from src.xai.methods import run_xai_method
from src.xai.visualization import (
    visualize_xai_rgb,
    xai_summary_dict,
    plot_month_importance,
    plot_band_importance,
)

# Optional OpenAI integration (LLM report)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
try:
    from pydantic import BaseModel, Field
except Exception:
    BaseModel = None
    Field = None
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


# Page configuration
st.set_page_config(
    page_title="SITS Forecasting",
    page_icon="satellite",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_device() -> torch.device:
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_checkpoints(base_dir: str) -> List[str]:
    """Find all checkpoint files in directory."""
    checkpoints = []
    base_path = Path(base_dir)
    if base_path.exists():
        for ckpt in base_path.rglob("*.pth"):
            checkpoints.append(str(ckpt))
    return sorted(checkpoints)


def find_dataset_folders(base_dir: str) -> List[str]:
    """Find all normalized folders in dataset directory."""
    folders = []
    base_path = Path(base_dir)
    if base_path.exists():
        for norm_folder in base_path.rglob("normalized"):
            if norm_folder.is_dir():
                # Check if it contains .tif files
                if list(norm_folder.glob("*.tif")):
                    folders.append(str(norm_folder))
    return sorted(folders)


def tensor_to_rgb(tensor: torch.Tensor, cfg: XAIConfig) -> np.ndarray:
    """Convert tensor to RGB image (no stretch, notebook-consistent)."""
    arr = tensor.detach().cpu().float().numpy()
    C, H, W = arr.shape
    b0, b1, b2 = cfg.rgb_bands
    b0 = int(np.clip(b0, 0, C - 1))
    b1 = int(np.clip(b1, 0, C - 1))
    b2 = int(np.clip(b2, 0, C - 1))
    rgb = np.stack([arr[b0], arr[b1], arr[b2]], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def create_sequence_preview(
    dataset: SingleZoneFolderDataset,
    cfg: XAIConfig,
) -> plt.Figure:
    """Create preview of input sequence."""
    x, y, _, zone = dataset[0]

    n_input = x.shape[0]
    n_output = y.shape[0] if y.sum() > 0 else 0
    total = n_input + n_output

    fig, axes = plt.subplots(1, total, figsize=(4 * total, 4))
    if total == 1:
        axes = [axes]

    dates = dataset.get_date_info()

    for i in range(n_input):
        rgb = tensor_to_rgb(x[i], cfg)
        axes[i].imshow(rgb)
        axes[i].set_title(f"Input {i+1}\n{dates[i] if i < len(dates) else ''}")
        axes[i].axis("off")

    for i in range(n_output):
        rgb = tensor_to_rgb(y[i], cfg)
        axes[n_input + i].imshow(rgb)
        axes[n_input + i].set_title(f"Target {i+1}\n{dates[n_input + i] if n_input + i < len(dates) else ''}")
        axes[n_input + i].axis("off")

    plt.tight_layout()
    return fig


def run_prediction(
    model: UNet3DTemporal,
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run prediction on input sequence."""
    model.eval()
    with torch.no_grad():
        x_device = x.unsqueeze(0).to(device)
        pred = model(x_device)
        pred = torch.clamp(pred, 0.0, 1.0)
    return pred.cpu()


def build_default_xai_config(t_out: int) -> XAIConfig:
    cfg = XAIConfig()
    cfg.target_type = "error_abs"
    cfg.target_mode = "full_mean"
    cfg.patch_radius = 6
    cfg.t_out = int(t_out)
    cfg.rgb_bands = (2, 1, 0)
    return cfg


def _encode_image_png(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


class XAIReport(BaseModel):
    tasks_A_temporal_reliance: str = Field(..., description="Answer Task A")
    tasks_B_spectral_reliance: str = Field(..., description="Answer Task B")
    tasks_C_spatial_pattern: str = Field(..., description="Answer Task C")
    tasks_D_trustworthiness: str = Field(..., description="Answer Task D")
    thesis_ready_paragraph: str = Field(..., description="150-250 word paragraph")
    concise_summary: str = Field(..., description="1-3 bullet summary or short paragraph")


def make_llm_prompt(summary_json_text: str) -> str:
    return f"""
You are an explainability analyst reviewing a multiband satellite forecasting model.

You are given:
1) An XAI visualization PNG with the following panels:
   - Last Input (RGB)
   - Predicted Next (RGB)
   - Ground Truth (RGB)
   - Difference (scalar error heatmap)
   - XAI Overlay (attribution over image)
   - Important Region Only (thresholded attribution mask)
2) A JSON summary containing quantitative explainability statistics.

Your goal is to write a clear, professional, evidence-based interpretation suitable for a technical project report or internal model review.

ANALYSIS RULES
- Do not speculate about physical meaning, land-cover types, or band semantics.
- Refer to time steps and bands strictly by index (months M1..Mk, bands B0..B(n-1)).
- Support every claim about importance or dominance using numeric values from the JSON.
- Base spatial conclusions only on visible agreement between:
  Difference vs XAI Overlay vs Important Region mask.
- If evidence is weak or mixed, state that explicitly.
- Maintain a neutral, analyst tone (precise, factual, non-promotional).

INTERPRETATION GUIDELINES
- Temporal dominance:
  - Describe as "strong" only if one month contributes >= ~40% or clearly exceeds others.
- Spectral dominance:
  - Comment on whether importance is concentrated (few bands dominate) or distributed.
- Spatial attribution:
  - Use importance_entropy to characterize attribution as concentrated vs diffuse.
  - Cross-check entropy with visual patterns in the overlay and mask.
- Trustworthiness:
  - Assess whether attribution highlights areas that also show higher prediction error.

TASKS
A) Temporal reliance
Describe how the model distributes importance across input months.
Report the most influential month and compare it to the remaining months.
Briefly state whether the model appears to rely mainly on recent inputs or integrates information over multiple months.

B) Spectral reliance
Summarize how importance is distributed across spectral bands.
Identify the most influential bands and comment on whether the distribution is sharply peaked or relatively balanced.
If RGB bands are provided, note any overlap by index only.

C) Spatial attribution pattern
Describe the spatial structure of attribution using both importance_entropy and the visual overlay.
Comment on whether attribution forms coherent regions, scattered hotspots, or diffuse patterns.
Use hotspot coordinates to support observations about clustering or dispersion.

D) Trustworthiness check
Compare regions of high prediction error with regions of high attribution.
State whether attribution aligns well, partially, or poorly with error patterns.
Provide a brief overall trust assessment (high / moderate / low) with justification.

FINAL SUMMARY
Write a short concluding paragraph that:
- Mentions the MAE,
- Summarizes temporal and spectral reliance,
- Describes spatial concentration,
- States whether the explanation appears consistent with observed errors.

INPUT
JSON summary:
{summary_json_text}
""".strip()


def generate_llm_report(
    api_key: str,
    summary: Dict[str, Any],
    fig: plt.Figure,
    model_name: str = "gpt-4o-2024-08-06",
) -> str:
    if OpenAI is None:
        return "OpenAI SDK not installed. Add `openai` to requirements to enable LLM reports."
    if BaseModel is None or Field is None:
        return "Pydantic not installed. Add `pydantic` to requirements to enable LLM reports."
    if not api_key:
        return "OpenAI API key not provided."

    if fig is None:
        return "LLM report skipped: XAI figure was not generated."

    try:
        if load_dotenv is not None:
            load_dotenv()
        client = OpenAI(api_key=api_key)
        summary_text = json.dumps(summary, indent=2)
        prompt_text = make_llm_prompt(summary_text)

        image_url = _encode_image_png(fig)

        content = [
            {"type": "input_text", "text": prompt_text},
            {"type": "input_image", "image_url": image_url},
        ]
        resp = client.responses.parse(
            model=model_name,
            input=[{"role": "user", "content": content}],
            text_format=XAIReport,
        )
        report = resp.output_parsed
        md = f"""# XAI LLM Report

## Task A — Temporal reliance
{report.tasks_A_temporal_reliance}

## Task B — Spectral reliance
{report.tasks_B_spectral_reliance}

## Task C — Spatial attribution pattern + implication
{report.tasks_C_spatial_pattern}

## Task D — Trustworthiness check (XAI vs error)
{report.tasks_D_trustworthiness}

## Thesis-ready paragraph
{report.thesis_ready_paragraph}

## Concise summary
{report.concise_summary}
"""
        return md
    except Exception as e:
        return f"LLM report failed: {e}"


def main():
    st.title("SITS Forecasting")
    st.markdown("### Satellite Image Time Series Forecasting with Explainable AI")

    # Initialize session state
    if "model" not in st.session_state:
        st.session_state.model = None
    if "model_config" not in st.session_state:
        st.session_state.model_config = None
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "xai_results" not in st.session_state:
        st.session_state.xai_results = None
    if "llm_reports" not in st.session_state:
        st.session_state.llm_reports = {}
    if "display_t_out" not in st.session_state:
        st.session_state.display_t_out = 0
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""

    device = get_device()

    def ensure_prediction() -> Optional[dict]:
        if st.session_state.model is None or st.session_state.dataset is None:
            return None
        if st.session_state.prediction is None:
            dataset = st.session_state.dataset
            model = st.session_state.model
            x, y, _, zone = dataset[0]
            pred = run_prediction(model, x, device)
            st.session_state.prediction = {
                "x": x,
                "y": y,
                "pred": pred,
                "zone": zone,
            }
        return st.session_state.prediction

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        # Model Loading Section
        st.subheader("1. Load Model Checkpoint")

        checkpoint_dir = st.text_input(
            "Checkpoint Directory",
            value="./checkpoints",
            help="Directory containing trained model checkpoints",
        )

        # Find checkpoints
        checkpoints = find_checkpoints(checkpoint_dir)

        if checkpoints:
            selected_ckpt = st.selectbox(
                "Select Checkpoint",
                options=checkpoints,
                format_func=lambda x: Path(x).name,
            )

            if st.button("Load Model", type="primary"):
                with st.spinner("Loading model..."):
                    try:
                        # Get model info
                        info = get_model_info(selected_ckpt, str(device))
                        config_dict = info.get("config", {})

                        # Create config
                        in_channels = config_dict.get("in_channels", config_dict.get("IN_CHANNELS", 10))
                        base_channels = config_dict.get("base_channels", config_dict.get("BASE_CHANNELS", 12))
                        pred_len = config_dict.get("pred_len", config_dict.get("PRED_LEN", 3))

                        # Load model
                        model = load_model(
                            selected_ckpt,
                            in_channels=in_channels,
                            base_channels=base_channels,
                            pred_len=pred_len,
                            device=str(device),
                        )

                        # Prepare safe display strings
                        epoch_val = info.get("epoch", "N/A")
                        val_loss_raw = info.get("val_loss", "N/A")
                        if isinstance(val_loss_raw, (int, float)):
                            val_loss_str = f"{val_loss_raw:.6f}"
                        else:
                            val_loss_str = str(val_loss_raw)

                        st.session_state.model = model
                        st.session_state.model_config = config_dict
                        st.success(f"Model loaded! Epoch: {epoch_val}, Val Loss: {val_loss_str}")
                    except Exception as e:
                        st.error(f"Failed to load model: {e}")
        else:
            st.info("No checkpoints found. Upload or specify a different directory.")

            uploaded_file = st.file_uploader("Upload Checkpoint", type=["pth"])
            if uploaded_file:
                # Save to temp location
                temp_path = Path("./temp_checkpoint.pth")
                temp_path.write_bytes(uploaded_file.read())

                if st.button("Load Uploaded Model"):
                    with st.spinner("Loading model..."):
                        try:
                            info = get_model_info(str(temp_path), str(device))
                            config_dict = info.get("config", {})

                            in_channels = config_dict.get("in_channels", config_dict.get("IN_CHANNELS", 10))
                            base_channels = config_dict.get("base_channels", config_dict.get("BASE_CHANNELS", 12))
                            pred_len = config_dict.get("pred_len", config_dict.get("PRED_LEN", 3))

                            model = load_model(
                                str(temp_path),
                                in_channels=in_channels,
                                base_channels=base_channels,
                                pred_len=pred_len,
                                device=str(device),
                            )

                            st.session_state.model = model
                            st.session_state.model_config = config_dict
                            st.success("Model loaded!")
                        except Exception as e:
                            st.error(f"Failed to load model: {e}")

        st.divider()

        # Dataset Selection Section
        st.subheader("2. Select Dataset")

        dataset_dir = st.text_input(
            "Dataset Directory",
            value="./data",
            help="Root directory containing dataset zones",
        )

        # Find dataset folders
        dataset_folders = find_dataset_folders(dataset_dir)

        if dataset_folders:
            selected_folder = st.selectbox(
                "Select Zone",
                options=dataset_folders,
                format_func=lambda x: "/".join(Path(x).parts[-2:]),
            )
        else:
            selected_folder = st.text_input(
                "Or enter path to normalized folder",
                help="Path to folder containing monthly .tif files",
            )

        st.divider()

        # Parameters Section
        st.subheader("3. Parameters")

        config = st.session_state.model_config or {}

        input_len = st.number_input(
            "Look-back (input months)",
            min_value=1,
            max_value=24,
            value=config.get("input_len", config.get("INPUT_LEN", 6)),
            help="Number of past months to use as input",
        )

        pred_len = st.number_input(
            "Forecast horizon (output months)",
            min_value=1,
            max_value=12,
            value=config.get("pred_len", config.get("PRED_LEN", 3)),
            help="Number of months ahead to predict",
        )

        patch_size = st.number_input(
            "Patch Size",
            min_value=32,
            max_value=512,
            value=config.get("patch_size", config.get("PATCH_SIZE", 128)),
            help="Size of image patches",
        )

        in_channels = st.number_input(
            "Input Channels",
            min_value=1,
            max_value=20,
            value=config.get("in_channels", config.get("IN_CHANNELS", 10)),
            help="Number of spectral bands",
        )

        st.divider()

        st.subheader("4. Display / Target Month")
        st.session_state.display_t_out = st.number_input(
            "Target Month Index (0-based)",
            min_value=0,
            max_value=max(0, int(pred_len) - 1),
            value=min(st.session_state.display_t_out, max(0, int(pred_len) - 1)),
            help="Select which predicted month to display and explain",
        )

        st.divider()
        st.subheader("5. OpenAI API Key")
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key,
            help="Required to generate LLM report in XAI tab",
        )

        st.divider()

        # Load Dataset Button
        if st.button("Load Dataset", type="secondary"):
            if selected_folder and Path(selected_folder).exists():
                with st.spinner("Loading dataset..."):
                    try:
                        clamp_lo = config.get("clamp_lo", config.get("CLAMP_LO", -2.0))
                        clamp_hi = config.get("clamp_hi", config.get("CLAMP_HI", 6.0))

                        dataset = SingleZoneFolderDataset(
                            folder=selected_folder,
                            input_len=input_len,
                            pred_len=pred_len,
                            in_channels=in_channels,
                            patch_size=patch_size,
                            clamp_lo=clamp_lo,
                            clamp_hi=clamp_hi,
                            deterministic_patch=True,
                        )

                        st.session_state.dataset = dataset
                        st.session_state.prediction = None
                        st.session_state.xai_results = None
                        st.session_state.llm_reports = {}
                        st.success(f"Dataset loaded: {dataset.zone}")
                    except Exception as e:
                        st.error(f"Failed to load dataset: {e}")
            else:
                st.warning("Please select a valid dataset folder.")

        st.divider()

        if st.button("Run Predictions", type="primary"):
            if st.session_state.model is None or st.session_state.dataset is None:
                st.warning("Load both a model and dataset before running.")
            else:
                with st.spinner("Running prediction and XAI..."):
                    progress = st.progress(0)
                    progress_label = st.empty()
                    pred_data = ensure_prediction()
                    if pred_data is None:
                        st.warning("Prediction failed to run.")
                    else:
                        progress_label.write("Prediction complete. Running XAI...")
                        progress.progress(10)
                        xai_cfg = build_default_xai_config(st.session_state.display_t_out)
                        x = pred_data["x"].unsqueeze(0).to(device)
                        y = pred_data["y"].unsqueeze(0).to(device)
                        model = st.session_state.model

                        xai_results = {}
                        llm_reports = {}
                        methods = ["saliency", "ig", "occlusion"]
                        step = 90 // max(1, len(methods))
                        current = 10
                        for method in methods:
                            progress_label.write(f"Running XAI: {method}...")
                            pred, pix_attr, per_month, per_band = run_xai_method(
                                model, x, y, method, xai_cfg
                            )
                            xai_results[method] = {
                                "method": method,
                                "pred": pred.cpu(),
                                "pix_attr": pix_attr.cpu() if torch.is_tensor(pix_attr) else pix_attr,
                                "per_month": per_month.cpu() if torch.is_tensor(per_month) else per_month,
                                "per_band": per_band.cpu() if torch.is_tensor(per_band) else per_band,
                                "config": xai_cfg,
                                "x": x.cpu(),
                                "y": y.cpu(),
                            }

                            mae, fig = visualize_xai_rgb(
                                xai_results[method]["x"],
                                xai_results[method]["pred"],
                                xai_results[method]["y"],
                                xai_results[method]["pix_attr"],
                                xai_results[method]["config"],
                                show_inline=False,
                                return_fig=True,
                            )
                            summary = xai_summary_dict(
                                zone=pred_data["zone"],
                                method=method,
                                cfg=xai_results[method]["config"],
                                mae=mae,
                                per_m=xai_results[method]["per_month"],
                                per_b=xai_results[method]["per_band"],
                                pix_attr=xai_results[method]["pix_attr"],
                                png_path="<not saved>",
                            )
                            llm_reports[method] = generate_llm_report(
                                st.session_state.openai_api_key,
                                summary,
                                fig,
                            )
                            if fig is not None:
                                plt.close(fig)
                            current = min(95, current + step)
                            progress.progress(current)

                        st.session_state.xai_results = xai_results
                        st.session_state.llm_reports = llm_reports
                        progress.progress(100)
                        progress_label.write("Done.")
                        st.success("Prediction and XAI complete!")
    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Tabs for different views
        tabs = st.tabs(["Input Preview", "Prediction", "XAI Analysis"])

        # XAI Config (shared)
        xai_cfg = XAIConfig()

        with tabs[0]:
            st.header("Input Sequence Preview")

            if st.session_state.dataset is not None:
                dataset = st.session_state.dataset

                # Show sequence info
                st.info(f"Zone: {dataset.zone} | Input: {dataset.input_len} months | Predict: {dataset.pred_len} months")

                # Date info
                dates = dataset.get_date_info()
                st.write("**Sequence dates:**", " -> ".join(dates[:dataset.input_len + dataset.pred_len]))

                # Preview plot
                fig = create_sequence_preview(dataset, xai_cfg)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Load a dataset to see the input sequence preview.")

        with tabs[1]:
            st.header("Prediction Results")

            if st.session_state.model is not None and st.session_state.dataset is not None:
                if st.session_state.prediction is not None:
                    pred_data = st.session_state.prediction
                    x = pred_data["x"]
                    y = pred_data["y"]
                    pred = pred_data["pred"]
                    t_out = int(min(st.session_state.display_t_out, pred.shape[1] - 1))

                    # Compute metrics if ground truth available
                    if y.sum() > 0:
                        pred_t = pred[:, t_out : t_out + 1]
                        y_t = y.unsqueeze(0)[:, t_out : t_out + 1]
                        metrics = compute_metrics(pred_t, y_t)

                        # Display metrics
                        metric_cols = st.columns(5)
                        metric_cols[0].metric("MAE", f"{metrics['MAE']:.4f}")
                        metric_cols[1].metric("MSE", f"{metrics['MSE']:.4f}")
                        metric_cols[2].metric("RMSE", f"{metrics['RMSE']:.4f}")
                        metric_cols[3].metric("PSNR", f"{metrics['PSNR']:.2f} dB")
                        metric_cols[4].metric("SSIM", f"{metrics['SSIM']:.4f}")

                    # Visualization (last input + selected prediction)
                    st.subheader("Last Input (RGB)")
                    last_in_rgb = tensor_to_rgb(x[-1], xai_cfg)
                    st.image(last_in_rgb, caption="Last Input", use_container_width=True)

                    st.subheader("Predicted Frame (Selected Month)")
                    pred_rgb = tensor_to_rgb(pred[0, t_out], xai_cfg)
                    st.image(pred_rgb, caption=f"Predicted Month {t_out + 1}", use_container_width=True)
            else:
                st.info("Load both a model and dataset to run predictions.")

        with tabs[2]:
            st.header("XAI Analysis")

            if st.session_state.xai_results is not None:
                method_tabs = st.tabs(["Saliency", "Integrated Gradients", "Occlusion"])
                method_map = {
                    "Saliency": "saliency",
                    "Integrated Gradients": "ig",
                    "Occlusion": "occlusion",
                }

                for tab, label in zip(method_tabs, method_map.keys()):
                    with tab:
                        method = method_map[label]
                        if method not in st.session_state.xai_results:
                            st.info("Run predictions to generate XAI outputs.")
                            continue

                        xai_results = st.session_state.xai_results[method]

                        st.subheader("Attribution Visualization")
                        mae, fig = visualize_xai_rgb(
                            xai_results["x"],
                            xai_results["pred"],
                            xai_results["y"],
                            xai_results["pix_attr"],
                            xai_results["config"],
                            show_inline=False,
                            return_fig=True,
                        )
                        if fig is not None:
                            st.pyplot(fig)
                            plt.close(fig)

                        st.subheader("Attribution Analysis")
                        imp_col1, imp_col2 = st.columns(2)
                        with imp_col1:
                            st.write("**Temporal Importance**")
                            per_month = xai_results["per_month"]
                            if torch.is_tensor(per_month):
                                per_month = per_month.numpy()
                            fig_month = plot_month_importance(per_month)
                            st.pyplot(fig_month)
                            plt.close(fig_month)

                        with imp_col2:
                            st.write("**Spectral Importance**")
                            per_band = xai_results["per_band"]
                            if torch.is_tensor(per_band):
                                per_band = per_band.numpy()
                            fig_band = plot_band_importance(per_band, top_k=10)
                            st.pyplot(fig_band)
                            plt.close(fig_band)

                        st.subheader("LLM Report")
                        report_text = st.session_state.llm_reports.get(method, "")
                        if report_text:
                            st.write(report_text)
                        else:
                            st.info("No LLM report available. Enter API key and run predictions.")
            else:
                st.info("Run predictions to generate XAI analysis.")

    with col2:
        st.header("Status")

        # Model status
        st.subheader("Model")
        if st.session_state.model is not None:
            st.success("Loaded")
            if st.session_state.model_config:
                with st.expander("Model Config"):
                    st.json(st.session_state.model_config)
        else:
            st.warning("Not loaded")

        # Dataset status
        st.subheader("Dataset")
        if st.session_state.dataset is not None:
            st.success(f"Zone: {st.session_state.dataset.zone}")
            st.write(f"- Input months: {st.session_state.dataset.input_len}")
            st.write(f"- Predict months: {st.session_state.dataset.pred_len}")
            st.write(f"- Patch size: {st.session_state.dataset.P}")
            st.write(f"- Has future: {st.session_state.dataset.use_future}")
        else:
            st.warning("Not loaded")

        # Device info
        st.subheader("Device")
        st.info(f"Using: {device}")

        # Quick actions
        st.subheader("Quick Actions")

        if st.button("Clear All", type="secondary"):
            st.session_state.model = None
            st.session_state.model_config = None
            st.session_state.dataset = None
            st.session_state.prediction = None
            st.session_state.xai_results = None
            st.session_state.llm_reports = {}
            st.rerun()


if __name__ == "__main__":
    main()

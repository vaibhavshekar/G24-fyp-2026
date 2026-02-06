"""
Example usage of SITS Forecasting library.

This script demonstrates how to:
1. Load a trained model
2. Run predictions on a dataset
3. Perform XAI analysis
"""

from pathlib import Path
from src import (
    predict_folder,
    run_xai_analysis,
    load_model,
    get_model_info,
    XAIConfig,
)


def main():
    # Paths - update these to your actual paths
    checkpoint_path = "./checkpoints/RUN_20251227_155431/unet3d_best_mae_w_ssim.pth"
    data_folder = "./data/D004_2021/normalized"  # Path to folder with .tif files
    output_dir = "./outputs"

    # Check if paths exist
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please update the checkpoint_path variable.")
        return

    if not Path(data_folder).exists():
        print(f"Data folder not found: {data_folder}")
        print("Please update the data_folder variable.")
        return

    # Get model info
    print("=" * 50)
    print("Model Information")
    print("=" * 50)
    info = get_model_info(checkpoint_path)
    print(f"Epoch: {info.get('epoch', 'N/A')}")
    print(f"Val Loss: {info.get('val_loss', 'N/A')}")
    print(f"Loss Name: {info.get('loss_name', 'N/A')}")

    # Run prediction
    print("\n" + "=" * 50)
    print("Running Prediction")
    print("=" * 50)
    result = predict_folder(
        checkpoint_path=checkpoint_path,
        data_folder=data_folder,
        output_dir=output_dir,
    )

    print(f"Zone: {result['zone']}")
    print(f"Input shape: {result['input'].shape}")
    print(f"Prediction shape: {result['prediction'].shape}")
    print(f"Dates: {result['dates']}")

    if result["metrics"]:
        print("\nMetrics:")
        for key, value in result["metrics"].items():
            print(f"  {key}: {value:.4f}")

    # Run XAI analysis
    print("\n" + "=" * 50)
    print("Running XAI Analysis")
    print("=" * 50)

    for method in ["saliency", "ig", "occlusion"]:
        print(f"\nMethod: {method}")
        xai_result = run_xai_analysis(
            checkpoint_path=checkpoint_path,
            data_folder=data_folder,
            method=method,
            output_dir=output_dir,
        )

        if "output_files" in xai_result:
            print(f"  Visualization: {xai_result['output_files']['visualization']}")
            print(f"  Summary: {xai_result['output_files']['summary']}")

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()

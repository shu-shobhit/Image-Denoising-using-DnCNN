inference_config = {
    "model_path": "./Results/final_trained_model.pth",  # Path to the trained model
    "input_dir": "./dataset/val_noisy35",  # Directory containing input noisy images for inference
    "output_dir": "./Results",  # Directory to save denoised outputs after inference
    "save_comparison": True,  # Save side-by-side comparisons
    "depth": 17,  # Should match the trained model
    "channels": 128,  # Should match the trained model
    "original_dir": "./dataset/val_original",  # Optional path to ground truth images for evaluation
}

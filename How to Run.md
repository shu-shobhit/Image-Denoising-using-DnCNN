# Image Denoising using DnCNN

## Project Structure

```
├── configurations/          # Configuration files
│   ├── train_config.py      # Training configuration parameters
│   └── inference_config.py  # Inference configuration parameters
├── dataset/
|   |__ dataset.py
|   |__ inference_dataset.py
├── inference/               # Inference code
│   └── inference.py         # Main inference implementation
├── model/                   # Model architecture
│   └── model.py             # DnCNN model implementation
├── Results/                 # Saved results directory
├── training/                # Training code
│   └── trainer.py           # Training implementation
├── Comparison_Visualization.png  # Visual comparison of results
├── HyperParameter_Tuning.ipynb   # Jupyter notebook for hyperparameter experiments
├── Visualization.ipynb      # Jupyter notebook for visualizing results
├── main.py                  # Main entry point
├── utils.py                 # Utility functions
└── val_denoised_Images.png  # Example of denoised validation images
```

## Prerequisites

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- PIL/Pillow
- tqdm
- wandb

## Configuration

The project uses two main configuration files that you should modify before running the code:

### Training Configuration

The `configurations/train_config.py` contains the following parameters:

```python
train_config = {
    "validate": True, # whether to create a val_set from the training set and perform validation on it
    "learning_rate": 1e-4,  # Learning rate for the AdamW optimizer
    "batch_size": 64,  # Batch size for training and validation
    "epochs": 50,  # Number of training epochs
    "depth": 17,  # Depth of the DnCNN model
    "channels": 128, # Channel dimensions of layers in the DnCNN model
    "patch_size": 50,  # Size of the patches extracted from images
    "stride": 20,  # Stride for the sliding window patch extraction
    "val_size": 0.2,  # Fraction of data used for validation
    "checkpoint_dir": "./checkpoints",  # Directory to save model checkpoints
    "log_interval": 1,  # Log results every 1 epoch
    "wandb_enabled": True,  # Enable Weights & Biases logging
    "wandb_mode": "online",  # "online", "offline", or "disabled"
    "project_name": "ImageDenoising_1",  # Project name for Weights & Biases
    "noisy_dir": "/kaggle/input/cbsd68/noisy35",  # Directory containing noisy images
    "original_dir": "/kaggle/input/cbsd68/original_png",  # Directory containing original images
}
```

**Note:** The dataset paths are configured for Kaggle environments by default. If you're running locally, update the `noisy_dir` and `original_dir` paths accordingly.

### Inference Configuration

The `configurations/inference_config.py` contains:

```python
inference_config = {
    "model_path": "./checkpoints/best_model.pth",  # Path to the trained model
    "input_dir": "./test_data/noisy",  # Directory containing input noisy images for inference
    "output_dir": "./test_data/denoised",  # Directory to save denoised outputs after inference
    "save_comparison": True,  # Save side-by-side comparisons
    "depth": 17,  # Should match the trained model
    "channels": 128,  # Should match the trained model
    "original_dir": None,  # Optional path to ground truth images for evaluation
}
```

**Note:** You can optionally provide ground truth directory in case of evaluation. However if no "original_dir" is provided then the function just outputs the denoised image and save those in "output_dir" (also comaprison plots too if save_comparison = True).

## Usage

1. **Prepare your dataset**:

   - Place your noisy training images in a directory
   - Place your original (clean) images in a directory
   - Update the paths in `train_config`
2. **Training mode**:

   ```bash
   python main.py --mode train
   ```
3. **Inference mode**:

   - Place test noisy images in a dir and update the path in inference_config
   - Run:

   ```bash
   python main.py --mode inference
   ```

## Weights & Biases Integration

This project supports experiment tracking with Weights & Biases:

1. Create an account at [wandb.ai](https://wandb.ai) if you don't have one
2. Login via command line: `wandb login`
3. Set `wandb_enabled` to `True` in the training configuration
4. Choose the mode: "online" (sync in real-time), "offline" (save locally), or "disabled"
5. Set your project name in `project_name`

## Results and Visualization

The repository includes visualization tools to assess model performance:

- `Visualization.ipynb`: Explore denoising results on various images
- `HyperParameter_Tuning.ipynb`: Analysis of different hyperparameter settings
- `Comparison_Visualization.png`: Visualize the ground truth, denoised output and noisy image with convienience
- `val_denoised_Images.png`: Examples of denoised validation images

## Performance Metrics

The model is evaluated using:

- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index Measure (SSIM)
- Visual quality assessment

train_config = {
    "validate": True,
    "learning_rate": 1e-4,  # Learning rate for the AdamW optimizer
    "batch_size": 64,  # Batch size for training and validation
    "epochs": 50,  # Number of training epochs
    "depth": 17,  # Depth of the DnCNN model
    "channels": 128,
    "patch_size": 50,  # Size of the patches extracted from images
    "stride": 20,  # Stride for the sliding window patch extraction
    "val_size": 0.2,  # Fraction of data used for validation
    "checkpoint_dir": "./checkpoints",  # Directory to save model checkpoints
    "log_interval": 1,  # Log results every 1 epoch
    "wandb_enabled": True,  # Enable Weights & Biases logging
    "wandb_mode": "online",  # "online", "offline", or "disabled"
    "project_name": "ImageDenoising_using_dncnn",  # Project name for Weights & Biases
    "noisy_dir": "./dataset/train_noisy35",  # Directory containing noisy images
    "original_dir": "./dataset/train_original",  # Directory containing original images
}

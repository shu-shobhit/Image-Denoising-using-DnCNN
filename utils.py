import os
import torch
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt


def save_image(tensor, path):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = torch.clamp(tensor, 0, 1)
    img = F.to_pil_image(tensor)
    img.save(path)
    return True


def create_comparison_image(
    noisy_tensor, denoised_tensor, original_tensor=None, save_path=None
):
    """Create a comparison image with noisy, denoised, and optionally original images."""
    # Convert tensors to numpy arrays for plotting
    if noisy_tensor.is_cuda:
        noisy_tensor = noisy_tensor.cpu()
    if denoised_tensor.is_cuda:
        denoised_tensor = denoised_tensor.cpu()

    noisy_img = F.to_pil_image(torch.clamp(noisy_tensor, 0, 1))
    denoised_img = F.to_pil_image(torch.clamp(denoised_tensor, 0, 1))

    if original_tensor is not None:
        if original_tensor.is_cuda:
            original_tensor = original_tensor.cpu()
        original_img = F.to_pil_image(torch.clamp(original_tensor, 0, 1))

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(np.array(noisy_img))
        axes[0].set_title("Noisy")
        axes[0].axis("off")

        axes[1].imshow(np.array(denoised_img))
        axes[1].set_title("Denoised")
        axes[1].axis("off")

        axes[2].imshow(np.array(original_img))
        axes[2].set_title("Original")
        axes[2].axis("off")
    else:
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Display images
        axes[0].imshow(np.array(noisy_img))
        axes[0].set_title("Noisy")
        axes[0].axis("off")

        axes[1].imshow(np.array(denoised_img))
        axes[1].set_title("Denoised")
        axes[1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return True
    else:
        return fig

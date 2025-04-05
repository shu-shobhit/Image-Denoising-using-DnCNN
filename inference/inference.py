import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from dataset.inference_dataset import InferenceDataset
from model.model import DnCNN
from utils import save_image, create_comparison_image


class Inference:
    def __init__(self, config):
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")

        self._init_model()

        # Create output directory if it doesn't exist
        os.makedirs(self.config["output_dir"], exist_ok=True)

        # Initialize metrics if ground truth is available
        if self.config["original_dir"] is not None:
            self.psnr = PeakSignalNoiseRatio().to(self.device)
            self.ssim = StructuralSimilarityIndexMeasure().to(self.device)
            self.evaluation_mode = True
            print("Evaluation mode: ON (Ground truth available)")
        else:
            self.evaluation_mode = False
            print("Evaluation mode: OFF (No ground truth)")

    def _init_model(self):
        self.model = DnCNN(
            depth=self.config["depth"], channels=self.config["channels"]
        ).to(self.device)

        # Load model weights
        if os.path.isfile(self.config["model_path"]):
            checkpoint = torch.load(self.config["model_path"], map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"Model loaded from {self.config['model_path']}")
        else:
            raise FileNotFoundError(f"Model not found at {self.config['model_path']}")

        self.model.eval()

    def run(self):
        """Run inference on all images in the val input directory."""
        dataset = InferenceDataset(
            input_dir=self.config["input_dir"], original_dir=self.config["original_dir"]
        )

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        if self.evaluation_mode:
            self.psnr.reset()
            self.ssim.reset()
            total_psnr = 0
            total_ssim = 0

        with torch.no_grad():
            for batch in dataloader:
                noisy_img = batch["noisy_image"].to(self.device)
                filename = batch["filename"][0]

                if self.evaluation_mode:
                    original_img = batch["original_image"].to(self.device)

                predicted_noise = self.model(noisy_img)

                denoised_img = noisy_img - predicted_noise

                denoised_img = torch.clamp(denoised_img, 0, 1)

                save_path = os.path.join(self.config["output_dir"], filename)
                save_image(denoised_img[0], save_path)

                if self.config["save_comparison"]:
                    if self.evaluation_mode:
                        comparison_path = os.path.join(
                            self.config["output_dir"], f"comparison_{filename}"
                        )
                        create_comparison_image(
                            noisy_img[0],
                            denoised_img[0],
                            original_img[0],
                            comparison_path,
                        )
                    else:
                        # Create comparison without ground truth(if not provided)
                        comparison_path = os.path.join(
                            self.config["output_dir"], f"comparison_{filename}"
                        )
                        create_comparison_image(
                            noisy_img[0], denoised_img[0], None, comparison_path
                        )

                if self.evaluation_mode:
                    self.psnr.update(denoised_img, original_img)
                    self.ssim.update(denoised_img, original_img)

                    # Calculate per-image metrics
                    img_psnr = PeakSignalNoiseRatio().to(self.device)
                    img_ssim = StructuralSimilarityIndexMeasure().to(self.device)

                    img_psnr.update(denoised_img, original_img)
                    img_ssim.update(denoised_img, original_img)

                    total_psnr += img_psnr.compute().item()
                    total_ssim += img_ssim.compute().item()

                    print(
                        f"{filename} - PSNR: {img_psnr.compute().item():.4f}, SSIM: {img_ssim.compute().item():.4f}"
                    )

        if self.evaluation_mode:
            avg_psnr = self.psnr.compute().item()
            avg_ssim = self.ssim.compute().item()

            per_img_psnr = total_psnr / len(dataloader)
            per_img_ssim = total_ssim / len(dataloader)

            print(f"\nAverage PSNR: {avg_psnr:.4f}")
            print(f"Average SSIM: {avg_ssim:.4f}")
            print(f"Per-image average PSNR: {per_img_psnr:.4f}")
            print(f"Per-image average SSIM: {per_img_ssim:.4f}")

            with open(os.path.join(self.config["output_dir"], "metrics.txt"), "w") as f:
                f.write(f"Average PSNR: {avg_psnr:.4f}\n")
                f.write(f"Average SSIM: {avg_ssim:.4f}\n")
                f.write(f"Per-image average PSNR: {per_img_psnr:.4f}\n")
                f.write(f"Per-image average SSIM: {per_img_ssim:.4f}\n")
        print("Inference Completed !!")

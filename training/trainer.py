import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
import wandb
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from dataset.dataset import ImageDenoisingDataset
from model.model import DnCNN


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._prepare_dataloader()
        self._init_model()

        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)

        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        self.criterion = nn.MSELoss()

        self.train_psnr = PeakSignalNoiseRatio().to(self.device)
        self.train_ssim = StructuralSimilarityIndexMeasure().to(self.device)

        if self.config["validate"]:
            self.val_psnr = PeakSignalNoiseRatio().to(self.device)
            self.val_ssim = StructuralSimilarityIndexMeasure().to(self.device)

        if self.config["wandb_enabled"] and self.config["wandb_mode"] != "disabled":
            exclude_keys = [
                "project_name",
                "wandb_mode",
                "wandb_enabled",
                "noisy_dir",
                "original_dir",
                "val_size",
            ]

            parsed_config = wandb.helper.parse_config(self.config, exclude=exclude_keys)

            self.wandb_run = wandb.init(
                project=self.config["project_name"],
                config=parsed_config,
                mode=self.config["wandb_mode"],
            )
            wandb.watch(models=self.model, criterion=self.criterion, log="gradients")

        else:
            self.wandb_run = None

    def _prepare_dataloader(self):
        print("Preparing Data...")

        dataset = ImageDenoisingDataset(
            noisy_dir=self.config["noisy_dir"],
            original_dir=self.config["original_dir"],
            patch_size=self.config["patch_size"],
            stride=self.config["stride"],
        )

        if self.config["validate"]:
            dataset_size = len(dataset)
            val_size = int(self.config["val_size"] * dataset_size)
            train_size = dataset_size - val_size

            train_data, val_data = random_split(dataset, [train_size, val_size])

            self.train_loader = DataLoader(
                train_data, batch_size=self.config["batch_size"], shuffle=True
            )
            self.val_loader = DataLoader(val_data, batch_size=self.config["batch_size"])
            print("Dataloaders prepared !!")

        else:
            self.train_loader = DataLoader(
                dataset, batch_size=self.config["batch_size"], shuffle=True
            )

    def _init_model(self):
        self.model = DnCNN(
            depth=self.config["depth"], channels=self.config["channels"]
        ).to(self.device)
        print("Model Initialized !!")

    def train(self):
        best_val_psnr = 0
        step = 0
        agg_loss = 0

        for epoch in range(self.config["epochs"]):
            self.train_psnr.reset()
            self.train_ssim.reset()

            self.model.train()
            batch_losses = []

            batch_iterator = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}",
                leave=True,
                dynamic_ncols=True,
            )

            for batch in batch_iterator:
                y = batch["noisy_patch"].to(self.device)
                x = batch["original_patch"].to(self.device)
                target_residual = y - x

                self.optimizer.zero_grad()

                output_residual = self.model(y)

                denoised_image = y - output_residual

                loss = self.criterion(output_residual, target_residual)
                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                agg_loss += loss.item()

                self.train_psnr.update(denoised_image, x)
                self.train_ssim.update(denoised_image, x)

                if self.wandb_run:
                    wandb.log(
                        {"avg loss": agg_loss / (step + 1), "training_step": step},
                        step=step,
                    )
                step += 1

                batch_losses.append(loss.item())

                batch_iterator.set_postfix({"Avg Loss": agg_loss / (step + 1)})

            train_loss = sum(batch_losses) / len(self.train_loader)
            self.scheduler.step()
            print(f"Latest Learning rate: {self.scheduler.get_last_lr()}")

            if (epoch + 1) % self.config["log_interval"] == 0:
                train_psnr = self.train_psnr.compute()
                train_ssim = self.train_ssim.compute()

                if self.config["validate"]:
                    val_loss, val_psnr, val_ssim = self.validate()

                if self.wandb_run:
                    if self.config["validate"]:
                        log_metrics = {
                            "epoch": epoch + 1,
                            "train_loss": train_loss,
                            "train_psnr": train_psnr.item(),
                            "train_ssim": train_ssim.item(),
                            "val_loss": val_loss,
                            "val_psnr": val_psnr,
                            "val_ssim": val_ssim,
                        }
                    else:
                        log_metrics = {
                            "epoch": epoch + 1,
                            "train_loss": train_loss,
                            "train_psnr": train_psnr.item(),
                            "train_ssim": train_ssim.item(),
                        }

                    wandb.log(log_metrics, step=step)

                print(
                    f"Epoch {epoch + 1}/{self.config['epochs']}, Step: {step}, "
                    f"Train Loss: {train_loss:.6f}, "
                    f"Train PSNR: {train_psnr.item():.6f}, Train SSIM: {train_ssim.item():.6f}"
                    + (
                        f", Val Loss: {val_loss:.6f}, Val PSNR: {val_psnr:.6f}, Val SSIM: {val_ssim:.6f}"
                        if self.config["validate"]
                        else ""
                    )
                )
                if self.config["validate"]:
                    if val_psnr > best_val_psnr:
                        best_val_psnr = val_psnr
                        print(
                            f"New best model found at epoch {epoch + 1} with val PSNR: {best_val_psnr:.6f}"
                        )
                        model_path = f"{self.config['checkpoint_dir']}/best_model.pth"
                        torch.save(self.model.state_dict(), model_path)
                        if self.wandb_run:
                            wandb.log({"best_val_psnr": best_val_psnr}, step=step)
                            wandb.save(model_path)
                            self._log_model_as_artifact(model_path)

        if self.wandb_run:
            wandb.finish()
        print("Training completed.")

    def validate(self):
        self.model.eval()
        self.val_psnr.reset()
        self.val_ssim.reset()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                y = batch["noisy_patch"].to(self.device)
                x = batch["original_patch"].to(self.device)
                target_residual = y - x

                outputs_residual = self.model(y)
                # Reconstruct the denoised image
                denoised_image = y - outputs_residual

                loss = self.criterion(outputs_residual, target_residual)
                total_loss += loss.item()

                self.val_psnr.update(denoised_image, x)
                self.val_ssim.update(denoised_image, x)

        val_loss = total_loss / len(self.val_loader)
        return (
            val_loss,
            self.val_psnr.compute().item(),
            self.val_ssim.compute().item(),
        )

    def _log_model_as_artifact(self, model_path):
        artifact = wandb.Artifact(f"best_model", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as transforms
import random

class ImageDenoisingDataset(Dataset):
    def __init__(self, noisy_dir, original_dir, patch_size=50, stride=25):
        self.noisy_dir = noisy_dir
        self.original_dir = original_dir
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        )

        self.noisy_images = sorted(os.listdir(noisy_dir))
        self.original_images = sorted(os.listdir(original_dir))

        assert len(self.noisy_images) == len(self.original_images), (
            "Mismatch in dataset sizes"
        )

        self.patches = self._generate_patches()

    def _generate_patches(self):
        """Generate all patches using a sliding window approach."""
        patches = []

        for img_idx in range(len(self.noisy_images)):
            noisy_path = os.path.join(self.noisy_dir, self.noisy_images[img_idx])
            original_path = os.path.join(
                self.original_dir, self.original_images[img_idx]
            )

            noisy_img = Image.open(noisy_path).convert("RGB")
            original_img = Image.open(original_path).convert("RGB")

            assert noisy_img.size == original_img.size, (
                f"Image size mismatch for {img_idx}"
            )

            w, h = noisy_img.size
            ps = self.patch_size
            stride = self.stride

            for y in range(0, h - ps + 1, stride):
                for x in range(0, w - ps + 1, stride):
                    patches.append((img_idx, x, y))

        return patches

    @staticmethod
    def apply_same_transforms(noisy_img, original_img):
        seed = random.randint(0, 2**32)

        def get_transforms(seed):
            random.seed(seed)
            torch.manual_seed(seed)

            angle = random.uniform(-180, 180)
            flip = random.random() > 0.5

            def transform(img):
                img = F.rotate(img, angle)

                if flip:
                    img = F.hflip(img)

                return img

            return transform

        transform = get_transforms(seed)

        transformed_noisy = transform(noisy_img)
        transformed_original = transform(original_img)

        random.seed(42)
        torch.manual_seed(42)
        
        return transformed_noisy, transformed_original

    
    def __len__(self):
        return len(self.patches)


    def __getitem__(self, idx):
        
        img_idx, x, y = self.patches[idx]

        noisy_path = os.path.join(self.noisy_dir, self.noisy_images[img_idx])
        original_path = os.path.join(self.original_dir, self.original_images[img_idx])

        noisy_img = Image.open(noisy_path).convert("RGB")
        original_img = Image.open(original_path).convert("RGB")

        noisy_patch = noisy_img.crop((x, y, x + self.patch_size, y + self.patch_size))
        original_patch = original_img.crop(
            (x, y, x + self.patch_size, y + self.patch_size)
        )

        transformed_noisy, transformed_original = self.apply_same_transforms(
            noisy_patch, original_patch
        )

        transformed_noisy_tensor = self.transform(transformed_noisy)
        transformed_original_tensor = self.transform(transformed_original)

        return {
            "noisy_patch": transformed_noisy_tensor,
            "original_patch": transformed_original_tensor,
        }

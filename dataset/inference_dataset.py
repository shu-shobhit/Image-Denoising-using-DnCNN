import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from PIL import Image

class InferenceDataset(Dataset):
    """Dataset for inference on noisy images with optional ground truth."""
    def __init__(self, input_dir, original_dir = None):
        self.input_dir = input_dir
        self.original_dir = original_dir
        
        self.transform = transforms.Compose([
            transforms.ToImage(), 
            transforms.ToDtype(torch.float32, scale=True)
        ])
        
        self.input_files = sorted([
            f for f in os.listdir(input_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ])
        
        if len(self.input_files) == 0:
            raise ValueError(f"No valid image files found in {input_dir}")
            
        print(f"Found {len(self.input_files)} images for inference")
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        filename = self.input_files[idx]
        noisy_path = os.path.join(self.input_dir, filename)
        
        noisy_img = Image.open(noisy_path).convert("RGB")
        noisy_tensor = self.transform(noisy_img)
        
        result = {
            "noisy_image": noisy_tensor,
            "filename": filename
        }
        
        if self.original_dir is not None:
            original_path = os.path.join(self.original_dir, filename)
            if os.path.exists(original_path):
                original_img = Image.open(original_path).convert("RGB")
                original_tensor = self.transform(original_img)
                result["original_image"] = original_tensor
        
        return result
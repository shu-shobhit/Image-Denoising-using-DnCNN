import os
import argparse
import torch

from configurations.train_config import train_config
from configurations.inference_config import inference_config
from training.trainer import Trainer
from inference.inference import Inference


def parse_args():
    parser = argparse.ArgumentParser(description="DnCNN Image Denoising")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["train", "inference"], 
        default="train",
        help="Operation mode: train or inference"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.mode == "train":
        
        print("Starting training with the following configuration:")
        for k, v in train_config.items():
            print(f"{k}: {v}")
        
        trainer = Trainer(train_config)
        trainer.train()
        
    elif args.mode == "inference":
        
        print("Starting inference with the following configuration:")
        for k, v in inference_config.items():
            print(f"{k}: {v}")
        
        inference = Inference(inference_config)
        inference.run()


if __name__ == "__main__":
    main()
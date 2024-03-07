#!/usr/bin/env python3

import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from torchinfo import summary
from pathlib import Path

import data_setup, engine
from helper_functions import download_data, set_seeds, plot_loss_curves


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

image_path = download_data(source="https://github.com/aadityapattabhiraman/pytorch0/raw/main/data/pizza_steak_sushi.zip",
                           destination="pizza_steak_sushi")
print(image_path)

train_dir = image_path / "train"
test_dir = image_path / "test"

img_size = 224
manual_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size))
])
print(f"Manually created transforms: {manual_transforms}")

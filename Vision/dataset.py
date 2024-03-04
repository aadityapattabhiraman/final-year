#!/usr/bin/env python3

import torch
from torch import nn


# define the device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

image_path = "/home/akugyo/GitHub/final-year/Vision/pizza_steak_sushi/"
train_dir = image_path / "train"
test_dir = image_path / "test"

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

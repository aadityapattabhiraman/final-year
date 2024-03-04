#!/usr/bin/env python3

import torch
from torch import nn


# define the device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
print("Using device:", device)
if device == "cuda":
    print(f"Device name: {torch.cuda.get_device_name(device.index)}")
    print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
elif (device == "mps"):
    print(f"Device name: <mps>")
else:
    print("NOTE: If you have a GPU, consider using it or training.")
device = torch.device(device)

# Make sure the weights

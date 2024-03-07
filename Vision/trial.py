#!/usr/bin/env python3

import datasetup
import matplotlib.pyplot as plt
from dataset import dataset
from helper_functions import download_data

dataset = dataset()

batch_size = 32
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir = dataset.train_dir,
    test_dir = dataset.train_dir,
    transforms = dataset.manual_transforms,
    batch_size = batch_size
)

print(train_dataloader, test_dataloader, class_names)

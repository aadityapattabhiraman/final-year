#!/usr/bin/env python3

import data_setup
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import dataset
from helper_functions import download_data

dataset = dataset()

batch_size = 32
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir = dataset["train_dir"],
    test_dir = dataset["train_dir"],
    transform = dataset["manual_transforms"],
    batch_size = batch_size
)

print(train_dataloader, test_dataloader, class_names)

image_batch, label_batch = next(iter(train_dataloader))
image, label = image_batch[0], label_batch[0]
print(image.shape, label)

plt.imshow(image.permute(1, 2, 0))
plt.title(class_names[label])
plt.axis(False)
plt.show()

#!/usr/bin/env python3

from torchvision.io import read_image
from torchvision.models import ViT_L_16_Weights
from model import vit_l_16


img = read_image("/home/akugyo/Whitepersiancatoncouch-aba536ea9760403dac2042cc4c47144d-2576739925.jpg")

# Step 1: Initialize model with the best available weights
weights = ViT_L_16_Weights.IMAGENET1K_V1
model = vit_l_16(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")

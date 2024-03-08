#!/usr/bin/env python3

import torchvision
from model import vision_transformer


vit_weights = torchvision.models.ViT_L_16_Weights.DEFAULT
model = vision_transformer(patch_size=16, mlp_dim=4096, hidden_dim=1024, num_heads=16, num_layers=24, weights=vit_weights, progress=True)
print(model)

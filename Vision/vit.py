#!/usr/bin/env python3

import torchvision
from model import vision_transformer
from typing import List, Tuple
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


vit_weights = torchvision.models.ViT_L_16_Weights.DEFAULT
model = vision_transformer(patch_size=16, mlp_dim=4096, hidden_dim=1024, num_heads=16, num_layers=24, weights=vit_weights, progress=True)
print(model)


def pred(model: torch.nn.Module, class_names: List[str], image_path: str, image_size: Tuple[int, int]=(224,224),
         transform: torchvision.transforms=None, device: torch.device="cuda"):
    img = Image.open(image_path)

    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])

    model.to(device)
    model.eval()

    with torch.inference_mode():
        transformed_image = image_transform(img).unsqueeze(dim=0)
        target_image_pred = model(transformed_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
    )
    plt.axis(False)
    plt.show()

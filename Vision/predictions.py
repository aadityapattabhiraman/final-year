#!/usr/bin/env python3

from typing import List, Tuple
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch, torchvision


def pred(model: torch.nn.Module, weights, image_path: str, image_size: Tuple[int, int]=(224,224),
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

    batch = image_transform(img).unsqueeze(0)
    prediction = model(batch.to(device)).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]

    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Pred: {category_name} | Prob: {100 * score:.1f}%"
    )
    plt.axis(False)
    plt.show()

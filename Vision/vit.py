#!/usr/bin/env python3

import torchvision
from PIL import Image
from model import vit
import matplotlib.pyplot as plt


weights = torchvision.models.ViT_L_16_Weights.DEFAULT
img = Image.open(str(input("Enter image path: ")))
model = vit(weights=weights).to("cuda")
model.eval()

preprocess = weights.transforms()
batch = preprocess(img).unsqueeze(0).to("cuda")
predictions = model(batch).squeeze(0).softmax(0)
class_id = predictions.argmax().item()
score = predictions[class_id].item()
category_name = weights.meta["categories"][class_id]

plt.figure()
plt.imshow(img)
plt.title(
    f"Pred: {category_name} | Prob: {100 * score:.1f}%"
)
plt.axis(False)
plt.show()

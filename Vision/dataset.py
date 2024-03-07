#!/usr/bin/env python3

from helper_functions import download_data


def dataset():
    image_path = download_data(source="https://github.com/aadityapattabhiraman/pytorch0/raw/main/data/pizza_steak_sushi.zip",
                           destination="pizza_steak_sushi")
    print(image_path)

    train_dir = image_path / "train"
    test_dir = image_path / "test"

    img_size = 224
    manual_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size))
    ])
    print(f"Manually created transforms: {manual_transforms}")

    return {"train_dir": train_dir, "test_dir": test_dir, "manual_transforms": manual_transforms}

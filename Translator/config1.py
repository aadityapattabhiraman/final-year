#!/usr/bin/env python3
from pathlib import Path


def get_config():
    return {
        "batch_size": 7,
        "num_epochs": 60,
        "lr": 10**-3,
        "seq_len": 512,
        "d_model": 512,
        "datasource": "ai4bharat/samanantar",
        "lang_src": "en",
        "lang_tgt": "ta",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch:str):
    model_folder = "/home/akugyo/GitHub/final-year/Translator/kde4_weights/"
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = f"kde4_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

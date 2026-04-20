"""
Usage:
    python train.py configs/wavlm_base_head.yaml
    python train.py configs/hubert_large_lora.yaml --lr 5e-5 --epochs 5
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from transformers import AutoFeatureExtractor

from src.config import load_config
from src.dataset import get_dataloaders
from src.models import build_model
from src.trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to experiment YAML config")
    parser.add_argument("--override", nargs="*", metavar="KEY=VALUE", default=[])
    args = parser.parse_args()

    config = load_config(args.config)

    # apply CLI overrides
    for kv in args.override:
        key, _, val = kv.partition("=")
        if not hasattr(config, key):
            raise ValueError(f"Unknown config key: {key!r}")
        attr_type = type(getattr(config, key))
        setattr(config, key, attr_type(val))

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:   {device}", flush=True)

    processor = AutoFeatureExtractor.from_pretrained(config.processor_name or config.model_name)
    train_loader, dev_loader, test_loader = get_dataloaders(config, processor)

    print(f"Train: {len(train_loader.dataset)} | Dev: {len(dev_loader.dataset)} | Test: {len(test_loader.dataset)}", flush=True)

    model = build_model(config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params:   {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)", flush=True)

    trainer = Trainer(model, config, train_loader, dev_loader, test_loader, device)
    trainer.fit()


if __name__ == "__main__":
    main()

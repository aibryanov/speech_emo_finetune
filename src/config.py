from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


CONFIGS_DIR = Path(__file__).parent.parent / "configs"


@dataclass
class ExperimentConfig:
    # --- model ---
    model_name: str = "microsoft/wavlm-base"
    fine_tune_strategy: str = "head_only"  # head_only | lora | top_n | full | lstm

    # --- LoRA ---
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # --- top_n ---
    top_n_layers: int = 4

    # --- LSTM head ---
    lstm_hidden: int = 256
    lstm_layers: int = 2
    lstm_dropout: float = 0.1

    # --- training ---
    batch_size: int = 8
    grad_accum_steps: int = 4
    lr: float = 3e-4
    epochs: int = 10
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1

    # --- data ---
    dev_ratio: float = 0.15
    seed: int = 42
    max_audio_len_s: float = 10.0
    num_workers: int = 2

    # --- output ---
    output_dir: str = "outputs/experiment"
    run_name: str = "experiment"

    # --- misc ---
    num_labels: int = 7


def load_config(path: str) -> ExperimentConfig:
    base_path = CONFIGS_DIR / "base.yaml"
    with open(base_path) as f:
        base = yaml.safe_load(f) or {}

    with open(path) as f:
        override = yaml.safe_load(f) or {}

    merged = {**base, **override}

    cfg = ExperimentConfig()
    for k, v in merged.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    return cfg

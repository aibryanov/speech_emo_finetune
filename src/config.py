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
    processor_name: Optional[str] = None  # if None, falls back to model_name
    fine_tune_strategy: str = "head_only"  # head_only | lora | top_n | full | lstm

    # --- LoRA ---
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # --- top_n ---
    top_n_layers: int = 4

    # --- LSTM on audio features (fine_tune_strategy == "lstm_features") ---
    lstm_hidden: int = 256
    lstm_layers: int = 2
    lstm_dropout: float = 0.1
    # feature_type: mfcc | mfcc_delta | logmel | combined (mfcc_delta + logmel)
    feature_type: str = "combined"
    n_mfcc: int = 40
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160

    # --- training ---
    head_lr: float = 1e-3  # classifier head learning rate (independent of lr)
    batch_size: int = 8
    grad_accum_steps: int = 4
    lr: float = 3e-4
    epochs: int = 10
    weight_decay: float = 1e-2
    fp16: bool = False      # mixed precision training (faster on GPU with Tensor Cores)
    warmup_ratio: float = 0.1
    warmup_steps: int = 0  # if > 0, overrides warmup_ratio
    scheduler_type: str = "linear"  # linear | cosine
    label_smoothing: float = 0.0
    use_focal_loss: bool = False
    focal_gamma: float = 2.0

    # --- augmentation ---
    augment: bool = False
    aug_noise_std: float = 0.005
    aug_time_mask_ratio: float = 0.1
    aug_amplitude_range: List[float] = field(default_factory=lambda: [0.8, 1.2])
    use_spec_augment: bool = False

    # --- data ---
    train_ratio: float = 0.0  # if > 0, use only this fraction of data for train (rest discarded)
    dev_ratio: float = 0.15
    eval_every_n_steps: int = 0  # if > 0, eval every N optimizer steps instead of every epoch
    seed: int = 42
    max_audio_len_s: float = 10.0
    num_workers: int = 2

    # --- checkpointing ---
    save_every_n_epochs: int = 5
    resume_from: str = ""  # path to checkpoint to resume from; empty = start fresh

    # --- output ---
    output_dir: str = "outputs/experiment"
    run_name: str = "experiment"

    # --- MLP head (lstm_features and fusion) ---
    mlp_hidden: List[int] = field(default_factory=lambda: [256])
    mlp_dropout: float = 0.3

    # --- fusion model ---
    fusion_hidden: int = 512
    fusion_dropout: float = 0.3
    lstm_pretrained: str = ""  # path to pre-trained LSTMFeaturesModel checkpoint

    # --- Dusha dataset ---
    aggregated_tsv: str = ""   # path to aggregated TSV (Dawid-Skene or majority)
    audio_dir: str = ""        # root directory containing audio files

    # --- misc ---
    merge_labels: bool = False  # merge 7 classes → 4 (positive/negative/neutral/other)
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

    if cfg.merge_labels and "num_labels" not in override:
        cfg.num_labels = 4

    return cfg

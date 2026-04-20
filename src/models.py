from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from src.config import ExperimentConfig
from src.dataset import NUM_LABELS


class SpeechEmotionModel(nn.Module):
    """Wraps a transformer backbone with a classification head."""

    def __init__(self, backbone: nn.Module, hidden_size: int, num_labels: int = NUM_LABELS):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(hidden_size, num_labels)

    def _mean_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden: (B, T, H), attention_mask: (B, T)
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        return summed / lengths  # (B, H)

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None):
        out = self.backbone(input_values=input_values, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, T_out, H)

        # Downsample waveform-resolution mask to backbone output resolution
        T_out = hidden.shape[1]
        if attention_mask is not None and attention_mask.shape[1] != T_out:
            idx = (torch.arange(T_out, device=hidden.device).float()
                   * (attention_mask.shape[1] / T_out)).long().clamp(max=attention_mask.shape[1] - 1)
            out_mask = attention_mask[:, idx]
        else:
            out_mask = attention_mask if attention_mask is not None else torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.long)

        logits = self.classifier(self._mean_pool(hidden, out_mask))
        return logits


class LSTMSpeechEmotionModel(nn.Module):
    """Frozen backbone → BiLSTM → classifier."""

    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        lstm_hidden: int,
        lstm_layers: int,
        lstm_dropout: float,
        num_labels: int = NUM_LABELS,
    ):
        super().__init__()
        self.backbone = backbone
        self.lstm = nn.LSTM(
            hidden_size,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(lstm_hidden * 2, num_labels)

    def _mean_pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        return summed / lengths

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None):
        with torch.no_grad():
            out = self.backbone(input_values=input_values, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, T, H)
        lstm_out, _ = self.lstm(hidden)  # (B, T, 2*lstm_hidden)

        if attention_mask is None:
            attention_mask = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.long)

        T_out = lstm_out.shape[1]
        if attention_mask.shape[1] != T_out:
            idx = (torch.arange(T_out, device=lstm_out.device).float()
                   * (attention_mask.shape[1] / T_out)).long().clamp(max=attention_mask.shape[1] - 1)
            out_mask = attention_mask[:, idx]
        else:
            out_mask = attention_mask

        pooled = self._mean_pool(lstm_out, out_mask)
        logits = self.classifier(pooled)
        return logits


def _lstm_feature_dim(config) -> int:
    if config.feature_type == "mfcc":
        return config.n_mfcc
    elif config.feature_type == "mfcc_delta":
        return config.n_mfcc * 3
    elif config.feature_type == "logmel":
        return config.n_mels
    else:  # combined
        return config.n_mfcc * 3 + config.n_mels


class LSTMFeaturesModel(nn.Module):
    """BiLSTM trained on handcrafted audio features (MFCC, log-mel, deltas).
    No transformer backbone — standalone model.
    """

    def __init__(self, feature_dim: int, lstm_hidden: int, lstm_layers: int, lstm_dropout: float, num_labels: int = NUM_LABELS):
        super().__init__()
        self.lstm = nn.LSTM(
            feature_dim, lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(lstm_hidden * 2, num_labels)

    def forward(self, features: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs):
        # features: (B, T, feature_dim)
        lstm_out, _ = self.lstm(features)  # (B, T, 2*lstm_hidden)

        if attention_mask is None:
            attention_mask = torch.ones(lstm_out.shape[:2], device=lstm_out.device, dtype=torch.long)

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (lstm_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.classifier(pooled)


def _get_backbone_hidden_size(backbone: nn.Module) -> int:
    cfg = backbone.config
    for attr in ("hidden_size", "d_model", "encoder_embed_dim"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    raise ValueError("Cannot determine backbone hidden size from config.")


def _freeze_all(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


def _freeze_backbone_except_top_n(backbone: nn.Module, top_n: int):
    _freeze_all(backbone)
    # unfreeze last top_n encoder layers
    encoder = None
    for attr in ("encoder", "hubert", "wav2vec2", "wavlm"):
        if hasattr(backbone, attr):
            sub = getattr(backbone, attr)
            if hasattr(sub, "encoder"):
                encoder = sub.encoder
                break
            encoder = sub
            break
    if encoder is None:
        raise ValueError("Cannot locate encoder layers in backbone.")

    layers = None
    for attr in ("layers", "layer"):
        if hasattr(encoder, attr):
            layers = getattr(encoder, attr)
            break
    if layers is None:
        raise ValueError("Cannot locate layers in encoder.")

    for layer in layers[-top_n:]:
        for p in layer.parameters():
            p.requires_grad_(True)


def build_model(config: ExperimentConfig) -> nn.Module:
    backbone = AutoModel.from_pretrained(config.model_name)

    if getattr(config, "use_spec_augment", False):
        backbone.config.apply_spec_augment = True

    hidden_size = _get_backbone_hidden_size(backbone)
    strategy = config.fine_tune_strategy

    if strategy == "head_only":
        _freeze_all(backbone)
        model = SpeechEmotionModel(backbone, hidden_size, config.num_labels)

    elif strategy == "lora":
        from peft import LoraConfig, get_peft_model, TaskType

        lora_cfg = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
        )
        backbone = get_peft_model(backbone, lora_cfg)
        model = SpeechEmotionModel(backbone, hidden_size, config.num_labels)

    elif strategy == "top_n":
        _freeze_backbone_except_top_n(backbone, config.top_n_layers)
        model = SpeechEmotionModel(backbone, hidden_size, config.num_labels)

    elif strategy == "full":
        model = SpeechEmotionModel(backbone, hidden_size, config.num_labels)

    elif strategy == "lstm_features":
        feature_dim = _lstm_feature_dim(config)
        model = LSTMFeaturesModel(feature_dim, config.lstm_hidden, config.lstm_layers, config.lstm_dropout, config.num_labels)
        return model  # no backbone needed

    else:
        raise ValueError(f"Unknown fine_tune_strategy: {strategy!r}")

    return model

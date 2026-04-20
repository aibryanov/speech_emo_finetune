from __future__ import annotations

from functools import partial
from typing import Dict, List, Tuple

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor

from src.config import ExperimentConfig

LABEL2ID: Dict[str, int] = {
    "happiness": 0,
    "sadness": 1,
    "anger": 2,
    "fear": 3,
    "disgust": 4,
    "enthusiasm": 5,
    "neutral": 6,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

SAMPLE_RATE = 16_000


def _decode_audio(speech_field) -> tuple:
    """Return (waveform: float32 tensor 1D, sample_rate: int).
    Handles both torchcodec AudioDecoder and legacy dict format.
    """
    try:
        samples = speech_field.get_all_samples()
        waveform = samples.data  # (channels, time)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        return waveform.float(), int(samples.sample_rate)
    except AttributeError:
        return torch.tensor(speech_field["array"], dtype=torch.float32), speech_field["sampling_rate"]


def load_resd():
    """Return raw HF DatasetDict with 'train' and 'test' splits."""
    return load_dataset("Aniemore/resd")


def stratified_split(hf_dataset, dev_ratio: float, seed: int):
    """Split an HF Dataset into (train_subset, dev_subset) stratified by label."""
    labels = [LABEL2ID[ex["emotion"]] for ex in hf_dataset]
    import numpy as np
    indices = np.arange(len(hf_dataset))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=dev_ratio, random_state=seed)
    train_idx, dev_idx = next(sss.split(indices, labels))

    return hf_dataset.select(train_idx.tolist()), hf_dataset.select(dev_idx.tolist())


class EmotionDataset(Dataset):
    def __init__(self, hf_dataset, processor: AutoFeatureExtractor, max_len_samples: int):
        self.data = hf_dataset
        self.processor = processor
        self.max_len_samples = max_len_samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.data[idx]
        waveform, sr = _decode_audio(ex["speech"])

        if sr != SAMPLE_RATE:
            waveform = F.resample(waveform.unsqueeze(0), sr, SAMPLE_RATE).squeeze(0)

        waveform = waveform[: self.max_len_samples]

        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].squeeze(0)  # (T,)

        label = torch.tensor(LABEL2ID[ex["emotion"]], dtype=torch.long)
        return {"input_values": input_values, "label": label}


def collate_fn(batch: List[Dict], max_len_samples: int) -> Dict[str, torch.Tensor]:
    lengths = [min(ex["input_values"].shape[0], max_len_samples) for ex in batch]
    max_len = max(lengths)

    B = len(batch)
    padded = torch.zeros(B, max_len, dtype=torch.float32)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)

    for i, (ex, L) in enumerate(zip(batch, lengths)):
        padded[i, :L] = ex["input_values"][:L]
        attention_mask[i, :L] = 1

    labels = torch.stack([ex["label"] for ex in batch])
    return {
        "input_values": padded,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _extract_features(waveform: torch.Tensor, config: ExperimentConfig) -> torch.Tensor:
    """Extract handcrafted audio features. Returns (T_frames, feature_dim)."""
    mfcc_transform = T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=config.n_mfcc,
        melkwargs={"n_fft": config.n_fft, "hop_length": config.hop_length, "n_mels": config.n_mels},
    )
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=config.n_fft, hop_length=config.hop_length, n_mels=config.n_mels
    )
    db_transform = T.AmplitudeToDB()

    x = waveform.unsqueeze(0)  # (1, T)

    if config.feature_type == "mfcc":
        feat = mfcc_transform(x).squeeze(0)  # (n_mfcc, T_frames)

    elif config.feature_type == "mfcc_delta":
        mfcc = mfcc_transform(x).squeeze(0)
        delta = torchaudio.functional.compute_deltas(mfcc)
        delta2 = torchaudio.functional.compute_deltas(delta)
        feat = torch.cat([mfcc, delta, delta2], dim=0)  # (3*n_mfcc, T_frames)

    elif config.feature_type == "logmel":
        feat = db_transform(mel_transform(x)).squeeze(0)  # (n_mels, T_frames)

    else:  # combined: mfcc_delta + logmel
        mfcc = mfcc_transform(x).squeeze(0)
        delta = torchaudio.functional.compute_deltas(mfcc)
        delta2 = torchaudio.functional.compute_deltas(delta)
        logmel = db_transform(mel_transform(x)).squeeze(0)
        feat = torch.cat([mfcc, delta, delta2, logmel], dim=0)  # (3*n_mfcc+n_mels, T_frames)

    return feat.transpose(0, 1)  # (T_frames, feature_dim)


class AudioFeaturesDataset(Dataset):
    """Dataset that extracts MFCC/mel features — used for lstm_features strategy."""

    def __init__(self, hf_dataset, config: ExperimentConfig, max_len_samples: int):
        self.data = hf_dataset
        self.config = config
        self.max_len_samples = max_len_samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.data[idx]
        waveform, sr = _decode_audio(ex["speech"])

        if sr != SAMPLE_RATE:
            waveform = F.resample(waveform.unsqueeze(0), sr, SAMPLE_RATE).squeeze(0)

        waveform = waveform[: self.max_len_samples]
        features = _extract_features(waveform, self.config)  # (T_frames, feature_dim)
        label = torch.tensor(LABEL2ID[ex["emotion"]], dtype=torch.long)
        return {"features": features, "label": label}


def features_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    lengths = [ex["features"].shape[0] for ex in batch]
    max_len = max(lengths)
    feature_dim = batch[0]["features"].shape[1]

    B = len(batch)
    padded = torch.zeros(B, max_len, feature_dim, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.long)

    for i, (ex, L) in enumerate(zip(batch, lengths)):
        padded[i, :L] = ex["features"]
        mask[i, :L] = 1

    labels = torch.stack([ex["label"] for ex in batch])
    return {"features": padded, "attention_mask": mask, "labels": labels}


def get_feature_dataloaders(config: ExperimentConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    raw = load_resd()
    train_raw, dev_raw = stratified_split(raw["train"], config.dev_ratio, config.seed)
    test_raw = raw["test"]

    max_len = int(config.max_audio_len_s * SAMPLE_RATE)

    train_ds = AudioFeaturesDataset(train_raw, config, max_len)
    dev_ds = AudioFeaturesDataset(dev_raw, config, max_len)
    test_ds = AudioFeaturesDataset(test_raw, config, max_len)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, collate_fn=features_collate_fn, pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=config.batch_size * 2, shuffle=False,
        num_workers=config.num_workers, collate_fn=features_collate_fn, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size * 2, shuffle=False,
        num_workers=config.num_workers, collate_fn=features_collate_fn, pin_memory=True,
    )
    return train_loader, dev_loader, test_loader


def get_dataloaders(
    config: ExperimentConfig, processor: AutoFeatureExtractor
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    raw = load_resd()
    train_raw, dev_raw = stratified_split(raw["train"], config.dev_ratio, config.seed)
    test_raw = raw["test"]

    max_len = int(config.max_audio_len_s * SAMPLE_RATE)
    _collate = partial(collate_fn, max_len_samples=max_len)

    train_ds = EmotionDataset(train_raw, processor, max_len)
    dev_ds = EmotionDataset(dev_raw, processor, max_len)
    test_ds = EmotionDataset(test_raw, processor, max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=_collate,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=_collate,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=_collate,
        pin_memory=True,
    )
    return train_loader, dev_loader, test_loader

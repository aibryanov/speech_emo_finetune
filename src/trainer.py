from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import ExperimentConfig
from src.metrics import compute_metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.device = device

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)

        total_steps = (len(train_loader) // config.grad_accum_steps) * config.epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        self.criterion = nn.CrossEntropyLoss()
        self.best_metric = -1.0
        self.metrics_log: list = []

    # ------------------------------------------------------------------
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        steps = 0
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [train]", leave=False)
        for i, batch in enumerate(pbar):
            input_values = batch["input_values"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(input_values=input_values, attention_mask=attention_mask)
            loss = self.criterion(logits, labels) / self.config.grad_accum_steps
            loss.backward()

            if (i + 1) % self.config.grad_accum_steps == 0 or (i + 1) == len(self.train_loader):
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                steps += 1

            total_loss += loss.item() * self.config.grad_accum_steps
            pbar.set_postfix(loss=f"{loss.item() * self.config.grad_accum_steps:.4f}")

        return total_loss / len(self.train_loader)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader, split: str) -> Dict:
        self.model.eval()
        all_preds, all_labels = [], []

        for batch in tqdm(loader, desc=f"  [{split}]", leave=False):
            input_values = batch["input_values"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"]

            logits = self.model(input_values=input_values, attention_mask=attention_mask)
            preds = logits.argmax(dim=-1).cpu()

            all_preds.append(preds)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        return compute_metrics(all_preds, all_labels)

    # ------------------------------------------------------------------
    def _log(self, record: Dict):
        print(json.dumps(record, indent=None))
        self.metrics_log.append(record)
        with open(self.output_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    def save_checkpoint(self, path: Path):
        torch.save({"model_state_dict": self.model.state_dict()}, path)

    def load_checkpoint(self, path: Path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])

    # ------------------------------------------------------------------
    def fit(self):
        best_ckpt = self.output_dir / "best_model.pt"

        for epoch in range(self.config.epochs):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            dev_metrics = self.eval_epoch(self.dev_loader, "dev")
            elapsed = time.time() - t0

            record = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "dev_accuracy": round(dev_metrics["accuracy"], 4),
                "dev_weighted_accuracy": round(dev_metrics["weighted_accuracy"], 4),
                "dev_f1_macro": round(dev_metrics["f1_macro"], 4),
                "dev_f1_weighted": round(dev_metrics["f1_weighted"], 4),
                "elapsed_s": round(elapsed, 1),
            }
            self._log(record)

            if dev_metrics["f1_weighted"] > self.best_metric:
                self.best_metric = dev_metrics["f1_weighted"]
                self.save_checkpoint(best_ckpt)
                print(f"  -> New best (f1_weighted={self.best_metric:.4f}), saved checkpoint.")

        # final evaluation on test set using best checkpoint
        print("\nLoading best checkpoint for test evaluation...")
        self.load_checkpoint(best_ckpt)
        test_metrics = self.eval_epoch(self.test_loader, "test")

        test_record = {
            "split": "test",
            "accuracy": round(test_metrics["accuracy"], 4),
            "weighted_accuracy": round(test_metrics["weighted_accuracy"], 4),
            "f1_macro": round(test_metrics["f1_macro"], 4),
            "f1_weighted": round(test_metrics["f1_weighted"], 4),
            "per_class": test_metrics["per_class"],
            "confusion_matrix": test_metrics["confusion_matrix"],
        }
        self._log(test_record)
        print("\n=== Test Results ===")
        print(f"  accuracy:          {test_record['accuracy']:.4f}")
        print(f"  weighted_accuracy: {test_record['weighted_accuracy']:.4f}")
        print(f"  f1_macro:          {test_record['f1_macro']:.4f}")
        print(f"  f1_weighted:       {test_record['f1_weighted']:.4f}")
        print("\nPer-class metrics:")
        for cls, vals in test_metrics["per_class"].items():
            print(f"  {cls:12s}  P={vals['precision']:.3f}  R={vals['recall']:.3f}  F1={vals['f1']:.3f}  n={int(vals['support'])}")

        return test_metrics

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
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

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
        if getattr(config, "scheduler_type", "linear") == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
        else:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=getattr(config, "label_smoothing", 0.0))
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
        self.metrics_log.append(record)
        with open(self.output_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    def save_checkpoint(self, path: Path, epoch: int):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": self.best_metric,
        }, path)

    def load_checkpoint(self, path: Path) -> int:
        """Load full checkpoint. Returns the next epoch index to start from."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "best_metric" in ckpt:
            self.best_metric = ckpt["best_metric"]
        return ckpt.get("epoch", 0)

    def _load_best_model(self, path: Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])

    # ------------------------------------------------------------------
    def fit(self, checkpoint_callback=None):
        best_ckpt = self.output_dir / "best_model.pt"
        save_every = getattr(self.config, "save_every_n_epochs", 5)

        start_epoch = 0
        resume_from = getattr(self.config, "resume_from", "")
        if resume_from and Path(resume_from).exists():
            start_epoch = self.load_checkpoint(Path(resume_from))
            print(f"Resumed from {resume_from} — starting at epoch {start_epoch + 1}", flush=True)

        header = f"{'Epoch':>5}  {'Loss':>7}  {'Acc':>6}  {'WAcc':>6}  {'F1mac':>6}  {'F1w':>6}  {'Time':>6}  {'Best':>4}"
        sep = "-" * len(header)
        print(f"\nRun: {self.config.run_name}  |  strategy: {self.config.fine_tune_strategy}  |  model: {self.config.model_name}", flush=True)
        print(sep, flush=True)
        print(header, flush=True)
        print(sep, flush=True)

        for epoch in range(start_epoch, self.config.epochs):
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

            is_best = dev_metrics["f1_weighted"] > self.best_metric
            if is_best:
                self.best_metric = dev_metrics["f1_weighted"]
                self.save_checkpoint(best_ckpt, epoch + 1)

            is_periodic = save_every > 0 and (epoch + 1) % save_every == 0
            is_first = (epoch + 1 == start_epoch + 1)
            if is_periodic or is_first:
                periodic_ckpt = self.output_dir / f"checkpoint_epoch_{epoch+1:03d}.pt"
                self.save_checkpoint(periodic_ckpt, epoch + 1)
                if checkpoint_callback is not None:
                    checkpoint_callback(periodic_ckpt)

            print(
                f"{epoch+1:>5}  {train_loss:>7.4f}  "
                f"{dev_metrics['accuracy']:>6.4f}  {dev_metrics['weighted_accuracy']:>6.4f}  "
                f"{dev_metrics['f1_macro']:>6.4f}  {dev_metrics['f1_weighted']:>6.4f}  "
                f"{elapsed:>5.0f}s  {'*' if is_best else ''}",
                flush=True,
            )

        print(sep, flush=True)

        # final evaluation on test set using best checkpoint
        print(f"\nLoading best checkpoint (f1_weighted={self.best_metric:.4f})...", flush=True)
        self._load_best_model(best_ckpt)
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

        print("\n=== Test Results ===", flush=True)
        print(f"  accuracy          {test_record['accuracy']:.4f}", flush=True)
        print(f"  weighted_accuracy {test_record['weighted_accuracy']:.4f}", flush=True)
        print(f"  f1_macro          {test_record['f1_macro']:.4f}", flush=True)
        print(f"  f1_weighted       {test_record['f1_weighted']:.4f}", flush=True)
        print(f"\n{'Class':12s}  {'P':>6}  {'R':>6}  {'F1':>6}  {'n':>5}", flush=True)
        for cls, vals in test_metrics["per_class"].items():
            print(f"  {cls:12s}  {vals['precision']:6.3f}  {vals['recall']:6.3f}  {vals['f1']:6.3f}  {int(vals['support']):5d}", flush=True)

        return test_metrics

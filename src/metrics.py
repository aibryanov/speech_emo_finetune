from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.dataset import ID2LABEL, NUM_LABELS


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict:
    label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]

    report = classification_report(
        labels, preds, labels=list(range(NUM_LABELS)), target_names=label_names, output_dict=True, zero_division=0
    )

    per_class = {
        name: {
            "precision": report[name]["precision"],
            "recall": report[name]["recall"],
            "f1": report[name]["f1-score"],
            "support": report[name]["support"],
        }
        for name in label_names
        if name in report
    }

    cm = confusion_matrix(labels, preds, labels=list(range(NUM_LABELS)))

    return {
        "accuracy": accuracy_score(labels, preds),
        "weighted_accuracy": balanced_accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }

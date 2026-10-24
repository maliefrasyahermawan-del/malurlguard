"""Model evaluation — metrics, confusion matrix, latency benchmark."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from malurlguard.models.base import LABEL_MAP

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Compute and persist evaluation metrics for a trained model."""

    def __init__(self, output_dir: str | Path = "outputs") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """
        Compute classification metrics.

        Args:
            y_true: Ground-truth label indices.
            y_pred: Predicted label indices.
            y_proba: Probability matrix (n_samples, n_classes), optional.
            model_name: Name used for output file prefixes.

        Returns:
            Dictionary of metrics.
        """
        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        label_names = [LABEL_MAP.get(i, str(i)) for i in unique_labels]

        metrics: Dict[str, Any] = {
            "model": model_name,
            "n_samples": int(len(y_true)),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision_macro": round(
                float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
            ),
            "recall_macro": round(
                float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
            ),
            "f1_macro": round(
                float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4
            ),
        }

        if y_proba is not None and y_proba.shape[1] > 1:
            try:
                auc = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro", labels=unique_labels
                )
                metrics["roc_auc_macro"] = round(float(auc), 4)
            except Exception:
                pass

        # Per-class metrics
        report = classification_report(
            y_true, y_pred, labels=unique_labels, target_names=label_names, output_dict=True, zero_division=0
        )
        metrics["per_class"] = {
            name: {
                "precision": round(report[name]["precision"], 4),
                "recall": round(report[name]["recall"], 4),
                "f1": round(report[name]["f1-score"], 4),
                "support": int(report[name]["support"]),
            }
            for name in label_names
            if name in report
        }

        logger.info(
            "[%s] accuracy=%.4f f1_macro=%.4f",
            model_name, metrics["accuracy"], metrics["f1_macro"],
        )

        # Persist JSON
        json_path = self.output_dir / f"metrics_{model_name}.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved to %s", json_path)

        # Confusion matrix plot
        self._plot_confusion_matrix(y_true, y_pred, unique_labels, label_names, model_name)

        return metrics

    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[int],
        label_names: List[str],
        model_name: str,
    ) -> None:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set(
            xticks=range(len(label_names)),
            yticks=range(len(label_names)),
            xticklabels=label_names,
            yticklabels=label_names,
            xlabel="Predicted",
            ylabel="True",
            title=f"Confusion Matrix — {model_name}",
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        out_path = self.output_dir / f"confusion_matrix_{model_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix saved to %s", out_path)

    def benchmark_latency(
        self, predictor_fn, feature_matrix: np.ndarray, n_trials: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark inference latency.

        Args:
            predictor_fn: Callable that takes an (n, features) array and returns predictions.
            feature_matrix: Sample feature matrix for timing.
            n_trials: Number of timing iterations.

        Returns:
            Dict with mean_ms, p50_ms, p95_ms, p99_ms.
        """
        # Warm up
        predictor_fn(feature_matrix[:1])

        latencies: List[float] = []
        single = feature_matrix[:1]
        for _ in range(n_trials):
            t0 = time.perf_counter()
            predictor_fn(single)
            latencies.append((time.perf_counter() - t0) * 1000)

        arr = np.array(latencies)
        result = {
            "mean_ms": round(float(arr.mean()), 3),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
        }
        logger.info("Latency benchmark: %s", result)
        return result

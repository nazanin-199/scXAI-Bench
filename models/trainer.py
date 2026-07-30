"""Training loop for scXAI-Bench classifiers.

`Trainer` is model-agnostic: it only depends on `BaseClassifier`'s
`forward` method plus standard PyTorch `DataLoader`s, so any future
model dropped into `models/` trains with this same code unchanged.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scxai_bench.models.base import BaseClassifier
from scxai_bench.utils import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters for a training run. Every field has a simple default.

    Attributes:
        epochs: Maximum number of training epochs.
        batch_size: Batch size. Not used directly by `Trainer` (callers
            build their own `DataLoader`s), kept here so the full set of
            hyperparameters for a run lives in one place.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay (L2 regularization).
        patience: Number of epochs with no validation-loss improvement
            (beyond `min_delta`) before early stopping triggers.
        min_delta: Minimum decrease in validation loss to count as an
            improvement for early stopping / checkpointing "best" model.
        use_scheduler: If True, use `ReduceLROnPlateau` on validation loss.
        scheduler_factor: Factor to multiply the learning rate by on plateau.
        scheduler_patience: Epochs with no improvement before the
            scheduler reduces the learning rate.
        device: Torch device string (e.g. `"cuda"`, `"cpu"`). If None,
            automatically uses CUDA when available, else CPU.
    """

    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5
    min_delta: float = 1e-4
    use_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    device: Optional[str] = None


@dataclass
class TrainingHistory:
    """Per-epoch metrics recorded over the course of a training run."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List[float]]:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
        }


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Resolve a device string, auto-detecting CUDA when not specified.

    Args:
        device: Explicit device string, or None to auto-detect.

    Returns:
        A `torch.device`, using CUDA if available and not overridden.
    """
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """Trains a `BaseClassifier` with early stopping and an optional LR scheduler.

    Args:
        model: The classifier to train (moved to the resolved device).
        config: Training hyperparameters. Defaults to `TrainingConfig()`.
    """

    def __init__(self, model: BaseClassifier, config: Optional[TrainingConfig] = None) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        self.device = resolve_device(self.config.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
            )
            if self.config.use_scheduler
            else None
        )

        logger.info(
            "Trainer initialized on device=%s (epochs=%d, lr=%g, weight_decay=%g, "
            "patience=%d, use_scheduler=%s)",
            self.device,
            self.config.epochs,
            self.config.learning_rate,
            self.config.weight_decay,
            self.config.patience,
            self.config.use_scheduler,
        )

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch. Returns the mean training loss."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            self.optimizer.step()

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

        return total_loss / max(n_samples, 1)

    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Run one validation epoch. Returns `(mean_val_loss, val_accuracy)`."""
        self.model.eval()
        total_loss = 0.0
        n_correct = 0
        n_samples = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)

                batch_size = x_batch.size(0)
                total_loss += loss.item() * batch_size
                n_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                n_samples += batch_size

        mean_loss = total_loss / max(n_samples, 1)
        accuracy = n_correct / max(n_samples, 1)
        return mean_loss, accuracy

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Tuple[TrainingHistory, Dict[str, float]]:
        """Train with early stopping, restoring the best validation-loss weights.

        Args:
            train_loader: Training data.
            val_loader: Validation data, used for early stopping,
                scheduling, and reporting.

        Returns:
            A `(history, best_metrics)` tuple. `history` has per-epoch
            `train_loss`, `val_loss`, `val_acc` lists (one entry per
            epoch actually run, i.e. shorter than `config.epochs` if
            early stopping triggered). `best_metrics` is a dict with the
            best epoch's `epoch`, `val_loss`, and `val_acc`. The model's
            weights are left set to the best (lowest val_loss) epoch.
        """
        history = TrainingHistory()
        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_epoch = 0
        best_state = copy.deepcopy(self.model.state_dict())
        epochs_without_improvement = 0

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            history.val_acc.append(val_acc)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
                epoch,
                self.config.epochs,
                train_loss,
                val_loss,
                val_acc,
            )

            if val_loss < best_val_loss - self.config.min_delta:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    logger.info(
                        "Early stopping triggered at epoch %d (no improvement for %d epochs)",
                        epoch,
                        self.config.patience,
                    )
                    break

        self.model.load_state_dict(best_state)
        logger.info(
            "Training complete. Best epoch=%d, val_loss=%.4f, val_acc=%.4f",
            best_epoch,
            best_val_loss,
            best_val_acc,
        )

        best_metrics = {
            "epoch": best_epoch,
            "val_loss": best_val_loss,
            "val_acc": best_val_acc,
        }
        return history, best_metrics

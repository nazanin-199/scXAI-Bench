"""Save and load model checkpoints.

Checkpoints store everything needed to reconstruct the exact model
architecture (`model_name` + `model_kwargs`) alongside its trained
weights, plus optional training history / metadata. `load_checkpoint`
takes the model registry as a parameter (rather than importing
`MODEL_REGISTRY` from `models/__init__.py` directly) to avoid a
circular import, since `__init__.py` itself exposes these functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union

import torch

from scxai_bench.models.base import BaseClassifier
from scxai_bench.utils import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]


def save_checkpoint(
    path: PathLike,
    model: BaseClassifier,
    model_name: str,
    model_kwargs: Dict[str, Any],
    history: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a model checkpoint to disk.

    Args:
        path: Destination file path (e.g. `outputs/models/best_model.pt`).
            Parent directories are created automatically if missing.
        model: The trained model to save.
        model_name: Registry key identifying the model class (e.g.
            `"mlp"`), used by `load_checkpoint` to reconstruct it.
        model_kwargs: Constructor kwargs needed to rebuild this exact
            architecture (e.g. `model.get_config()` for `MLPClassifier`).
        history: Optional training history to embed in the checkpoint
            (e.g. from `Trainer.fit`'s `TrainingHistory.as_dict()`).
        extra: Optional additional metadata (e.g. label class names,
            gene names, random seed) useful for reproducibility and for
            Phase 4's explainers.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "model_state_dict": model.state_dict(),
        "history": history,
        "extra": extra or {},
    }
    torch.save(payload, path)
    logger.info("Saved checkpoint -> %s", path)


def load_checkpoint(
    path: PathLike,
    model_registry: Dict[str, Type[BaseClassifier]],
    device: Union[str, torch.device, None] = None,
) -> Tuple[BaseClassifier, Dict[str, Any]]:
    """Load a model checkpoint from disk.

    Args:
        path: Path to a checkpoint saved by `save_checkpoint`.
        model_registry: Maps model names (e.g. `"mlp"`) to their class,
            used to reconstruct the correct architecture. Pass
            `scxai_bench.models.MODEL_REGISTRY` for the built-in models.
        device: Device to load the model onto. Defaults to CPU.

    Returns:
        A `(model, payload)` tuple: the reconstructed model with trained
        weights loaded (in eval mode), and the full checkpoint payload
        dict (containing `history` and `extra` metadata).

    Raises:
        KeyError: If the checkpoint's `model_name` is not in `model_registry`.
    """
    device = torch.device(device) if device is not None else torch.device("cpu")
    payload = torch.load(Path(path), map_location=device, weights_only=False)

    model_name = payload["model_name"]
    if model_name not in model_registry:
        available = ", ".join(sorted(model_registry))
        raise KeyError(
            f"Unknown model '{model_name}' in checkpoint. Available: {available}"
        )

    model_cls = model_registry[model_name]
    model = model_cls(**payload["model_kwargs"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info("Loaded checkpoint from %s (model=%s)", path, model_name)
    return model, payload

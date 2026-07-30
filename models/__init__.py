"""Model definitions, training, inference, and checkpointing for scXAI-Bench.

Adding a new model in a future phase requires exactly one new step:
create a `BaseClassifier` subclass (see `mlp.py` for the pattern) and
add it to `MODEL_REGISTRY` below. `Trainer`, `Predictor`, and
`save_checkpoint`/`load_checkpoint` are all model-agnostic and need no
changes.
"""

from typing import Dict, Type

from scxai_bench.models.base import BaseClassifier
from scxai_bench.models.mlp import MLPClassifier
from scxai_bench.models.trainer import Trainer, TrainingConfig, TrainingHistory, resolve_device
from scxai_bench.models.predictor import Predictor
from scxai_bench.models.checkpoint import save_checkpoint, load_checkpoint

MODEL_REGISTRY: Dict[str, Type[BaseClassifier]] = {
    "mlp": MLPClassifier,
}


def get_model(name: str, **kwargs) -> BaseClassifier:
    """Instantiate a registered model by name.

    Args:
        name: Key into `MODEL_REGISTRY` (e.g. `"mlp"`).
        **kwargs: Forwarded to the model class constructor.

    Returns:
        An instantiated `BaseClassifier` subclass.

    Raises:
        KeyError: If `name` is not a registered model.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{name}'. Available models: {available}")
    return MODEL_REGISTRY[name](**kwargs)


__all__ = [
    "BaseClassifier",
    "MLPClassifier",
    "Trainer",
    "TrainingConfig",
    "TrainingHistory",
    "resolve_device",
    "Predictor",
    "save_checkpoint",
    "load_checkpoint",
    "MODEL_REGISTRY",
    "get_model",
]

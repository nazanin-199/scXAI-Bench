"""A small, lightweight MLP classifier.

The single model implementation for Phase 1-3. Intentionally simple:

    Input -> Linear(256) -> ReLU -> Dropout -> Linear(128) -> ReLU -> Linear(num_classes)

No CNNs, transformers, GNNs, or foundation models -- this is meant to
run comfortably on a single consumer GPU (or CPU only).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from scxai_bench.models.base import BaseClassifier


class MLPClassifier(BaseClassifier):
    """A small multi-layer perceptron classifier.

    Args:
        input_dim: Number of input features (e.g. number of HVGs).
        num_classes: Number of output classes.
        hidden_dims: Sizes of the hidden layers, in order. Defaults to
            `(256, 128)`, matching the Phase 1 reference architecture.
        dropout: Dropout probability applied after the first hidden
            layer only (matching the reference architecture: dropout
            regularizes the widest layer, not every layer).
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if num_classes <= 1:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer")

        self._input_dim = input_dim
        self._num_classes = num_classes
        self._hidden_dims: Tuple[int, ...] = tuple(hidden_dims)
        self._dropout = dropout

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            layers += [nn.Linear(prev_dim, hidden_dim), nn.ReLU()]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_config(self) -> dict:
        """Constructor kwargs needed to reconstruct this exact architecture.

        Used by `checkpoint.py` to rebuild the model before loading a
        saved `state_dict`.
        """
        return {
            "input_dim": self._input_dim,
            "num_classes": self._num_classes,
            "hidden_dims": self._hidden_dims,
            "dropout": self._dropout,
        }

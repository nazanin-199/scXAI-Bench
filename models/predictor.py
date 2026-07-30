"""Inference wrapper around a trained `BaseClassifier`.

`Predictor` is the seam Phase 4's explainers will plug into: SHAP wants
a plain `numpy -> numpy` prediction function, which `predict_proba`
provides directly. Captum's Integrated Gradients works straight off the
underlying `torch.nn.Module` (`predictor.model`) since it needs
differentiable tensor-in/tensor-out behavior that raw numpy functions
can't provide. Both explainers therefore consume this same trained
model with zero adapter code.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

from scxai_bench.models.base import BaseClassifier


class Predictor:
    """Wraps a trained classifier for batched inference.

    Args:
        model: A trained `BaseClassifier`. Not modified except moved to
            `device` and set to eval mode.
        device: Device to run inference on. If None, uses the device the
            model's parameters are already on.
    """

    def __init__(self, model: BaseClassifier, device: Union[str, torch.device, None] = None) -> None:
        self.model = model
        self.device = (
            torch.device(device) if device is not None else next(model.parameters()).device
        )
        self.model.to(self.device)
        self.model.eval()

    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        return x.to(self.device)

    def predict_proba(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Class probabilities for a batch of inputs.

        Args:
            x: Input features, shape `(n_samples, input_dim)`. NumPy
                array or torch Tensor.

        Returns:
            Probabilities as a NumPy array, shape `(n_samples, num_classes)`.
        """
        x_tensor = self._to_tensor(x)
        with torch.no_grad():
            probs = self.model.predict_proba(x_tensor)
        return probs.cpu().numpy()

    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predicted class indices for a batch of inputs.

        Args:
            x: Input features, shape `(n_samples, input_dim)`.

        Returns:
            Predicted class indices as a NumPy array, shape `(n_samples,)`.
        """
        return self.predict_proba(x).argmax(axis=1)

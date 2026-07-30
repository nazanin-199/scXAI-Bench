"""Unit tests for scxai_bench.models."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scxai_bench.models import (
    MODEL_REGISTRY,
    MLPClassifier,
    Predictor,
    Trainer,
    TrainingConfig,
    get_model,
    load_checkpoint,
    save_checkpoint,
)
from scxai_bench.models.base import BaseClassifier
from scxai_bench.models.trainer import resolve_device


def make_loaders(
    n_train: int = 64,
    n_val: int = 32,
    input_dim: int = 10,
    num_classes: int = 3,
    batch_size: int = 16,
    seed: int = 0,
):
    gen = torch.Generator().manual_seed(seed)
    X_train = torch.randn(n_train, input_dim, generator=gen)
    y_train = torch.randint(0, num_classes, (n_train,), generator=gen)
    X_val = torch.randn(n_val, input_dim, generator=gen)
    y_val = torch.randint(0, num_classes, (n_val,), generator=gen)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, X_val, y_val


class TestBaseClassifier:
    def test_cannot_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            BaseClassifier()  # type: ignore[abstract]


class TestMLPClassifier:
    def test_forward_shape(self) -> None:
        model = MLPClassifier(input_dim=10, num_classes=4)
        x = torch.randn(5, 10)
        logits = model(x)
        assert logits.shape == (5, 4)

    def test_predict_proba_sums_to_one(self) -> None:
        model = MLPClassifier(input_dim=10, num_classes=4)
        x = torch.randn(5, 10)
        probs = model.predict_proba(x)
        assert torch.allclose(probs.sum(dim=1), torch.ones(5), atol=1e-5)

    def test_predict_returns_valid_class_indices(self) -> None:
        model = MLPClassifier(input_dim=10, num_classes=4)
        x = torch.randn(5, 10)
        preds = model.predict(x)
        assert preds.shape == (5,)
        assert preds.min() >= 0 and preds.max() < 4

    def test_predict_proba_restores_training_mode(self) -> None:
        model = MLPClassifier(input_dim=10, num_classes=3)
        model.train()
        model.predict_proba(torch.randn(2, 10))
        assert model.training is True

        model.eval()
        model.predict_proba(torch.randn(2, 10))
        assert model.training is False

    def test_get_config_roundtrip(self) -> None:
        model = MLPClassifier(input_dim=15, num_classes=5, hidden_dims=(64, 32), dropout=0.1)
        config = model.get_config()
        rebuilt = MLPClassifier(**config)
        assert rebuilt.input_dim == 15
        assert rebuilt.num_classes == 5

    def test_invalid_input_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            MLPClassifier(input_dim=0, num_classes=3)

    def test_invalid_num_classes_raises(self) -> None:
        with pytest.raises(ValueError):
            MLPClassifier(input_dim=10, num_classes=1)

    def test_empty_hidden_dims_raises(self) -> None:
        with pytest.raises(ValueError):
            MLPClassifier(input_dim=10, num_classes=3, hidden_dims=())

    def test_reference_architecture_layer_sizes(self) -> None:
        model = MLPClassifier(input_dim=10, num_classes=3, hidden_dims=(256, 128))
        linear_layers = [m for m in model.network if isinstance(m, torch.nn.Linear)]
        assert [l.out_features for l in linear_layers] == [256, 128, 3]


class TestResolveDevice:
    def test_explicit_device(self) -> None:
        assert resolve_device("cpu") == torch.device("cpu")

    def test_auto_detect_returns_valid_device(self) -> None:
        device = resolve_device(None)
        assert device.type in ("cpu", "cuda")


class TestTrainer:
    def test_fit_runs_and_returns_history(self) -> None:
        train_loader, val_loader, _, _ = make_loaders()
        model = MLPClassifier(input_dim=10, num_classes=3)
        config = TrainingConfig(epochs=3, patience=10, device="cpu")
        trainer = Trainer(model, config)

        history, best_metrics = trainer.fit(train_loader, val_loader)

        assert len(history.train_loss) == 3
        assert len(history.val_loss) == 3
        assert len(history.val_acc) == 3
        assert 1 <= best_metrics["epoch"] <= 3
        assert best_metrics["val_loss"] >= 0

    def test_early_stopping_shortens_history(self) -> None:
        train_loader, val_loader, _, _ = make_loaders()
        model = MLPClassifier(input_dim=10, num_classes=3)
        # min_delta huge -> only epoch 1 (vs. the inf baseline) ever counts
        # as an improvement -> stops after `patience` further epochs.
        patience = 2
        config = TrainingConfig(epochs=50, patience=patience, min_delta=1e6, device="cpu")
        trainer = Trainer(model, config)

        history, _ = trainer.fit(train_loader, val_loader)

        assert len(history.train_loss) == patience + 1
        assert len(history.train_loss) < 50

    def test_history_as_dict(self) -> None:
        train_loader, val_loader, _, _ = make_loaders()
        model = MLPClassifier(input_dim=10, num_classes=3)
        trainer = Trainer(model, TrainingConfig(epochs=2, device="cpu"))
        history, _ = trainer.fit(train_loader, val_loader)
        d = history.as_dict()
        assert set(d.keys()) == {"train_loss", "val_loss", "val_acc"}

    def test_loss_decreases_or_stays_low_over_training(self) -> None:
        # Not a strict monotonicity guarantee, but on a simple separable
        # task the model should learn something over more epochs.
        torch.manual_seed(1)
        n = 200
        input_dim = 6
        X = torch.randn(n, input_dim)
        y = (X[:, 0] > 0).long() * 1  # simple linearly separable-ish signal
        y = torch.clamp(y, max=1)
        # pad to 2 classes minimum requirement already satisfied
        train_loader = DataLoader(TensorDataset(X[:160], y[:160]), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(X[160:], y[160:]), batch_size=32, shuffle=False)

        model = MLPClassifier(input_dim=input_dim, num_classes=2)
        trainer = Trainer(model, TrainingConfig(epochs=20, patience=20, device="cpu"))
        history, best_metrics = trainer.fit(train_loader, val_loader)

        assert history.train_loss[-1] <= history.train_loss[0]


class TestPredictor:
    def test_predict_proba_shape_and_range(self) -> None:
        model = MLPClassifier(input_dim=10, num_classes=3)
        predictor = Predictor(model, device="cpu")
        X = np.random.randn(7, 10).astype(np.float32)

        probs = predictor.predict_proba(X)

        assert probs.shape == (7, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_returns_numpy_indices(self) -> None:
        model = MLPClassifier(input_dim=10, num_classes=3)
        predictor = Predictor(model, device="cpu")
        X = np.random.randn(7, 10).astype(np.float32)

        preds = predictor.predict(X)

        assert isinstance(preds, np.ndarray)
        assert preds.shape == (7,)

    def test_accepts_torch_tensor_input(self) -> None:
        model = MLPClassifier(input_dim=10, num_classes=3)
        predictor = Predictor(model, device="cpu")
        X = torch.randn(4, 10)

        probs = predictor.predict_proba(X)

        assert probs.shape == (4, 3)


class TestCheckpoint:
    def test_save_and_load_roundtrip(self, tmp_path) -> None:
        model = MLPClassifier(input_dim=10, num_classes=3)
        path = tmp_path / "ckpt.pt"

        save_checkpoint(
            path,
            model,
            model_name="mlp",
            model_kwargs=model.get_config(),
            history={"train_loss": [1.0, 0.5]},
            extra={"note": "unit test"},
        )
        loaded_model, payload = load_checkpoint(path, MODEL_REGISTRY, device="cpu")

        assert loaded_model.get_config() == model.get_config()
        assert payload["history"] == {"train_loss": [1.0, 0.5]}
        assert payload["extra"] == {"note": "unit test"}

    def test_loaded_model_predictions_match_original(self, tmp_path) -> None:
        model = MLPClassifier(input_dim=10, num_classes=3)
        model.eval()
        X = torch.randn(5, 10)
        original_preds = model.predict(X)

        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, model_name="mlp", model_kwargs=model.get_config())
        loaded_model, _ = load_checkpoint(path, MODEL_REGISTRY, device="cpu")

        loaded_preds = loaded_model.predict(X)
        assert torch.equal(original_preds, loaded_preds)

    def test_creates_parent_dirs(self, tmp_path) -> None:
        model = MLPClassifier(input_dim=5, num_classes=2)
        path = tmp_path / "nested" / "dir" / "ckpt.pt"
        save_checkpoint(path, model, model_name="mlp", model_kwargs=model.get_config())
        assert path.exists()

    def test_unknown_model_name_raises(self, tmp_path) -> None:
        model = MLPClassifier(input_dim=5, num_classes=2)
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, model_name="not_a_real_model", model_kwargs=model.get_config())

        with pytest.raises(KeyError):
            load_checkpoint(path, MODEL_REGISTRY, device="cpu")


class TestRegistry:
    def test_mlp_registered(self) -> None:
        assert "mlp" in MODEL_REGISTRY

    def test_get_model_returns_instance(self) -> None:
        model = get_model("mlp", input_dim=10, num_classes=3)
        assert isinstance(model, MLPClassifier)

    def test_get_model_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError):
            get_model("not_a_real_model", input_dim=10, num_classes=3)

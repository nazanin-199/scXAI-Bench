"""Unit tests for scxai_bench.explainers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scxai_bench.explainers import (
    EXPLAINER_REGISTRY,
    BaseExplainer,
    IntegratedGradientsExplainer,
    SHAPExplainer,
    get_explainer,
)
from scxai_bench.models import MLPClassifier


@pytest.fixture
def trained_like_model() -> MLPClassifier:
    torch.manual_seed(0)
    model = MLPClassifier(input_dim=12, num_classes=3)
    model.eval()
    return model


@pytest.fixture
def sample_X() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((10, 12)).astype(np.float32)


class TestBaseExplainer:
    def test_cannot_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            BaseExplainer()  # type: ignore[abstract]


class TestSHAPExplainer:
    def test_name(self) -> None:
        assert SHAPExplainer().name == "shap"

    def test_output_shape_matches_input(self, trained_like_model, sample_X) -> None:
        explainer = SHAPExplainer(n_background=8)
        attributions = explainer.explain(trained_like_model, sample_X)
        assert attributions.shape == sample_X.shape

    def test_no_nans_or_infs(self, trained_like_model, sample_X) -> None:
        explainer = SHAPExplainer(n_background=8)
        attributions = explainer.explain(trained_like_model, sample_X)
        assert np.isfinite(attributions).all()

    def test_explicit_background_accepted(self, trained_like_model, sample_X) -> None:
        rng = np.random.default_rng(1)
        background = rng.standard_normal((5, 12)).astype(np.float32)
        explainer = SHAPExplainer()
        attributions = explainer.explain(trained_like_model, sample_X, background=background)
        assert attributions.shape == sample_X.shape

    def test_preserves_model_training_mode(self, trained_like_model, sample_X) -> None:
        trained_like_model.train()
        SHAPExplainer(n_background=8).explain(trained_like_model, sample_X)
        assert trained_like_model.training is True

        trained_like_model.eval()
        SHAPExplainer(n_background=8).explain(trained_like_model, sample_X)
        assert trained_like_model.training is False

    def test_background_larger_than_x_is_clipped(self, trained_like_model, sample_X) -> None:
        # n_background > n_samples in X should not raise.
        explainer = SHAPExplainer(n_background=1000)
        attributions = explainer.explain(trained_like_model, sample_X)
        assert attributions.shape == sample_X.shape


class TestIntegratedGradientsExplainer:
    def test_name(self) -> None:
        assert IntegratedGradientsExplainer().name == "integrated_gradients"

    def test_output_shape_matches_input(self, trained_like_model, sample_X) -> None:
        explainer = IntegratedGradientsExplainer(n_steps=20)
        attributions = explainer.explain(trained_like_model, sample_X)
        assert attributions.shape == sample_X.shape

    def test_no_nans_or_infs(self, trained_like_model, sample_X) -> None:
        explainer = IntegratedGradientsExplainer(n_steps=20)
        attributions = explainer.explain(trained_like_model, sample_X)
        assert np.isfinite(attributions).all()

    def test_zero_input_equal_to_zero_baseline_gives_near_zero_attributions(
        self, trained_like_model
    ) -> None:
        # If X == baseline (both zero), the straight-line path has zero
        # length, so attributions should be ~0 everywhere.
        X_zero = np.zeros((4, 12), dtype=np.float32)
        explainer = IntegratedGradientsExplainer(n_steps=20)
        attributions = explainer.explain(trained_like_model, X_zero)
        np.testing.assert_allclose(attributions, 0.0, atol=1e-5)

    def test_explicit_single_row_background_broadcasts(self, trained_like_model, sample_X) -> None:
        background = np.ones((1, 12), dtype=np.float32) * 0.5
        explainer = IntegratedGradientsExplainer(n_steps=20)
        attributions = explainer.explain(trained_like_model, sample_X, background=background)
        assert attributions.shape == sample_X.shape

    def test_multi_row_background_uses_mean(self, trained_like_model, sample_X) -> None:
        rng = np.random.default_rng(2)
        background = rng.standard_normal((7, 12)).astype(np.float32)
        explainer = IntegratedGradientsExplainer(n_steps=20)
        attributions = explainer.explain(trained_like_model, sample_X, background=background)
        assert attributions.shape == sample_X.shape

    def test_preserves_model_training_mode(self, trained_like_model, sample_X) -> None:
        trained_like_model.train()
        IntegratedGradientsExplainer(n_steps=10).explain(trained_like_model, sample_X)
        assert trained_like_model.training is True


class TestRegistry:
    def test_both_explainers_registered(self) -> None:
        assert set(EXPLAINER_REGISTRY.keys()) == {"shap", "integrated_gradients"}

    def test_get_explainer_returns_instance(self) -> None:
        explainer = get_explainer("shap", n_background=10)
        assert isinstance(explainer, SHAPExplainer)

    def test_get_explainer_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError):
            get_explainer("not_a_real_explainer")

    def test_both_explainers_agree_on_output_shape(self, trained_like_model, sample_X) -> None:
        shap_attr = get_explainer("shap", n_background=8).explain(trained_like_model, sample_X)
        ig_attr = get_explainer("integrated_gradients", n_steps=20).explain(
            trained_like_model, sample_X
        )
        assert shap_attr.shape == ig_attr.shape == sample_X.shape

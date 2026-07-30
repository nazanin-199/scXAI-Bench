"""Unit tests for scxai_bench.metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scxai_bench.metrics import (
    METRIC_REGISTRY,
    BaseMetric,
    FaithfulnessMetric,
    InfidelityMetric,
    MetricResult,
    SensitivityMetric,
    TopKRankingConsistencyMetric,
    get_metric,
)
from scxai_bench.metrics.base import get_target_class_probabilities, resolve_target_classes
from scxai_bench.models import MLPClassifier


@pytest.fixture
def model() -> MLPClassifier:
    torch.manual_seed(0)
    m = MLPClassifier(input_dim=10, num_classes=3)
    m.eval()
    return m


@pytest.fixture
def X() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((12, 10)).astype(np.float32)


@pytest.fixture
def explanations(model, X) -> np.ndarray:
    # Simple stand-in "explanation": input gradient magnitude proxy via
    # random values correlated with X itself (deterministic, no need
    # for a real explainer here -- metrics must not care).
    rng = np.random.default_rng(1)
    return (X * 0.1 + rng.standard_normal(X.shape).astype(np.float32) * 0.01)


def make_explain_fn(model):
    def explain_fn(x: np.ndarray) -> np.ndarray:
        # A deterministic, cheap stand-in explainer for testing metrics
        # that need to regenerate explanations: gradient of the
        # predicted class logit w.r.t. input.
        x_t = torch.from_numpy(x).float().requires_grad_(True)
        logits = model(x_t)
        preds = logits.argmax(dim=1)
        selected = logits.gather(1, preds.unsqueeze(1)).sum()
        grads = torch.autograd.grad(selected, x_t)[0]
        return grads.detach().numpy()

    return explain_fn


class TestHelperFunctions:
    def test_resolve_target_classes_uses_labels_if_given(self, model, X) -> None:
        labels = np.array([0, 1, 2] * 4)
        result = resolve_target_classes(model, X, labels)
        np.testing.assert_array_equal(result, labels)

    def test_resolve_target_classes_falls_back_to_predictions(self, model, X) -> None:
        result = resolve_target_classes(model, X, None)
        x_t = torch.from_numpy(X).float()
        expected = model.predict(x_t).numpy()
        np.testing.assert_array_equal(result, expected)

    def test_get_target_class_probabilities_shape_and_range(self, model, X) -> None:
        target_classes = resolve_target_classes(model, X, None)
        probs = get_target_class_probabilities(model, X, target_classes)
        assert probs.shape == (X.shape[0],)
        assert np.all((probs >= 0) & (probs <= 1))


class TestMetricResult:
    def test_as_dict(self) -> None:
        result = MetricResult(metric_name="foo", score=0.5, runtime=0.1, metadata={"a": 1})
        d = result.as_dict()
        assert d == {"metric_name": "foo", "score": 0.5, "runtime": 0.1, "metadata": {"a": 1}}


class TestBaseMetric:
    def test_cannot_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            BaseMetric()  # type: ignore[abstract]

    def test_evaluate_measures_runtime(self, model, X, explanations) -> None:
        metric = FaithfulnessMetric(n_trials=2)
        result = metric.evaluate(model, X, explanations)
        assert result.runtime >= 0
        assert result.metric_name == "faithfulness"


class TestFaithfulnessMetric:
    def test_score_in_valid_correlation_range(self, model, X, explanations) -> None:
        metric = FaithfulnessMetric(n_trials=5, mask_fraction=0.3, seed=1)
        result = metric.evaluate(model, X, explanations)
        assert -1.0 <= result.score <= 1.0

    def test_metadata_fields(self, model, X, explanations) -> None:
        metric = FaithfulnessMetric(n_trials=3, mask_fraction=0.2)
        result = metric.evaluate(model, X, explanations)
        assert result.metadata["n_trials"] == 3
        assert result.metadata["mask_fraction"] == 0.2

    def test_deterministic_with_fixed_seed(self, model, X, explanations) -> None:
        metric_a = FaithfulnessMetric(n_trials=4, seed=7)
        metric_b = FaithfulnessMetric(n_trials=4, seed=7)
        result_a = metric_a.evaluate(model, X, explanations)
        result_b = metric_b.evaluate(model, X, explanations)
        assert result_a.score == pytest.approx(result_b.score)

    def test_correct_signed_explanation_scores_higher_than_negated_one(self, model, X) -> None:
        # A deliberately wrong-signed explanation (attribution sign
        # flipped) must score lower than the correctly-signed one,
        # regardless of whether the underlying model happens to be
        # well-trained -- this isolates the metric's own correctness
        # from the model's training quality.
        explain_fn = make_explain_fn(model)
        correct_explanations = explain_fn(X)
        negated_explanations = -correct_explanations

        metric = FaithfulnessMetric(n_trials=20, mask_fraction=0.3, seed=2)
        correct_score = metric.evaluate(model, X, correct_explanations).score
        negated_score = metric.evaluate(model, X, negated_explanations).score

        assert correct_score > negated_score


class TestInfidelityMetric:
    def test_score_is_non_negative(self, model, X, explanations) -> None:
        metric = InfidelityMetric(n_perturbations=5)
        result = metric.evaluate(model, X, explanations)
        assert result.score >= 0

    def test_correct_signed_explanation_has_lower_infidelity_than_negated(self, model, X) -> None:
        # Negating a gradient-based explanation flips the sign of the
        # predicted output change (I . phi) while the actual output
        # change is unchanged -- infidelity must be higher for the
        # negated (wrong-signed) explanation, regardless of whether the
        # underlying model happens to be well-trained.
        explain_fn = make_explain_fn(model)
        correct_explanations = explain_fn(X)
        negated_explanations = -correct_explanations

        metric = InfidelityMetric(n_perturbations=30, noise_scale=0.05, seed=3)
        correct_infidelity = metric.evaluate(model, X, correct_explanations).score
        negated_infidelity = metric.evaluate(model, X, negated_explanations).score

        assert correct_infidelity <= negated_infidelity


class TestSensitivityMetric:
    def test_requires_explain_fn(self, model, X, explanations) -> None:
        metric = SensitivityMetric()
        with pytest.raises(ValueError):
            metric.evaluate(model, X, explanations, explain_fn=None)

    def test_score_is_non_negative(self, model, X, explanations) -> None:
        metric = SensitivityMetric(n_perturbations=3)
        explain_fn = make_explain_fn(model)
        result = metric.evaluate(model, X, explanations, explain_fn=explain_fn)
        assert result.score >= 0

    def test_identical_explain_fn_gives_near_zero_if_deterministic(self, model, X) -> None:
        explain_fn = make_explain_fn(model)
        explanations = explain_fn(X)
        metric = SensitivityMetric(n_perturbations=3, noise_scale=1e-6, seed=5)
        result = metric.evaluate(model, X, explanations, explain_fn=explain_fn)
        # Tiny perturbation -> explanation should barely change.
        assert result.score < 0.5


class TestTopKRankingConsistencyMetric:
    def test_requires_explain_fn(self, model, X, explanations) -> None:
        metric = TopKRankingConsistencyMetric()
        with pytest.raises(ValueError):
            metric.evaluate(model, X, explanations, explain_fn=None)

    def test_score_in_unit_interval(self, model, X, explanations) -> None:
        metric = TopKRankingConsistencyMetric(k=0.3, n_perturbations=3)
        explain_fn = make_explain_fn(model)
        result = metric.evaluate(model, X, explanations, explain_fn=explain_fn)
        assert 0.0 <= result.score <= 1.0

    def test_absolute_k_resolves_correctly(self, model, X, explanations) -> None:
        metric = TopKRankingConsistencyMetric(k=3, n_perturbations=2)
        explain_fn = make_explain_fn(model)
        result = metric.evaluate(model, X, explanations, explain_fn=explain_fn)
        assert result.metadata["k"] == 3

    def test_fractional_k_resolves_correctly(self, model, X, explanations) -> None:
        metric = TopKRankingConsistencyMetric(k=0.5, n_perturbations=2)  # 50% of 10 features
        explain_fn = make_explain_fn(model)
        result = metric.evaluate(model, X, explanations, explain_fn=explain_fn)
        assert result.metadata["k"] == 5

    def test_identical_explanations_give_perfect_consistency(self, model, X) -> None:
        # A constant explain_fn (always returns the same attributions
        # regardless of input) must score perfect consistency (1.0).
        fixed_explanations = np.random.default_rng(0).standard_normal(X.shape).astype(np.float32)

        def constant_explain_fn(x):
            return fixed_explanations[: x.shape[0]]

        metric = TopKRankingConsistencyMetric(k=3, n_perturbations=3)
        result = metric.evaluate(model, X, fixed_explanations, explain_fn=constant_explain_fn)
        assert result.score == pytest.approx(1.0)


class TestRegistry:
    def test_all_four_metrics_registered(self) -> None:
        assert set(METRIC_REGISTRY.keys()) == {
            "faithfulness",
            "infidelity",
            "sensitivity",
            "topk_ranking_consistency",
        }

    def test_get_metric_returns_instance(self) -> None:
        metric = get_metric("faithfulness", n_trials=5)
        assert isinstance(metric, FaithfulnessMetric)

    def test_get_metric_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError):
            get_metric("not_a_real_metric")

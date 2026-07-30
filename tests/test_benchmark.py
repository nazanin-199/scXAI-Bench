"""Unit tests for scxai_bench.benchmark."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from scxai_bench.benchmark import (
    BenchmarkResult,
    BenchmarkRunner,
    build_results_table,
    generate_report,
    save_csv,
    save_json_report,
    save_markdown_report,
)
from scxai_bench.explainers import get_explainer
from scxai_bench.metrics import get_metric
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
def background() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.standard_normal((6, 10)).astype(np.float32)


@pytest.fixture
def small_explainers():
    return {
        "shap": get_explainer("shap", n_background=6),
        "integrated_gradients": get_explainer("integrated_gradients", n_steps=10),
    }


@pytest.fixture
def small_metrics():
    return [
        get_metric("faithfulness", n_trials=2, mask_fraction=0.3),
        get_metric("infidelity", n_perturbations=2),
        get_metric("sensitivity", n_perturbations=2),
        get_metric("topk_ranking_consistency", k=3, n_perturbations=2),
    ]


@pytest.fixture
def benchmark_result(model, X, background, small_explainers, small_metrics) -> BenchmarkResult:
    runner = BenchmarkRunner(model, small_explainers, small_metrics, background=background)
    return runner.run(X, model_name="mlp", dataset_name="test_dataset")


class TestBenchmarkRunner:
    def test_result_covers_every_explainer(self, benchmark_result, small_explainers) -> None:
        names = {r.explainer_name for r in benchmark_result.explainer_results}
        assert names == set(small_explainers.keys())

    def test_result_covers_every_metric_per_explainer(
        self, benchmark_result, small_metrics
    ) -> None:
        expected_metric_names = {m.name for m in small_metrics}
        for explainer_result in benchmark_result.explainer_results:
            actual_names = {mr.metric_name for mr in explainer_result.metric_results}
            assert actual_names == expected_metric_names

    def test_attributions_shape(self, benchmark_result, X) -> None:
        for explainer_result in benchmark_result.explainer_results:
            assert explainer_result.attributions.shape == X.shape

    def test_metadata_fields(self, benchmark_result) -> None:
        assert benchmark_result.n_samples == 12
        assert benchmark_result.n_features == 10
        assert benchmark_result.model_name == "mlp"
        assert benchmark_result.dataset_name == "test_dataset"

    def test_get_metric_score_lookup(self, benchmark_result) -> None:
        score = benchmark_result.get_metric_score("shap", "faithfulness")
        assert score is not None
        assert isinstance(score, float)

    def test_get_metric_score_unknown_pair_returns_none(self, benchmark_result) -> None:
        assert benchmark_result.get_metric_score("not_an_explainer", "faithfulness") is None
        assert benchmark_result.get_metric_score("shap", "not_a_metric") is None

    def test_works_with_single_explainer_and_metric(self, model, X, background) -> None:
        explainers = {"integrated_gradients": get_explainer("integrated_gradients", n_steps=5)}
        metrics = [get_metric("infidelity", n_perturbations=2)]
        runner = BenchmarkRunner(model, explainers, metrics, background=background)
        result = runner.run(X)
        assert len(result.explainer_results) == 1
        assert len(result.explainer_results[0].metric_results) == 1

    def test_works_without_background(self, model, X) -> None:
        explainers = {"integrated_gradients": get_explainer("integrated_gradients", n_steps=5)}
        metrics = [get_metric("infidelity", n_perturbations=2)]
        runner = BenchmarkRunner(model, explainers, metrics, background=None)
        result = runner.run(X)
        assert len(result.explainer_results) == 1


class TestBuildResultsTable:
    def test_returns_dataframe_with_expected_columns(self, benchmark_result) -> None:
        df = build_results_table(benchmark_result)
        assert isinstance(df, pd.DataFrame)
        assert {"explainer", "metric", "score", "runtime_seconds"}.issubset(df.columns)

    def test_row_count_matches_explainers_times_metrics(
        self, benchmark_result, small_explainers, small_metrics
    ) -> None:
        df = build_results_table(benchmark_result)
        assert len(df) == len(small_explainers) * len(small_metrics)


class TestReportGeneration:
    def test_save_csv_creates_file(self, benchmark_result, tmp_path) -> None:
        path = tmp_path / "results.csv"
        df = save_csv(benchmark_result, path)
        assert path.exists()
        reloaded = pd.read_csv(path)
        assert len(reloaded) == len(df)

    def test_save_json_report_creates_valid_json(self, benchmark_result, tmp_path) -> None:
        path = tmp_path / "results.json"
        save_json_report(benchmark_result, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["model_name"] == "mlp"
        assert "shap" in data["explainers"]
        assert "integrated_gradients" in data["explainers"]

    def test_save_markdown_report_creates_file_with_table(self, benchmark_result, tmp_path) -> None:
        path = tmp_path / "report.md"
        save_markdown_report(benchmark_result, path)
        assert path.exists()
        content = path.read_text()
        assert "# scXAI-Bench Benchmark Report" in content
        assert "faithfulness" in content
        assert "|" in content  # has a markdown table

    def test_generate_report_creates_all_three_files(self, benchmark_result, tmp_path) -> None:
        generate_report(benchmark_result, tmp_path)
        assert (tmp_path / "benchmark_results.csv").exists()
        assert (tmp_path / "benchmark_results.json").exists()
        assert (tmp_path / "benchmark_report.md").exists()

    def test_generate_report_creates_parent_dirs(self, benchmark_result, tmp_path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        generate_report(benchmark_result, nested)
        assert (nested / "benchmark_results.csv").exists()

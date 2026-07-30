"""Unit tests for scxai_bench.datasets.

These tests never touch the network: they exercise the pipeline entirely
against `load_synthetic_pbmc3k_like`, which is fast, deterministic, and
structurally representative of real PBMC3k data (sparse non-negative
integer counts, gene_ids in .var).
"""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from scxai_bench.datasets import (
    DATASET_REGISTRY,
    DatasetDownloadError,
    PBMC3kDataset,
    filter_cells_genes,
    get_dataset,
    load_synthetic_pbmc3k_like,
    normalize_and_log,
    preprocess_pipeline,
    select_highly_variable_genes,
    train_test_split_adata,
)
from scxai_bench.datasets.base import BaseDataset


@pytest.fixture
def synthetic_adata() -> AnnData:
    return load_synthetic_pbmc3k_like(n_cells=200, n_genes=300, n_cell_types=3, seed=0)


class TestBaseDataset:
    def test_cannot_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            BaseDataset()  # type: ignore[abstract]

    def test_default_label_key_is_none(self) -> None:
        class DummyDataset(BaseDataset):
            @property
            def name(self) -> str:
                return "dummy"

            def load(self) -> AnnData:
                return load_synthetic_pbmc3k_like(n_cells=5, n_genes=5)

        assert DummyDataset().label_key is None


class TestPBMC3kDataset:
    def test_name(self) -> None:
        assert PBMC3kDataset().name == "pbmc3k"

    def test_label_key_none_before_load_regardless_of_flag(self) -> None:
        # label_key must reflect actual loaded data, not the requested
        # flag -- before load() is called, that's unknown, so None.
        assert PBMC3kDataset(use_offline_fallback=True).label_key is None
        assert PBMC3kDataset(use_offline_fallback=False).label_key is None

    def test_load_without_fallback_raises_on_no_network(self) -> None:
        # In this test environment the real download host is
        # unreachable, so this should raise our clear error type.
        with pytest.raises(DatasetDownloadError):
            PBMC3kDataset(use_offline_fallback=False).load()

    def test_load_with_fallback_succeeds(self) -> None:
        adata = PBMC3kDataset(use_offline_fallback=True).load()
        assert adata.n_obs > 0
        assert adata.n_vars > 0
        assert adata.uns.get("synthetic") is True

    def test_label_key_after_synthetic_load(self) -> None:
        ds = PBMC3kDataset(use_offline_fallback=True)
        ds.load()
        assert ds.label_key == "cell_type"

    def test_label_key_reflects_actual_data_not_requested_flag(self) -> None:
        # Regression test for the real-world bug: on a machine with
        # working internet, use_offline_fallback=True is requested but
        # the REAL download succeeds (no fallback is actually used).
        # label_key must reflect that reality (None), not blindly report
        # "cell_type" just because the flag was set to True.
        ds = PBMC3kDataset(use_offline_fallback=True)

        # Simulate a successful real (non-synthetic) load without
        # touching the network: directly exercise the state transition
        # load() is responsible for.
        ds._last_load_was_synthetic = False
        assert ds.label_key is None

        ds._last_load_was_synthetic = True
        assert ds.label_key == "cell_type"


class TestSyntheticLoader:
    def test_shape(self, synthetic_adata: AnnData) -> None:
        assert synthetic_adata.shape == (200, 300)

    def test_reproducible_with_same_seed(self) -> None:
        a = load_synthetic_pbmc3k_like(n_cells=50, n_genes=50, seed=123)
        b = load_synthetic_pbmc3k_like(n_cells=50, n_genes=50, seed=123)
        np.testing.assert_array_equal(a.X.toarray(), b.X.toarray())

    def test_non_negative_counts(self, synthetic_adata: AnnData) -> None:
        assert synthetic_adata.X.min() >= 0

    def test_has_gene_ids(self, synthetic_adata: AnnData) -> None:
        assert "gene_ids" in synthetic_adata.var.columns

    def test_has_cell_type_labels(self, synthetic_adata: AnnData) -> None:
        assert "cell_type" in synthetic_adata.obs.columns
        assert synthetic_adata.obs["cell_type"].nunique() == 3


class TestPreprocessing:
    def test_filter_cells_genes_reduces_or_maintains_shape(
        self, synthetic_adata: AnnData
    ) -> None:
        n_obs_before, n_vars_before = synthetic_adata.shape
        filtered = filter_cells_genes(synthetic_adata, min_genes=1, min_cells=1)
        assert filtered.n_obs <= n_obs_before
        assert filtered.n_vars <= n_vars_before

    def test_filter_none_is_noop(self, synthetic_adata: AnnData) -> None:
        shape_before = synthetic_adata.shape
        filtered = filter_cells_genes(synthetic_adata, min_genes=None, min_cells=None)
        assert filtered.shape == shape_before

    def test_normalize_and_log_changes_values(self, synthetic_adata: AnnData) -> None:
        original_sum = float(synthetic_adata.X.sum())
        normalized = normalize_and_log(synthetic_adata, target_sum=1e4)
        assert float(normalized.X.sum()) != original_sum
        # log1p should make max value much smaller than raw counts could be
        assert normalized.X.max() < 20

    def test_select_hvg_respects_n_top_genes(self, synthetic_adata: AnnData) -> None:
        normalize_and_log(synthetic_adata)
        result = select_highly_variable_genes(synthetic_adata, n_top_genes=50, subset=True)
        assert result.n_vars == 50

    def test_select_hvg_clips_to_available_genes(self, synthetic_adata: AnnData) -> None:
        # Requesting more HVGs than genes exist should not error.
        normalize_and_log(synthetic_adata)
        result = select_highly_variable_genes(synthetic_adata, n_top_genes=10_000, subset=True)
        assert result.n_vars == 300

    def test_full_pipeline_runs_end_to_end(self, synthetic_adata: AnnData) -> None:
        result = preprocess_pipeline(
            synthetic_adata, min_genes=None, min_cells=1, n_top_genes=64
        )
        assert result.n_vars == 64
        assert result.n_obs > 0


class TestSplit:
    def test_split_sizes(self, synthetic_adata: AnnData) -> None:
        train, test = train_test_split_adata(synthetic_adata, test_size=0.25, seed=42)
        assert train.n_obs + test.n_obs == synthetic_adata.n_obs
        assert test.n_obs == round(0.25 * synthetic_adata.n_obs)

    def test_split_is_reproducible(self, synthetic_adata: AnnData) -> None:
        train1, test1 = train_test_split_adata(synthetic_adata, test_size=0.2, seed=7)
        train2, test2 = train_test_split_adata(synthetic_adata, test_size=0.2, seed=7)
        assert list(train1.obs_names) == list(train2.obs_names)
        assert list(test1.obs_names) == list(test2.obs_names)

    def test_split_no_overlap(self, synthetic_adata: AnnData) -> None:
        train, test = train_test_split_adata(synthetic_adata, test_size=0.2, seed=1)
        assert set(train.obs_names).isdisjoint(set(test.obs_names))

    def test_stratified_split_preserves_class_presence(
        self, synthetic_adata: AnnData
    ) -> None:
        train, test = train_test_split_adata(
            synthetic_adata, test_size=0.3, seed=1, stratify_key="cell_type"
        )
        assert set(train.obs["cell_type"].unique()) == set(
            synthetic_adata.obs["cell_type"].unique()
        )

    def test_missing_stratify_key_falls_back_gracefully(
        self, synthetic_adata: AnnData
    ) -> None:
        # Regression test for the reported KeyError: requesting a
        # stratify_key column that doesn't exist in adata.obs must not
        # raise -- it should fall back to a plain random split.
        del synthetic_adata.obs["cell_type"]  # simulate a labelless dataset
        train, test = train_test_split_adata(
            synthetic_adata, test_size=0.2, seed=42, stratify_key="cell_type"
        )
        assert train.n_obs + test.n_obs == synthetic_adata.n_obs
        assert test.n_obs == round(0.2 * synthetic_adata.n_obs)


class TestRegistry:
    def test_pbmc3k_registered(self) -> None:
        assert "pbmc3k" in DATASET_REGISTRY

    def test_get_dataset_returns_instance(self) -> None:
        ds = get_dataset("pbmc3k", use_offline_fallback=True)
        assert isinstance(ds, PBMC3kDataset)
        assert ds.name == "pbmc3k"

    def test_get_dataset_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError):
            get_dataset("not_a_real_dataset")

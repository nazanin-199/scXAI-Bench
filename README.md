# scXAI-Bench

A lightweight benchmark framework for explainability evaluation in
single-cell machine learning.

**Status: Phase 1, in progress.** This is a minimal, CPU/single-GPU
friendly benchmark: one classifier, two explainers (SHAP, Captum
Integrated Gradients), and a handful of generic XAI metrics
(faithfulness, infidelity, sensitivity, feature ranking correlation).

## Implemented so far

- [x] `scxai_bench/utils/` — reproducibility (seeding), array/JSON I/O,
      logging
- [x] `scxai_bench/datasets/` — PBMC3k loading, preprocessing
      (filter/normalize/log1p/HVG), and train/test splitting
- [x] `scxai_bench/models/` — MLP classifier, training loop (early
      stopping + LR scheduler), inference, checkpointing
- [x] `scxai_bench/explainers/` — SHAP (GradientExplainer) and Captum
      Integrated Gradients, behind one shared interface
- [x] `scxai_bench/metrics/` — Faithfulness, Infidelity, Sensitivity,
      Top-K Ranking Consistency
- [x] `scxai_bench/benchmark/` — BenchmarkRunner + CSV/JSON/Markdown
      report generation
- [ ] `scxai_bench/visualization/`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest -v
```

## Datasets

`scxai_bench.datasets.PBMC3kDataset` loads the real 3k PBMCs dataset
from 10x Genomics via `scanpy.datasets.pbmc3k()` (~5.9 MB, downloaded
once and cached locally). Raw PBMC3k ships with no cell-type labels.

```bash
python examples/demo_dataset.py
```

If no network access is available, the demo automatically falls back
to a small synthetic, PBMC3k-*shaped* dataset so the pipeline can still
be exercised end-to-end offline (this is always logged loudly and
recorded in `adata.uns["synthetic"]` — never silent). To require the
real dataset and fail instead of falling back, set `USE_REAL_ONLY = True`
at the top of `examples/demo_dataset.py`, or use
`PBMC3kDataset(use_offline_fallback=False)` (the default) directly.

## Models

`scxai_bench.models.MLPClassifier` is a small MLP
(`Linear(256) -> ReLU -> Dropout -> Linear(128) -> ReLU -> Linear(num_classes)`),
trained by `Trainer` (early stopping + optional `ReduceLROnPlateau`
scheduler, auto CPU/GPU device selection).

```bash
python examples/train_model.py
```

**A note on labels:** real raw PBMC3k has no cell-type annotations
(see Datasets above). When no real label column exists,
`examples/train_model.py` derives simple unsupervised pseudo-labels via
KMeans clustering on the processed expression matrix, purely so the
classifier has classes to learn for this demo — this is always logged
clearly and is not a claim about true cell identity. If real labels are
available (e.g. via a future labeled dataset, or the offline synthetic
fallback's `cell_type` column), those are used directly instead.

Produces `outputs/models/best_model.pt` (checkpoint), `training_history.json`,
and `loss_curve.png`.

## Explainers

Both `scxai_bench.explainers.SHAPExplainer` (via `shap.GradientExplainer`,
chosen for speed on a differentiable PyTorch model) and
`IntegratedGradientsExplainer` (via Captum) share the same interface:
given a trained model and a batch of samples, return per-sample,
per-feature attributions for each sample's *predicted* class.

```bash
python examples/train_model.py     # if you haven't already
python examples/explain_model.py
```

Explains a subset of test cells (30 by default), saves the raw
attribution arrays plus metadata to `outputs/explanations/`, and plots
each explainer's top genes plus a SHAP-vs-IG agreement scatter.

## Metrics & Benchmark

`scxai_bench.metrics` implements four explainer-agnostic XAI metrics --
Faithfulness, Infidelity, Sensitivity, and Top-K Ranking Consistency --
each a `BaseMetric` subclass returning a `MetricResult` (score, runtime,
metadata). Sensitivity and Top-K Ranking Consistency need to regenerate
attributions for perturbed inputs; they receive a generic `explain_fn`
callable rather than an explainer object, so no metric ever knows which
explainer produced the attributions it's evaluating.

`scxai_bench.benchmark.BenchmarkRunner` runs any set of explainers
against any set of metrics automatically and returns a `BenchmarkResult`;
`generate_report` turns that into CSV, JSON, and Markdown summaries.

```bash
python examples/train_model.py     # if you haven't already
python examples/run_benchmark.py
```

Benchmarks SHAP and Integrated Gradients (the two explainers
implemented in Phase 4) across all four metrics, prints a comparison
table, and saves `benchmark_results.csv`, `benchmark_results.json`, and
`benchmark_report.md` to `outputs/benchmark/`.

Adding a new explainer or metric never requires touching
`benchmark_runner.py`: both are consumed purely through their abstract
base classes (`BaseExplainer`, `BaseMetric`).


## Project structure

```
scxai_bench/
    datasets/         # dataset loaders, preprocessing, splitting (implemented)
        base.py          # BaseDataset (ABC)
        pbmc3k.py         # PBMC3kDataset(BaseDataset)
        loader.py          # download + synthetic offline-fallback loaders
        preprocess.py       # filter / normalize / log1p / HVG selection
        split.py              # train/test split
    models/           # classifiers, training, inference, checkpointing (implemented)
        base.py          # BaseClassifier (ABC)
        mlp.py             # MLPClassifier(BaseClassifier)
        trainer.py           # TrainingConfig, Trainer (early stopping + LR scheduler)
        predictor.py           # Predictor: inference wrapper (reused by Phase 4 explainers)
        checkpoint.py            # save_checkpoint / load_checkpoint
    explainers/       # explanation methods (BaseExplainer + SHAP/Captum, implemented)
        base.py                  # BaseExplainer (ABC)
        shap_explainer.py          # SHAPExplainer (shap.GradientExplainer)
        captum_ig_explainer.py       # IntegratedGradientsExplainer (Captum)
    metrics/          # XAI evaluation metrics (BaseMetric + concrete metrics, implemented)
        base.py                  # BaseMetric (ABC) + MetricResult
        faithfulness.py            # attribution-mass vs. output-drop correlation
        infidelity.py                # MSE vs. Yeh et al. (In)fidelity definition
        sensitivity.py                 # explanation stability under perturbation
        ranking.py                       # top-K feature overlap stability
        factory.py                         # METRIC_REGISTRY + get_metric()
    benchmark/        # orchestration: runs the full pipeline, builds reports (implemented)
        benchmark_runner.py       # BenchmarkRunner: any explainers x any metrics
        benchmark_report.py         # CSV / JSON / Markdown report generation
    visualization/    # plotting utilities
    utils/            # seeding, I/O, logging (implemented)
```

Each module is independently testable and extensible via a small
registry pattern. New datasets, models, explainers, and metrics can be
added in later phases without modifying existing code — e.g. a new
dataset requires only one new `BaseDataset` subclass plus one line in
`DATASET_REGISTRY`.

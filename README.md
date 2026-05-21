# MLOps DVC vs Monolithic Pipeline 🚀

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![DVC](https://img.shields.io/badge/DVC-3.x-13adc7.svg)](https://dvc.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed.svg)](https://www.docker.com/)
[![GitHub Actions](https://img.shields.io/badge/CI-CD-GitHub_Actions-2088ff.svg)](https://github.com/features/actions)
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.3.2-f7931e.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Enterprise-grade MLOps benchmark project comparing a traditional monolithic ML workflow against a modular, reproducible DVC pipeline. Built for reproducibility, velocity, and portfolio-grade presentation.

This repository is designed as a portfolio-quality reference implementation for modern ML engineering. It demonstrates how to structure, orchestrate, cache, benchmark, containerize, and continuously validate a production-style pipeline using Python, Scikit-learn, DVC, Docker, Docker Compose, Google Drive remote storage, and GitHub Actions.

---

## Executive Overview

`MLOps DVC vs Monolithic Pipeline` is a side-by-side comparison of two production-relevant approaches to machine learning delivery:

1. A **monolithic script** that performs data loading, cleaning, feature encoding, model training, evaluation, and artifact persistence in one execution path.
2. A **modular DVC pipeline** that decomposes the same workflow into deterministic stages with dependency tracking, cached execution, experiment lineage, and remote artifact synchronization.

The goal is not simply to train a model. The goal is to demonstrate the engineering tradeoffs that matter in real ML systems:

- reproducibility under change,
- incremental execution speed,
- experiment traceability,
- artifact management,
- container portability,
- and CI/CD enforcement.

**Primary model:** `RandomForestClassifier`  
**Key metrics:** `accuracy`, `auc`, `f1_macro`  
**Pipeline stages:** `prepare -> featurize -> train -> evaluate`

---

## Architecture 🏗️

The repository implements a clean split between the baseline and the orchestrated ML flow.

### High-Level Flow

```text
			  +-----------------------------+
			  |  UCI Adult Dataset (CSV)     |
			  |  data/adult.csv              |
			  +--------------+--------------+
						  |
						  v
			    +---------------------------+
			    | Monolithic Baseline       |
			    | train_monolithic.py       |
			    | one-shot training + eval  |
			    +-------------+-------------+
						   |
						   v
				    model.joblib / metrics.json

						  OR

			  +-----------------------------+
			  |  DVC Orchestrated Pipeline  |
			  +--------------+--------------+
						  |
						  v
	   +-----------+   +--------------+   +----------+   +------------+
	   | prepare   |-->| featurize    |-->| train    |-->| evaluate   |
	   | raw CSV   |   | encode/split |   | RF model |   | metrics    |
	   +-----------+   +--------------+   +----------+   +------------+
			 |               |               |               |
			 v               v               v               v
	   data/processed.csv  data/features.npz  models/model.joblib  metrics/scores.json

	DVC cache, params.yaml, dvc.lock, dvc exp, remote sync, CI validation
```

### Why This Architecture Matters

- The monolithic script is ideal as a direct baseline for benchmarking and simple iteration.
- The DVC pipeline turns the same logic into independently cacheable stages.
- Parameter changes only invalidate the affected downstream stages.
- The pipeline remains deterministic through tracked inputs, tracked params, and locked dependencies.

---

## DVC DAG 🔗

The DVC workflow is expressed as a directed acyclic graph (DAG) in `dvc.yaml`.

```text
data/adult.csv
	 |
	 v
   prepare
	 |
	 v
 data/processed.csv
	 |
	 v
  featurize
	 |
	 v
 data/features.npz
	 |
	 v
	train
	 |
	 v
 models/model.joblib
	 |
	 v
   evaluate
	 |
	 v
 metrics/scores.json
```

### Orchestration Model

DVC makes every stage explicit:

- **deps** define source-code and data dependencies.
- **outs** define versioned artifacts.
- **params** define tracked hyperparameters and configuration knobs.
- **metrics** define evaluation outputs that are not cached as regular artifacts.

This means pipeline execution is not driven by guesswork. It is driven by dependency state. If a tracked dependency or parameter changes, DVC recomputes only the stages that are genuinely affected.

---

## Project Structure 🗂️

```text
.
├── .github/workflows/ci.yml        # GitHub Actions CI pipeline
├── Dockerfile                      # Container image for reproducible execution
├── docker-compose.yml              # Long-lived development/runtime container
├── dvc.yaml                        # DVC pipeline DAG
├── dvc.lock                        # Locked stage checksums and resolved outputs
├── params.yaml                     # Tracked pipeline configuration
├── train_monolithic.py             # Monolithic baseline workflow
├── src/
│   ├── prepare.py                  # Raw CSV ingestion and cleaning
│   ├── featurize.py                # Encoding and train/test split
│   ├── train.py                   # Model training
│   └── evaluate.py                # Evaluation and metrics export
├── data/
│   ├── adult.csv.dvc               # DVC pointer to raw dataset
│   ├── processed.csv               # Prepared dataset artifact
│   └── features.npz                # Encoded feature bundle
├── models/
│   └── model.joblib                # Trained model artifact
├── metrics/
│   └── scores.json                 # DVC evaluation metrics
├── benchmark.md                    # Benchmark analysis and results
├── test_caching.sh                 # Cache-validation script
└── requirements.txt                # Python dependency manifest
```

### Repository Intent

- The root-level monolithic script gives a one-command baseline for comparison.
- The `src/` package contains the modular DVC stages.
- The `data/`, `models/`, and `metrics/` directories represent the artifact lifecycle.
- The workflow and container files make the project portable across local machines and CI runners.

---

## Pipeline Stages ⚙️

### `prepare`

**Responsibility:** ingest the raw UCI Adult CSV, normalize the schema, remove incomplete rows, and persist a clean dataset.

**Output:** `data/processed.csv`

This stage establishes the canonical tabular representation used by both the DVC pipeline and the monolithic baseline.

### `featurize`

**Responsibility:** encode categorical variables, split train/test partitions, and serialize model-ready arrays.

**Output:** `data/features.npz`

This stage transforms the cleaned table into reusable feature artifacts for training and evaluation.

### `train`

**Responsibility:** train a `RandomForestClassifier` using tracked hyperparameters from `params.yaml`.

**Output:** `models/model.joblib`

The model is saved as a portable serialized artifact and is also used by the evaluation stage.

### `evaluate`

**Responsibility:** score the trained model on the held-out test split and emit official evaluation metrics.

**Output:** `metrics/scores.json`

The stage calculates the primary metrics used throughout the repository: `accuracy`, `auc`, and `f1_macro`.

---

## Experiment Tracking 🧪

DVC experiment tracking is the central mechanism for controlled iteration in this project.

### Core Commands

```bash
dvc exp run --set-param train.n_estimators=211
dvc exp show
```

### Why It Matters

- `dvc exp run` creates isolated, reproducible experiment variants without losing the baseline state.
- `dvc exp show` provides a comparison surface across experiments, parameters, and metrics.
- Parameterized runs keep the history of model changes visible instead of burying them in ad hoc commits or local notebooks.
- Experiment lineage is preserved through Git and DVC metadata together.

### Practical Outcome

This gives teams a controlled way to answer questions like:

- Which hyperparameter setting improved `auc`?
- Did changing the split seed alter `f1_macro`?
- Which stage actually changed after a config update?

---

## DVC Caching ⚡

DVC caching is one of the strongest engineering advantages in this repository.

### What the Cache Does

- Reuses outputs when inputs are unchanged.
- Avoids recomputing upstream stages when only downstream parameters change.
- Keeps iteration time predictable during hyperparameter tuning.

### What Gets Invalidated

- A source code change in a stage script.
- A modification to a tracked parameter in `params.yaml`.
- A change in a dependent artifact such as `data/processed.csv` or `data/features.npz`.

### Example Behavior

When only `train.n_estimators` changes, DVC can skip `prepare` and `featurize` and rerun only the impacted downstream stages. That is the operational difference between a pipeline and a script.

---

## Docker 🐳

The repository is containerized so the same workflow can be reproduced locally, in CI, or in a shared environment.

### Build

```bash
docker build -t mlops-dvc-vs-monolithic .
```

### Run

```bash
docker run --rm -it mlops-dvc-vs-monolithic
```

### Docker Compose

```bash
docker compose up --build -d
docker compose ps
docker compose run --rm app bash test_caching.sh
```

### Containerized Reproducibility

- The container image standardizes the runtime environment.
- Docker Compose keeps the workspace mounted for iterative development.
- The same dependency graph can be exercised without relying on a developer’s local Python setup.

---

## Benchmarking 📈

The benchmark compares the monolithic baseline against the modular DVC pipeline on the same dataset and model family.

| Metric                                 | Monolithic Script | DVC Pipeline |
| -------------------------------------- | ----------------: | -----------: |
| Full pipeline run time (s)             |              3.23 |        10.52 |
| Re-run time after param change (s)     |              3.23 |         4.33 |
| Iteration speedup (full/partial ratio) |             1.00x |        2.43x |
| Reproducibility score                  |               Low |         High |

### Interpretation

- The monolithic workflow is faster on a cold start because it has no orchestration overhead.
- The DVC pipeline pays a one-time orchestration cost, then wins on repeated experimentation through cache reuse.
- The most relevant metric for production ML is not just first-run speed. It is the time to safely iterate on a model while preserving lineage.

### Benchmark Takeaway

For exploratory local runs, the monolithic script is acceptable.
For a team-oriented or enterprise pipeline, DVC becomes more valuable because incremental reruns reduce wasted compute and preserve traceability.

---

## Remote Storage ☁️

This project is designed to work with a Google Drive-backed DVC remote.

### Typical Remote Workflow

```bash
dvc push
dvc pull
```

### Operational Value

- Large or derived artifacts can be synchronized outside Git.
- Teams can share reproducible pipeline outputs without bloating the repository.
- Remote storage supports collaboration across machines and environments.
- The artifact store remains aligned with tracked metadata and stage locks.

---

## CI/CD 🤖

GitHub Actions provides automated validation for the repository.

### What the Workflow Verifies

- Python 3.10 environment setup.
- Dependency installation from `requirements.txt`.
- Successful execution of `train_monolithic.py`.
- Successful execution of `dvc repro`.
- Required artifact presence.
- DVC cache behavior through a second pipeline run.

### Why This Matters

The workflow acts as a guardrail for regression control. A broken pipeline, a missing dataset, or a change that invalidates reproducibility is caught before merge.

---

## Reproducibility 📦

Reproducibility is treated as a first-class engineering requirement.

### Deterministic Controls

- `params.yaml` stores tracked hyperparameters and execution settings.
- `dvc.lock` freezes resolved stage inputs and outputs.
- The pipeline stages read the same controlled data path and split logic.
- Git and DVC together preserve lineage across code, config, and artifacts.

### Reproducibility Outcomes

- Re-running the same commit should produce the same pipeline state.
- Changing tracked parameters should produce predictable downstream invalidation.
- Model artifacts and metrics remain auditable over time.

---

## Metrics 📊

The evaluation layer produces three metrics used for model comparison:

### `accuracy`

Measures overall classification correctness across the held-out test split.

### `auc`

Measures ranking quality across decision thresholds and is especially useful for classification quality analysis beyond simple accuracy.

### `f1_macro`

Balances precision and recall across classes and is useful when class balance or error symmetry matters.

### Reported Metrics Artifact

The DVC pipeline persists metrics in `metrics/scores.json`, while the monolithic workflow writes `metrics.json` for baseline comparison.

---

## Enterprise Features 🏢

- Modular stage separation for maintainability and clear ownership.
- DVC cache-aware execution for efficient retraining.
- Experiment tracking with `dvc exp run` and `dvc exp show`.
- Docker and Docker Compose support for environment parity.
- Google Drive remote support for artifact synchronization.
- GitHub Actions validation for automated quality gates.
- Benchmark documentation for performance and workflow comparison.
- Clean artifact boundaries between raw data, processed data, model binaries, and metrics.

---

## Future Improvements 🧭

- **MLflow** for richer experiment tracking, model metadata, and registry integration.
- **Kubernetes** for scalable deployment and pipeline execution at cluster level.
- **FastAPI serving** for production inference endpoints.
- **Airflow orchestration** for broader scheduling and dependency management.
- **Model registry** for lifecycle governance, approvals, and promotion workflows.

---

## Quick Start

```bash
python train_monolithic.py
dvc repro
dvc exp show
```

If you want the full reproducible environment:

```bash
docker compose up --build -d
docker compose run --rm app bash test_caching.sh
```

---

## Conclusion 🎯

This repository is a practical demonstration of how mature MLOps systems should be structured: explicit stages, controlled parameters, cache-aware execution, artifact governance, container portability, remote synchronization, and CI/CD enforcement.

The monolithic script provides a useful baseline, but the DVC pipeline is the stronger engineering model for teams that care about repeatability, collaborative experimentation, and maintainable machine learning delivery. For recruiters and engineering leads, the value of this project is that it shows not just model training, but the operational discipline required to run ML as a reliable software system.

---

## License

This project is provided for portfolio, learning, and demonstration purposes. Add your preferred license if you plan to distribute it publicly.

# Benchmark Report: Monolithic vs DVC Pipeline

## Summary

This document compares the performance and workflow efficiency of two machine learning pipeline approaches:

1. **Monolithic Script** - Traditional single-script approach
2. **DVC Pipeline** - Modular, version-controlled workflow

## Results Table

| Metric                                 | Monolithic Script | DVC Pipeline |
| -------------------------------------- | ----------------- | ------------ |
| Full pipeline run time (s)             | TBD               | TBD          |
| Re-run time after param change (s)     | TBD               | TBD          |
| Iteration speedup (full/partial ratio) | TBD               | TBD          |
| Memory usage (MB)                      | TBD               | TBD          |
| Reproducibility score                  | Low               | High         |

## Analysis

### 1. Full Pipeline Execution

- **Monolithic**: Time required to run entire script from scratch
- **DVC**: Time required to run all 4 stages (prepare → featurize → train → evaluate)

### 2. Partial Re-run Efficiency

Changed parameter: `train.n_estimators` from 100 to 200

- **Monolithic**: Entire script must re-execute (including data loading/processing)
- **DVC**: Only train and evaluate stages re-execute; prepare and featurize are cached

### 3. Caching Effectiveness

DVC's intelligent caching skips stages whose inputs haven't changed, demonstrating:

- Faster iteration during hyperparameter tuning
- Reduced computational overhead
- Better resource utilization

## Experiment Tracking Comparison

### Monolithic Challenges

- Manual parameter modification required
- Difficult to track experiment history
- Hard to reproduce past results
- No built-in metrics comparison

### DVC Advantages

- `dvc exp run --set-param` for easy parameter variations
- `dvc exp show` for comprehensive metrics table
- Automatic Git integration for reproducibility
- Full experiment lineage tracking

## Reproducibility Analysis

### When DVC Breaks Even

DVC setup overhead pays for itself when:

1. **Team size** > 1: Collaboration becomes critical
2. **Experiment count** > 5: Manual tracking becomes infeasible
3. **Project duration** > 2 weeks: Reproducibility needs increase
4. **Data size** > 100MB: Version control efficiency matters

### Recommendation

- **Start with monolithic** for quick POCs (< 1 week)
- **Migrate to DVC** for production projects with collaboration
- **Use DVC from day one** for team-based ML engineering

## Conclusions

[To be completed after benchmarking]

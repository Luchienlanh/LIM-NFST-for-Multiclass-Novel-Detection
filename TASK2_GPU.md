# Task 2 GPU grid search and resume

See `TASK2_GPU_AUDIT.md` for the CPU-reference audit, measured numerical
differences, projection-reuse checks, and the remaining CUDA validation gap.

Run from the repository root with a CUDA-enabled PyTorch installation and
NumPy, pandas and scikit-learn installed:

```bash
python examples/lim-models-gpu.py --grid-search --dataset BoT_IoT --seed 42 --cv-folds 5 --output-dir results/task2_gpu_BoT_IoT --resume
```

Use the **same command and output directory** after restarting the process.
`--resume` also starts a new run when the output directory is empty. Without
`--resume`, the script refuses to overwrite an existing grid run. Resume is
supported for grid search, not fixed experiments.

- The default grid still evaluates 3,375 configurations per outer novel class.
  There are 225 projection groups, each with 15 neighbor/quantile combinations.
  Each group fits once per seed, inner novel class and CV fold.
- After all configurations in one group fold have been evaluated, a checkpoint
  is written under `.checkpoints/` in the output directory. It stores metrics
  and RFF gamma at full precision, not GPU tensors or fitted models.
- Completed group folds are loaded without fitting or predicting again.
  Preprocessing is repeated; a fold interrupted before its checkpoint was
  committed is recomputed. A full group does not have to finish before progress
  is protected.
- Checkpoints and grid CSV/JSON artifacts use flushed temporary files followed
  by atomic replacement. An unfinished temporary file is never a checkpoint.
  Missing or incomplete checkpoint results are recomputed. Checkpoints with a
  different run context are rejected instead of mixed with this run.
- Resume requires matching code, library versions, grid settings, seeds, folds,
  novel-class selection and metric schema. `dataset_fingerprints.json` also
  checks the loaded data and its row order. Use a new output directory when
  intentionally changing an experiment.
- Per-configuration files are regenerated from completed folds. Global
  `grid_search_results.csv`, `grid_candidates.csv` and `best_parameters.csv`
  are rebuilt when the grid finishes; resume does not append duplicate rows.
- Runs created before checkpoint support are not imported automatically. Their
  CSV files alone do not establish compatible code, data and complete folds.
- Keep the whole output directory, including `.checkpoints/` and both manifest
  files, on persistent storage. Local VM or Colab storage that is deleted cannot
  be recovered by resume. Use only one process per output directory.

In a Colab code cell, prefix the command with `!`. If using a mounted persistent
drive, point `--output-dir` at a directory on that drive, and keep the same code,
library versions, dataset and arguments across sessions.

## Task 1 versus Task 2 results

The projection-reuse optimization and resume support do **not** make Task 2's
schema or metric definitions identical to Task 1.

| Aspect | Task 1: `examples/lim-models.py` | Task 2: `examples/lim-models-gpu.py` |
| --- | --- | --- |
| Evaluation | Closed-world StratifiedKFold | Strict nested LOCO for multiclass novelty detection |
| Metric columns | 30: 15 core, 9 curve, 2 probability, 4 timing | 11 existing classification/novelty metrics |
| Global result identifiers | `dataset`, `model` | Also `outer_novel_class`, `test_set` |
| Fold identifiers | Seed and fold | Also `inner_novel_class`; `known`, `novel`, `combined` test sets |
| Configuration layout | Dataset / readable configuration name | Dataset / outer novel class / `cfg_<id>` |
| Configuration CSVs | `fold_results.csv`, `results.csv`, `cv_summary.csv` | `cv_results.csv`, `cv_summary.csv` |

Task 2 retains accuracy and balanced accuracy for open/closed-world evaluation,
novel-detection precision/recall/F1, macro precision/recall/F1, and open-world MCC.
It does not currently export Task 1's micro/weighted precision/recall/F1,
specificity, Cohen's kappa, ROC-AUC/AP/PR-AUC, log loss, Brier score or timing
columns. Even metrics with the same name refer to different evaluation scopes.
Adding score/probability metrics for the novel class requires an explicit
scoring definition; this change does not invent one or change MND selection.

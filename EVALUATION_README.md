# Shared Evaluation Interface

This is the common evaluator for every COMP9517 group method. It keeps Top-1,
Top-5, macro precision, macro recall, macro F1, prediction rows, timing fields,
and confusion matrices consistent across handcrafted and deep models.

## 1. Export model scores

Each method must export one NPZ file. `scores` may contain logits,
probabilities, or SVM decision scores; a larger value must indicate a higher
predicted rank.

```python
import numpy as np

np.savez_compressed(
    "model_validation_scores.npz",
    scores=all_scores,                  # shape: (5000, 500)
    true_labels=all_true_labels,        # shape: (5000,)
    image_paths=np.asarray(all_paths, dtype=str),
    class_indices=np.arange(500),       # score-column order
)
```

Do not use `dtype=object`. The evaluator loads NPZ files with
`allow_pickle=False`.

## 2. Run the same evaluator

From the project root:

```bash
.venv/bin/python -m src.evaluation.evaluate_predictions \
  --input model_validation_scores.npz \
  --output-dir experiment_workspace/group_results/model_name_validation \
  --model-name "Model Name" \
  --split validation \
  --inference-seconds 12.34 \
  --training-seconds 3600
```

`--inference-seconds` must be measured around model inference by the model
owner. Do not include training or feature extraction unless the report labels
those times separately.

## 3. Required output

Every evaluated model receives the same files:

```text
model_name_validation/
├── metrics.json
├── predictions.csv
├── confusion_matrix.csv
├── confusion_matrix.png
└── evaluation_config.json
```

Keep validation and test in separate output directories. Select all
hyperparameters using validation only, freeze the model, and evaluate test once.
Never use `--overwrite` to replace a frozen final result.

## 4. Required checks before sharing

- Exactly 500 score columns in `class_indices` order.
- Exactly one label and one relative image path per score row.
- Validation/test rows match the frozen CSV for that split.
- All score values are finite.
- `class_indices` matches `data_metadata/class_to_idx.json`.
- Training and inference timing definitions are stated.

Run the automated interface tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

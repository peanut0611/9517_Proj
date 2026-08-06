# Member A -> Member B: Code Material

This folder contains Member A's code and metadata for final code integration.

## Include

- `notebooks/member_a_experiment.ipynb`: primary submission notebook, with saved
  experiment outputs and execution counts.
- `src/data/`: reproducible split, repair and audit scripts.
- `src/evaluation/`: shared evaluation interface.
- `tests/test_evaluate_predictions.py`: evaluator tests.
- `data_metadata/`: frozen relative-path CSVs, class mapping and audit records.
- `DATASET_README.md` and `EVALUATION_README.md`.
- `requirements.txt`.
- `FINAL_MODEL_CONFIG.json`: configuration record only; no trained model.

## Exclude from the final ZIP

- `Dataset_v1.0_FROZEN/` and all original images.
- `.venv/`, `feature_cache/`, `experiment_workspace/` and `__pycache__/`.
- NPZ score exports, trained models, checkpoints and result images.

The saved outputs in `member_a_experiment.ipynb` are intentionally retained as
experiment evidence. To rerun the notebook, place the separate frozen dataset
release at the project root as `Dataset_v1.0_FROZEN/`; the image dataset itself
must not be included in the final code ZIP.

Run the evaluator tests from the project root with:

```bash
python -m unittest discover -s tests -p 'test_evaluate_predictions.py' -v
```

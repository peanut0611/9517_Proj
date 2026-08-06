from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt

from .common import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare frozen C0/C1 runs.")
    parser.add_argument("--c1-run", type=Path, required=True)
    parser.add_argument("--c0-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {
        "C1 Scratch ResNet-18 + augmentation": args.c1_run,
        "C0 Scratch ResNet-18 without augmentation": args.c0_run,
    }

    metric_rows: list[dict[str, float | str]] = []
    metrics_by_run: dict[str, dict[str, dict[str, float]]] = {}
    for method, run_dir in runs.items():
        metrics_by_run[method] = {}
        for split in ("val", "test"):
            metrics = load_json(run_dir / f"metrics_{split}.json")
            metrics_by_run[method][split] = metrics
            metric_rows.append({"method": method, "split": split, **metrics})
    pd.DataFrame(metric_rows).to_csv(
        args.output_dir / "augmentation_ablation_metrics.csv", index=False
    )

    c1_test = metrics_by_run["C1 Scratch ResNet-18 + augmentation"]["test"]
    c0_test = metrics_by_run["C0 Scratch ResNet-18 without augmentation"]["test"]
    deltas = {
        key: float(c1_test[key] - c0_test[key])
        for key in (
            "top1_accuracy",
            "top5_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        )
    }
    save_json(
        {
            "definition": "C1 minus C0; raw fractions and percentage points",
            "raw_deltas": deltas,
            "percentage_point_deltas": {
                key: value * 100 for key, value in deltas.items()
            },
        },
        args.output_dir / "augmentation_ablation_test_deltas.json",
    )

    histories = {
        method: pd.read_csv(run_dir / "history.csv") for method, run_dir in runs.items()
    }
    colors = {
        "C1 Scratch ResNet-18 + augmentation": "#1f77b4",
        "C0 Scratch ResNet-18 without augmentation": "#d62728",
    }
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = (
        ("train_loss", "Training loss", False),
        ("val_loss", "Validation loss", False),
        ("val_top1", "Validation Top-1 (%)", True),
        ("val_macro_f1", "Validation macro-F1 (%)", True),
    )
    for axis, (column, title, as_percent) in zip(axes.flat, panels):
        for method, history in histories.items():
            values = history[column] * (100 if as_percent else 1)
            axis.plot(
                history["epoch"], values, label=method, color=colors[method], linewidth=1.8
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Scratch ResNet-18 augmentation ablation")
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "augmentation_ablation_curves.png",
        dpi=220,
        bbox_inches="tight",
    )
    figure.savefig(
        args.output_dir / "augmentation_ablation_curves.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)
    print(f"comparison written to {args.output_dir}")


if __name__ == "__main__":
    main()

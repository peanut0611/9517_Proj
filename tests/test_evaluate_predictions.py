import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.evaluate_predictions import evaluate_scores, save_evaluation


class SharedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.scores = np.array(
            [
                [0.9, 0.1, 0.0],
                [0.1, 0.7, 0.2],
                [0.6, 0.3, 0.1],
                [0.2, 0.5, 0.3],
            ]
        )
        self.labels = np.array([0, 1, 2, 2])
        self.paths = np.array(["a.jpg", "b.jpg", "c.jpg", "d.jpg"])

    def test_metrics_and_top_k(self):
        result = evaluate_scores(
            self.scores,
            self.labels,
            image_paths=self.paths,
            inference_seconds=0.04,
        )
        metrics = result["metrics"]
        self.assertEqual(metrics["num_images"], 4)
        self.assertEqual(metrics["num_classes"], 3)
        self.assertAlmostEqual(metrics["top1_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["top5_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["inference_ms_per_image"], 10.0)
        self.assertEqual(int(result["confusion_matrix"].sum()), 4)
        self.assertEqual(result["predictions"].predicted_class.tolist(), [0, 1, 0, 1])

    def test_non_contiguous_class_indices(self):
        result = evaluate_scores(
            self.scores,
            np.array([10, 20, 30, 30]),
            class_indices=np.array([10, 20, 30]),
        )
        self.assertEqual(result["predictions"].predicted_class.tolist(), [10, 20, 10, 20])
        self.assertEqual(result["confusion_matrix"].shape, (3, 3))

    def test_rejects_unknown_label_and_nonfinite_score(self):
        with self.assertRaisesRegex(ValueError, "absent from scores"):
            evaluate_scores(self.scores, np.array([0, 1, 2, 99]))
        invalid_scores = self.scores.copy()
        invalid_scores[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_scores(invalid_scores, self.labels)

    def test_saves_complete_artifact_set(self):
        result = evaluate_scores(self.scores, self.labels, image_paths=self.paths)
        with tempfile.TemporaryDirectory() as temporary:
            output = save_evaluation(
                result,
                Path(temporary) / "model_a_val",
                model_name="Synthetic",
                split="validation",
            )
            expected = {
                "metrics.json",
                "predictions.csv",
                "confusion_matrix.csv",
                "confusion_matrix.png",
                "evaluation_config.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            metrics = json.loads((output / "metrics.json").read_text())
            predictions = pd.read_csv(output / "predictions.csv")
            self.assertEqual(metrics["model_name"], "Synthetic")
            self.assertEqual(len(predictions), 4)


if __name__ == "__main__":
    unittest.main()

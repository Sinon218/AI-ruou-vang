import unittest

from wine_quality_pipeline import build_pipeline, run_pipeline


class TestWineQualityPipeline(unittest.TestCase):
    def test_pipeline_has_preprocessing_and_model(self):
        pipeline = build_pipeline()
        self.assertEqual([name for name, _ in pipeline.steps], ["scaler", "classifier"])

    def test_run_pipeline_returns_reasonable_accuracy(self):
        result = run_pipeline()
        self.assertGreaterEqual(result.accuracy, 0.85)
        self.assertIn("precision", result.report)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import unittest

from service.app import app, sample_submission


ROOT = Path(__file__).resolve().parents[1]


class SampleDownloadTests(unittest.TestCase):
    def test_only_basic_submission_download_route_is_exposed(self) -> None:
        sample_routes = sorted(
            route.path for route in app.routes if route.path.startswith("/samples/")
        )
        self.assertEqual(sample_routes, ["/samples/submission.py"])

    def test_download_is_exact_baseline_and_named_submission_py(self) -> None:
        response = sample_submission()
        expected = ROOT / "submissions" / "baseline_adamw" / "submission.py"
        self.assertEqual(Path(response.path), expected)
        self.assertEqual(response.media_type, "text/x-python")
        self.assertIn(
            'filename="submission.py"',
            response.headers["content-disposition"],
        )
        source = expected.read_text(encoding="utf-8")
        self.assertIn("class Model", source)
        self.assertIn("torch.optim.AdamW", source)
        self.assertNotIn("for loop in", source)


if __name__ == "__main__":
    unittest.main()

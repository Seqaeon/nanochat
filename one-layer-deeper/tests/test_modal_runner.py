from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import modal_runner


ROOT = Path(__file__).resolve().parents[1]


class ModalRunnerTests(unittest.TestCase):
    def test_smoke_accepts_finite_benchmark_score(self) -> None:
        benchmark_result = {
            "score": {"mean_exact_accuracy": 0.5},
            "seeds": [],
        }

        result = modal_runner._validate_smoke_result(
            {
                "returncode": 0,
                "timed_out": False,
                "benchmark_result": benchmark_result,
            }
        )

        self.assertIs(result, benchmark_result)

    def test_smoke_failure_includes_remote_log_tail(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            r"evaluator exited[\s\S]*remote failure",
        ):
            modal_runner._validate_smoke_result(
                {
                    "returncode": 1,
                    "timed_out": False,
                    "log_tail": "remote failure",
                }
            )

    def test_silent_process_is_terminated_at_wall_clock_timeout(self) -> None:
        with patch.object(modal_runner, "REMOTE_ROOT", str(ROOT)):
            result = modal_runner._run_command(
                [
                    sys.executable,
                    "-c",
                    "import time; time.sleep(5)",
                ],
                timeout_seconds=1,
            )

        self.assertTrue(result["timed_out"])
        self.assertIsNotNone(result["returncode"])
        self.assertLess(result["duration_seconds"], 4)


if __name__ == "__main__":
    unittest.main()

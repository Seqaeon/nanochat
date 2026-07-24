from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from benchmark.metrics import MetricRecorder
from client.cli import build_parser, command_metrics
from service import app as service_app
from service.evaluator import _separate_structured_metrics
from service.metrics import metrics_to_jsonl, validate_structured_metrics


def _benchmark_result() -> dict:
    return {
        "score": {"mean_exact_accuracy": 0.625},
        "seeds": [
            {
                "seed": 74,
                "completed_training_steps": 300,
                "training_seconds": 12.34,
                "evaluation": {
                    "test": {"loss": 1.23456, "exact_accuracy": 0.625},
                },
            }
        ],
    }


def _metric_payload() -> list[dict]:
    return [
        {
            "type": "train",
            "seed": 74,
            "step": 1,
            "elapsed_seconds": 0.14,
            "loss": 4.12349,
            "exact_accuracy": 0.0004,
        },
        {
            "type": "train",
            "seed": 74,
            "step": 300,
            "elapsed_seconds": 12.31,
            "loss": 1.23449,
            "exact_accuracy": 0.6249,
        },
        {
            "type": "evaluation",
            "seed": 74,
            "split": "test",
            "loss": 1.235,
            "exact_accuracy": 0.625,
        },
        {
            "type": "summary",
            "completed_steps": 300,
            "training_seconds": 12.3,
            "mean_exact_accuracy": 0.625,
        },
    ]


class MetricRecorderTests(unittest.TestCase):
    def test_records_are_quantized_bounded_and_keep_last_point(self) -> None:
        recorder = MetricRecorder(max_train_records=4)
        for step in range(1, 21):
            recorder.record_training(
                seed=74,
                step=step,
                elapsed_seconds=step / 3,
                loss=1.23456 + step,
                exact_accuracy=0.12349,
            )

        records = recorder.snapshot()

        self.assertLessEqual(len(records), 4)
        self.assertEqual(records[0]["step"], 1)
        self.assertEqual(records[-1]["step"], 20)
        self.assertEqual(records[-1]["loss"], 21.235)
        self.assertEqual(records[-1]["exact_accuracy"], 0.123)

    def test_snapshot_does_not_expose_mutable_recorder_state(self) -> None:
        recorder = MetricRecorder()
        recorder.record_training(
            seed=74,
            step=1,
            elapsed_seconds=0.1,
            loss=1.0,
            exact_accuracy=0.0,
        )

        snapshot = recorder.snapshot()
        snapshot[0]["loss"] = 999.0

        self.assertEqual(recorder.snapshot()[0]["loss"], 1.0)


class StructuredMetricValidationTests(unittest.TestCase):
    def test_valid_payload_is_allowlisted_and_quantized(self) -> None:
        metrics = validate_structured_metrics(
            _metric_payload(),
            _benchmark_result(),
        )

        self.assertEqual(metrics[0]["elapsed_seconds"], 0.1)
        self.assertEqual(metrics[0]["loss"], 4.123)
        self.assertEqual(metrics[1]["exact_accuracy"], 0.625)
        self.assertEqual(
            json.loads(metrics_to_jsonl(metrics).splitlines()[-1]),
            metrics[-1],
        )

    def test_unknown_fields_and_participant_text_are_rejected(self) -> None:
        payload = _metric_payload()
        payload[0]["participant_output"] = "PRIVATE_DATA_SENTINEL"

        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            validate_structured_metrics(payload, _benchmark_result())

    def test_evaluation_values_must_match_benchmark_result(self) -> None:
        payload = _metric_payload()
        payload[2]["exact_accuracy"] = 0.999

        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_structured_metrics(payload, _benchmark_result())

    def test_evaluator_removes_metrics_from_ordinary_result(self) -> None:
        raw_result = {
            **_benchmark_result(),
            "structured_metrics": _metric_payload(),
        }

        result, metrics = _separate_structured_metrics(raw_result)

        self.assertNotIn("structured_metrics", result)
        self.assertIsNotNone(metrics)
        self.assertIn("structured_metrics", raw_result)

    def test_invalid_metrics_do_not_invalidate_score(self) -> None:
        raw_result = {
            **_benchmark_result(),
            "structured_metrics": [{"type": "train", "raw": "PRIVATE_DATA_SENTINEL"}],
        }

        result, metrics = _separate_structured_metrics(raw_result)

        self.assertEqual(result["score"]["mean_exact_accuracy"], 0.625)
        self.assertIsNone(metrics)


class ParticipantMetricsApiTests(unittest.TestCase):
    def test_status_response_excludes_raw_output_and_error(self) -> None:
        user_id = uuid4()
        submission_id = uuid4()
        row = {
            "id": submission_id,
            "user_id": user_id,
            "filename": "submission.py",
            "status": "failed",
            "manifest_name": "h100_easy_e1.json",
            "run_id": uuid4(),
            "error": "PRIVATE_DATA_SENTINEL",
            "log_tail": "PRIVATE_DATA_SENTINEL",
            "metrics_available": False,
        }
        with (
            patch.object(service_app, "_api_user", return_value={"id": user_id}),
            patch.object(service_app.database, "get_submission", return_value=row),
        ):
            response = service_app.submission_api(
                submission_id,
                authorization="Bearer old_test",
            )

        payload = json.loads(response.body)
        self.assertNotIn("error", payload)
        self.assertNotIn("log_tail", payload)
        self.assertNotIn("user_id", payload)
        self.assertNotIn("PRIVATE_DATA_SENTINEL", response.body.decode())
        self.assertEqual(payload["error_code"], "EVALUATION_FAILED")

    def test_status_uses_persisted_error_code_after_details_are_purged(self) -> None:
        user_id = uuid4()
        submission_id = uuid4()
        row = {
            "id": submission_id,
            "user_id": user_id,
            "filename": "submission.py",
            "status": "failed",
            "manifest_name": "h100_easy_e1.json",
            "run_id": uuid4(),
            "error": None,
            "error_code": "OUT_OF_MEMORY",
            "log_tail": "",
            "metrics_available": False,
        }
        with (
            patch.object(service_app, "_api_user", return_value={"id": user_id}),
            patch.object(service_app.database, "get_submission", return_value=row),
        ):
            response = service_app.submission_api(
                submission_id,
                authorization="Bearer old_test",
            )

        self.assertEqual(json.loads(response.body)["error_code"], "OUT_OF_MEMORY")
    def test_owner_can_download_jsonl_metrics(self) -> None:
        user_id = uuid4()
        submission_id = uuid4()
        metrics = validate_structured_metrics(
            _metric_payload(),
            _benchmark_result(),
        )
        with (
            patch.object(service_app, "_api_user", return_value={"id": user_id}),
            patch.object(
                service_app.database,
                "get_submission_metrics",
                return_value={"status": "succeeded", "metrics": metrics},
            ) as get_metrics,
        ):
            response = service_app.submission_metrics_api(
                submission_id,
                authorization="Bearer old_test",
            )

        get_metrics.assert_called_once_with(submission_id, user_id)
        self.assertEqual(response.body.decode(), metrics_to_jsonl(metrics))
        self.assertIn(
            f'{submission_id}-metrics.jsonl',
            response.headers["content-disposition"],
        )


class MetricsCliTests(unittest.TestCase):
    def test_metrics_command_downloads_authenticated_artifact(self) -> None:
        submission_id = str(uuid4())
        response = Mock(content=b'{"type":"summary"}\n')
        args = build_parser().parse_args(
            [
                "metrics",
                submission_id,
                "--server",
                "https://one-layer.example/",
                "--api-key",
                "old_test",
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "metrics.jsonl"
            args.output = str(output_path)
            output = io.StringIO()
            with (
                patch("client.cli.httpx.get", return_value=response) as get,
                redirect_stdout(output),
            ):
                result = args.handler(args)

            self.assertEqual(output_path.read_bytes(), response.content)

        self.assertIs(args.handler, command_metrics)
        self.assertEqual(result, 0)
        self.assertIn("saved metrics", output.getvalue())
        get.assert_called_once_with(
            f"https://one-layer.example/api/submissions/{submission_id}/metrics",
            headers={"Authorization": "Bearer old_test"},
            timeout=30.0,
        )
        response.raise_for_status.assert_called_once_with()

    def test_existing_output_is_rejected_before_download(self) -> None:
        submission_id = str(uuid4())
        args = build_parser().parse_args(
            ["metrics", submission_id, "--api-key", "old_test"]
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "metrics.jsonl"
            output_path.write_text("existing\n", encoding="utf-8")
            args.output = str(output_path)
            with patch("client.cli.httpx.get") as get:
                with self.assertRaisesRegex(ValueError, "already exists"):
                    args.handler(args)

        get.assert_not_called()


if __name__ == "__main__":
    unittest.main()

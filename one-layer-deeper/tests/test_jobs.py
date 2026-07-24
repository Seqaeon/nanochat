from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from client.cli import build_parser, command_jobs
from service import app as service_app
from service.db import Database


class _Result:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def fetchall(self) -> list[dict]:
        return self.rows


class _Connection:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.query = ""
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query: str, params=None) -> _Result:
        self.query = " ".join(query.split())
        self.params = params
        return _Result(self.rows)


class JobsDatabaseTests(unittest.TestCase):
    def test_list_submissions_is_user_scoped_and_active_by_default(self) -> None:
        user_id = uuid4()
        rows = [{"id": uuid4(), "status": "running"}]
        connection = _Connection(rows)
        database = Database("unused")

        with patch.object(database, "_connect", return_value=connection):
            self.assertEqual(database.list_submissions(user_id), rows)

        self.assertIn("WHERE s.user_id = %s", connection.query)
        self.assertIn("r.status IN ('queued', 'running')", connection.query)
        self.assertEqual(connection.params, (user_id, True))


class JobsApiTests(unittest.TestCase):
    def test_jobs_endpoint_uses_authenticated_user(self) -> None:
        user_id = uuid4()
        rows = [{"id": str(uuid4()), "status": "failed"}]
        with (
            patch.object(service_app, "_api_user", return_value={"id": user_id}),
            patch.object(
                service_app.database,
                "list_submissions",
                return_value=rows,
            ) as list_submissions,
        ):
            response = service_app.submissions_api(
                active=False,
                authorization="Bearer old_test",
            )

        self.assertEqual(json.loads(response.body), rows)
        list_submissions.assert_called_once_with(user_id, active_only=False)


class JobsCliTests(unittest.TestCase):
    def test_jobs_defaults_to_active_and_prints_submission_id(self) -> None:
        submission_id = str(uuid4())
        response = Mock()
        response.json.return_value = [
            {
                "id": submission_id,
                "filename": "submission.py",
                "status": "running",
                "tier": "easy",
                "dataset_label": "E1",
            }
        ]
        args = build_parser().parse_args(
            [
                "jobs",
                "--server",
                "https://one-layer.example/",
                "--api-key",
                "old_test",
            ]
        )

        output = io.StringIO()
        with (
            patch("client.cli.httpx.get", return_value=response) as get,
            redirect_stdout(output),
        ):
            result = args.handler(args)

        self.assertIs(args.handler, command_jobs)
        self.assertEqual(result, 0)
        self.assertIn(submission_id, output.getvalue())
        get.assert_called_once_with(
            "https://one-layer.example/api/submissions",
            params={"active": True},
            headers={"Authorization": "Bearer old_test"},
            timeout=30.0,
        )

    def test_jobs_all_json_requests_history(self) -> None:
        rows = [{"id": str(uuid4()), "status": "succeeded"}]
        response = Mock()
        response.json.return_value = rows
        args = build_parser().parse_args(
            ["jobs", "--all", "--json", "--api-key", "old_test"]
        )

        output = io.StringIO()
        with patch("client.cli.httpx.get", return_value=response) as get:
            with redirect_stdout(output):
                result = args.handler(args)

        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue()), rows)
        self.assertEqual(get.call_args.kwargs["params"], {"active": False})

    def test_jobs_reports_when_no_active_jobs_exist(self) -> None:
        response = Mock()
        response.json.return_value = []
        args = build_parser().parse_args(["jobs", "--api-key", "old_test"])

        output = io.StringIO()
        with patch("client.cli.httpx.get", return_value=response):
            with redirect_stdout(output):
                result = args.handler(args)

        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue().strip(), "No active jobs.")


if __name__ == "__main__":
    unittest.main()

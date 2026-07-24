from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from service import app as service_app
from service.db import Database
from service.evaluator import evaluate_run
from service.tiers import TIER_BY_ID


class _Result:
    def __init__(self, *, row=None, rows=None, rowcount: int = 0) -> None:
        self.row = row
        self.rows = rows or []
        self.rowcount = rowcount

    def fetchone(self):
        return self.row

    def fetchall(self):
        return self.rows


class _Connection:
    def __init__(self, results: list[_Result] | None = None) -> None:
        self.results = list(results or [])
        self.queries: list[tuple[str, tuple | None]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def execute(self, query: str, params=None) -> _Result:
        self.queries.append((" ".join(query.split()), params))
        return self.results.pop(0) if self.results else _Result()


class DatabaseDeadlineTests(unittest.TestCase):
    def test_initialize_adds_deadline_column_and_partial_index(self) -> None:
        connection = _Connection()
        database = Database("unused")

        with patch.object(database, "_connect", return_value=connection):
            database.initialize()

        queries = [query for query, _ in connection.queries]
        self.assertTrue(
            any(
                "ALTER TABLE runs ADD COLUMN IF NOT EXISTS deadline_at TIMESTAMPTZ"
                in query
                for query in queries
            )
        )
        self.assertTrue(
            any(
                "ALTER TABLE runs ADD COLUMN IF NOT EXISTS metrics JSONB" in query
                for query in queries
            )
        )
        self.assertTrue(
            any(
                "ALTER TABLE runs ADD COLUMN IF NOT EXISTS error_code TEXT" in query
                for query in queries
            )
        )
        self.assertTrue(
            any(
                "runs_status_deadline_at_idx" in query
                and "WHERE deadline_at IS NOT NULL" in query
                for query in queries
            )
        )

    def test_mark_running_persists_absolute_database_deadline(self) -> None:
        run_id = uuid4()
        connection = _Connection([_Result(row={"id": run_id})])
        database = Database("unused")

        with patch.object(database, "_connect", return_value=connection):
            database.mark_running(run_id, "fc-test", 6180)

        query, params = connection.queries[0]
        self.assertIn("deadline_at = NOW() + (%s * INTERVAL '1 second')", query)
        self.assertEqual(params, ("fc-test", 6180, run_id))

    def test_mark_running_rejects_nonpositive_deadline(self) -> None:
        database = Database("unused")

        with self.assertRaisesRegex(ValueError, "must be positive"):
            database.mark_running(uuid4(), "fc-test", 0)

    def test_backfill_sets_missing_deadlines_from_original_start(self) -> None:
        connection = _Connection(
            [_Result(rowcount=2), _Result(rowcount=1)]
        )
        database = Database("unused")

        with patch.object(database, "_connect", return_value=connection):
            updated = database.backfill_run_deadlines(
                {"easy": 450, "hard": 6180}
            )

        self.assertEqual(updated, 3)
        for query, _ in connection.queries:
            self.assertIn("deadline_at = started_at", query)
            self.assertIn("status = 'running'", query)
            self.assertIn("deadline_at IS NULL", query)
        self.assertEqual(
            [params for _, params in connection.queries],
            [(450, "easy"), (6180, "hard")],
        )

    def test_expiration_is_one_atomic_guarded_update(self) -> None:
        expired = {
            "id": uuid4(),
            "submission_id": uuid4(),
            "modal_call_id": "fc-test",
            "deadline_at": "2026-07-16T21:40:00Z",
        }
        connection = _Connection([_Result(rows=[expired])])
        database = Database("unused")

        with patch.object(database, "_connect", return_value=connection):
            rows = database.fail_expired_runs()

        self.assertEqual(rows, [expired])
        query, params = connection.queries[0]
        self.assertIsNone(params)
        self.assertTrue(query.startswith("UPDATE runs SET status = 'failed'"))
        self.assertIn("error = 'TimeoutError:", query)
        self.assertIn("error_code = 'TIMEOUT'", query)
        self.assertIn("WHERE status = 'running'", query)
        self.assertIn("deadline_at <= NOW()", query)
        self.assertIn("RETURNING id, submission_id", query)

    def test_failed_run_persists_code_with_private_details(self) -> None:
        run_id = uuid4()
        connection = _Connection()
        database = Database("unused")

        with patch.object(database, "_connect", return_value=connection):
            database.mark_failed(
                run_id,
                "RuntimeError: CUDA out of memory",
                "private output",
            )

        query, params = connection.queries[0]
        self.assertIn("error_code = %s", query)
        self.assertEqual(
            params,
            (
                "RuntimeError: CUDA out of memory",
                "OUT_OF_MEMORY",
                "private output",
                run_id,
            ),
        )

    def test_log_purge_clears_only_expired_private_details(self) -> None:
        connection = _Connection([_Result(rowcount=3)])
        database = Database("unused")

        with patch.object(database, "_connect", return_value=connection):
            purged = database.purge_expired_run_logs(24 * 60 * 60)

        self.assertEqual(purged, 3)
        query, params = connection.queries[0]
        self.assertIn("SET log_tail = '', error = NULL", query)
        self.assertNotIn("error_code", query)
        self.assertNotIn("metrics", query)
        self.assertIn("finished_at < NOW()", query)
        self.assertEqual(params, (24 * 60 * 60,))

class EvaluatorDeadlineTests(unittest.TestCase):
    def test_modal_dispatch_uses_the_persisted_deadline_duration(self) -> None:
        tier = TIER_BY_ID["hard"]
        dataset = tier.datasets[0]
        run_id = uuid4()
        database = Mock()
        call = Mock(object_id="fc-test")
        call.get.return_value = {
            "returncode": 0,
            "timed_out": False,
            "log_tail": "",
            "benchmark_result": {
                "score": {"mean_exact_accuracy": 0.5},
            },
        }
        function = Mock()
        function.spawn.return_value = call
        settings = SimpleNamespace(
            evaluator_backend="modal",
            modal_app_name="test-app",
            modal_function_name="evaluate_submission",
        )

        with patch(
            "service.evaluator.modal.Function.from_name",
            return_value=function,
        ):
            evaluate_run(
                database=database,
                settings=settings,
                run_id=run_id,
                source="SUBMISSION = object()",
                tier=tier,
                dataset=dataset,
            )

        self.assertEqual(tier.run_deadline_seconds, 6180)
        database.mark_running.assert_called_once_with(run_id, "fc-test", 6180)
        call.get.assert_called_once_with(timeout=6180)
        database.mark_succeeded.assert_called_once()


class DeadlineWatchdogLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_backfills_and_expires_before_polling(self) -> None:
        watchdog = AsyncMock()
        with (
            patch.object(service_app.database, "initialize") as initialize,
            patch.object(
                service_app.database,
                "backfill_run_deadlines",
                return_value=1,
            ) as backfill,
            patch.object(
                service_app.database,
                "fail_expired_runs",
                return_value=[],
            ) as fail_expired,
            patch.object(
                service_app.database,
                "purge_expired_run_logs",
                return_value=0,
            ) as purge_logs,
            patch.object(
                service_app,
                "_run_deadline_watchdog",
                watchdog,
            ),
        ):
            async with service_app.lifespan(service_app.app):
                await asyncio.sleep(0)

        initialize.assert_called_once_with()
        backfill.assert_called_once_with(
            {"easy": 450, "medium": 1260, "hard": 6180}
        )
        fail_expired.assert_called_once_with()
        purge_logs.assert_called_once_with(24 * 60 * 60)
        watchdog.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()

"""Dispatch submissions either locally or to the deployed Modal H100 function."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import UUID, uuid4

import modal

from .config import Settings
from .db import Database
from .metrics import validate_structured_metrics
from .tiers import ComputeTier, DatasetOption


RESULT_PREFIX = "RESULT_JSON="
logger = logging.getLogger(__name__)


def _extract_result(output: str) -> dict:
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line.removeprefix(RESULT_PREFIX))
    raise ValueError("evaluator completed without a RESULT_JSON record")


def _evaluate_local(
    source: str,
    settings: Settings,
    tier: ComputeTier,
    dataset: DatasetOption,
) -> tuple[dict, str]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as file:
        file.write(source)
        submission_path = file.name
    try:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmark.runner",
                "--manifest",
                str(Path(settings.benchmark_manifest_dir) / dataset.manifest_filename),
                "--submission-file",
                submission_path,
                "--include-structured-metrics",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=tier.evaluator_timeout_seconds,
            check=False,
        )
        output = process.stdout
        if process.returncode != 0:
            raise RuntimeError(
                f"local evaluator exited with code {process.returncode}\n{output[-8000:]}"
            )
        return _extract_result(output), output[-40000:]
    finally:
        os.unlink(submission_path)


def _separate_structured_metrics(result: dict) -> tuple[dict, list[dict] | None]:
    result = dict(result)
    payload = result.pop("structured_metrics", None)
    if payload is None:
        return result, None
    try:
        metrics = validate_structured_metrics(payload, result)
    except (TypeError, ValueError, OverflowError):
        logger.warning("discarding invalid structured metrics from evaluator")
        return result, None
    return result, metrics


def evaluate_run(
    *,
    database: Database,
    settings: Settings,
    run_id: UUID,
    source: str,
    tier: ComputeTier,
    dataset: DatasetOption,
) -> None:
    log_tail = ""
    try:
        if settings.evaluator_backend == "modal":
            function = modal.Function.from_name(
                settings.modal_app_name,
                settings.modal_function_name,
            )
            call = function.spawn(
                submission_source=source,
                manifest_filename=dataset.manifest_filename,
                timeout_seconds=tier.evaluator_timeout_seconds,
            )
            call_id = call.object_id
            database.mark_running(run_id, call_id, tier.run_deadline_seconds)
            payload = call.get(timeout=tier.run_deadline_seconds)
            log_tail = str(payload.get("log_tail", ""))
            if payload.get("returncode") != 0 or payload.get("timed_out"):
                raise RuntimeError(
                    f"Modal evaluator failed (returncode={payload.get('returncode')}, "
                    f"timed_out={payload.get('timed_out')})"
                )
            result = payload.get("benchmark_result")
            if not isinstance(result, dict):
                raise ValueError("Modal evaluator returned no benchmark_result")
        else:
            call_id = f"local-{uuid4()}"
            database.mark_running(run_id, call_id, tier.run_deadline_seconds)
            result, log_tail = _evaluate_local(source, settings, tier, dataset)
        result, metrics = _separate_structured_metrics(result)
        database.mark_succeeded(run_id, result, log_tail, metrics)
    except Exception as exc:
        database.mark_failed(run_id, f"{type(exc).__name__}: {exc}", log_tail)

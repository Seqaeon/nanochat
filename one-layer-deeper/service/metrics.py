"""Validation and serialization for participant-downloadable metrics."""

from __future__ import annotations

import json
import math
from typing import Any


MAX_STRUCTURED_METRIC_RECORDS = 300
MAX_STRUCTURED_METRICS_BYTES = 128 * 1024
TRAIN_KEYS = frozenset(
    ("type", "seed", "step", "elapsed_seconds", "loss", "exact_accuracy")
)
EVALUATION_KEYS = frozenset(
    ("type", "seed", "split", "loss", "exact_accuracy")
)
SUMMARY_KEYS = frozenset(
    ("type", "completed_steps", "training_seconds", "mean_exact_accuracy")
)


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"structured metric {name} must be an integer >= {minimum}")
    return value


def _number(
    value: Any,
    name: str,
    *,
    minimum: float,
    maximum: float,
    digits: int,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"structured metric {name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(
            f"structured metric {name} must be finite and between "
            f"{minimum} and {maximum}"
        )
    return round(value, digits)


def _require_keys(record: dict, expected: frozenset[str]) -> None:
    if frozenset(record) != expected:
        raise ValueError("structured metric record has unexpected fields")


def _seed_results(result: dict) -> dict[int, dict]:
    raw_seeds = result.get("seeds")
    if not isinstance(raw_seeds, list):
        raise ValueError("benchmark result has no seed results")
    seeds: dict[int, dict] = {}
    for item in raw_seeds:
        if not isinstance(item, dict):
            raise ValueError("benchmark seed result must be an object")
        seed = _integer(item.get("seed"), "seed")
        if seed in seeds:
            raise ValueError("benchmark result contains duplicate seeds")
        seeds[seed] = item
    return seeds


def validate_structured_metrics(payload: Any, result: dict) -> list[dict[str, Any]]:
    """Return a normalized allowlisted payload or raise ValueError."""

    if not isinstance(payload, list):
        raise ValueError("structured metrics must be a list")
    if len(payload) > MAX_STRUCTURED_METRIC_RECORDS:
        raise ValueError("structured metrics contain too many records")

    seed_results = _seed_results(result)
    normalized: list[dict[str, Any]] = []
    last_step_by_seed: dict[int, int] = {}
    evaluation_records: set[tuple[int, str]] = set()
    summary_count = 0

    for raw_record in payload:
        if not isinstance(raw_record, dict):
            raise ValueError("structured metric record must be an object")
        record_type = raw_record.get("type")

        if record_type == "train":
            _require_keys(raw_record, TRAIN_KEYS)
            seed = _integer(raw_record["seed"], "seed")
            seed_result = seed_results.get(seed)
            if seed_result is None:
                raise ValueError("structured metric contains an unknown seed")
            step = _integer(raw_record["step"], "step", minimum=1)
            completed_steps = _integer(
                seed_result.get("completed_training_steps"),
                "completed_training_steps",
            )
            if step > completed_steps or step <= last_step_by_seed.get(seed, 0):
                raise ValueError("structured training steps must be ordered and completed")
            elapsed = _number(
                raw_record["elapsed_seconds"],
                "elapsed_seconds",
                minimum=0.0,
                maximum=1_000_000.0,
                digits=1,
            )
            training_seconds = _number(
                seed_result.get("training_seconds"),
                "training_seconds",
                minimum=0.0,
                maximum=1_000_000.0,
                digits=1,
            )
            if elapsed > training_seconds + 0.1:
                raise ValueError("structured metric elapsed time exceeds training time")
            last_step_by_seed[seed] = step
            normalized.append(
                {
                    "type": "train",
                    "seed": seed,
                    "step": step,
                    "elapsed_seconds": elapsed,
                    "loss": _number(
                        raw_record["loss"],
                        "loss",
                        minimum=-1_000_000.0,
                        maximum=1_000_000.0,
                        digits=3,
                    ),
                    "exact_accuracy": _number(
                        raw_record["exact_accuracy"],
                        "exact_accuracy",
                        minimum=0.0,
                        maximum=1.0,
                        digits=3,
                    ),
                }
            )
            continue

        if record_type == "evaluation":
            _require_keys(raw_record, EVALUATION_KEYS)
            seed = _integer(raw_record["seed"], "seed")
            seed_result = seed_results.get(seed)
            if seed_result is None:
                raise ValueError("structured metric contains an unknown seed")
            split = raw_record["split"]
            evaluation = seed_result.get("evaluation")
            if not isinstance(split, str) or not isinstance(evaluation, dict):
                raise ValueError("structured evaluation metric is invalid")
            expected = evaluation.get(split)
            if not isinstance(expected, dict):
                raise ValueError("structured metric contains an unknown split")
            key = (seed, split)
            if key in evaluation_records:
                raise ValueError("structured metrics contain duplicate evaluation records")
            evaluation_records.add(key)
            loss = _number(
                raw_record["loss"],
                "loss",
                minimum=0.0,
                maximum=1_000_000.0,
                digits=3,
            )
            accuracy = _number(
                raw_record["exact_accuracy"],
                "exact_accuracy",
                minimum=0.0,
                maximum=1.0,
                digits=3,
            )
            if loss != round(float(expected.get("loss")), 3) or accuracy != round(
                float(expected.get("exact_accuracy")), 3
            ):
                raise ValueError("structured evaluation metrics do not match the result")
            normalized.append(
                {
                    "type": "evaluation",
                    "seed": seed,
                    "split": split,
                    "loss": loss,
                    "exact_accuracy": accuracy,
                }
            )
            continue

        if record_type == "summary":
            _require_keys(raw_record, SUMMARY_KEYS)
            summary_count += 1
            if summary_count != 1:
                raise ValueError("structured metrics contain duplicate summaries")
            completed_steps = _integer(
                raw_record["completed_steps"],
                "completed_steps",
            )
            expected_steps = sum(
                _integer(item.get("completed_training_steps"), "completed_training_steps")
                for item in seed_results.values()
            )
            training_seconds = _number(
                raw_record["training_seconds"],
                "training_seconds",
                minimum=0.0,
                maximum=1_000_000.0,
                digits=1,
            )
            expected_seconds = round(
                sum(float(item.get("training_seconds")) for item in seed_results.values()),
                1,
            )
            mean_accuracy = _number(
                raw_record["mean_exact_accuracy"],
                "mean_exact_accuracy",
                minimum=0.0,
                maximum=1.0,
                digits=3,
            )
            score = result.get("score")
            if not isinstance(score, dict):
                raise ValueError("benchmark result has no score")
            expected_accuracy = round(float(score.get("mean_exact_accuracy")), 3)
            if (
                completed_steps != expected_steps
                or training_seconds != expected_seconds
                or mean_accuracy != expected_accuracy
            ):
                raise ValueError("structured summary does not match the result")
            normalized.append(
                {
                    "type": "summary",
                    "completed_steps": completed_steps,
                    "training_seconds": training_seconds,
                    "mean_exact_accuracy": mean_accuracy,
                }
            )
            continue

        raise ValueError("structured metric record has an unknown type")

    expected_evaluations = {
        (seed, split)
        for seed, item in seed_results.items()
        for split in (item.get("evaluation") or {})
    }
    if summary_count != 1 or evaluation_records != expected_evaluations:
        raise ValueError("structured metrics are incomplete")
    if not normalized or normalized[-1]["type"] != "summary":
        raise ValueError("structured metric summary must be last")
    if len(metrics_to_jsonl(normalized).encode("utf-8")) > MAX_STRUCTURED_METRICS_BYTES:
        raise ValueError("structured metrics exceed the size limit")
    return normalized


def metrics_to_jsonl(metrics: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        for record in metrics
    )

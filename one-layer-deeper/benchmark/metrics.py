"""Bounded evaluator-owned structured metric recording."""

from __future__ import annotations

import math
from typing import Any


MAX_TRAIN_METRIC_RECORDS = 256


def _finite_rounded(value: float, digits: int) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("structured metrics must be finite")
    return round(value, digits)


class MetricRecorder:
    """Collect a bounded metric history without exposing a global write target."""

    def __init__(self, max_train_records: int = MAX_TRAIN_METRIC_RECORDS) -> None:
        if max_train_records < 2:
            raise ValueError("max_train_records must be at least two")
        self._max_train_records = max_train_records
        self._train_records: list[tuple[int, dict[str, Any]]] = []
        self._other_records: list[dict[str, Any]] = []
        self._train_seen = 0
        self._stride = 1
        self._last_train_record: tuple[int, dict[str, Any]] | None = None

    def record_training(
        self,
        *,
        seed: int,
        step: int,
        elapsed_seconds: float,
        loss: float,
        exact_accuracy: float,
    ) -> None:
        self._train_seen += 1
        record = {
            "type": "train",
            "seed": int(seed),
            "step": int(step),
            "elapsed_seconds": _finite_rounded(elapsed_seconds, 1),
            "loss": _finite_rounded(loss, 3),
            "exact_accuracy": _finite_rounded(exact_accuracy, 3),
        }
        indexed = (self._train_seen, record)
        self._last_train_record = indexed
        if self._train_seen == 1 or self._train_seen % self._stride == 0:
            self._train_records.append(indexed)
        while len(self._train_records) > self._max_train_records:
            self._stride *= 2
            self._train_records = [
                item
                for item in self._train_records
                if item[0] == 1 or item[0] % self._stride == 0
            ]

    def record_evaluation(
        self,
        *,
        seed: int,
        split: str,
        loss: float,
        exact_accuracy: float,
    ) -> None:
        self._other_records.append(
            {
                "type": "evaluation",
                "seed": int(seed),
                "split": split,
                "loss": _finite_rounded(loss, 3),
                "exact_accuracy": _finite_rounded(exact_accuracy, 3),
            }
        )

    def record_summary(
        self,
        *,
        completed_steps: int,
        training_seconds: float,
        mean_exact_accuracy: float,
    ) -> None:
        self._other_records.append(
            {
                "type": "summary",
                "completed_steps": int(completed_steps),
                "training_seconds": _finite_rounded(training_seconds, 1),
                "mean_exact_accuracy": _finite_rounded(mean_exact_accuracy, 3),
            }
        )

    def snapshot(self) -> list[dict[str, Any]]:
        train_records = list(self._train_records)
        if self._last_train_record is not None and (
            not train_records
            or train_records[-1][0] != self._last_train_record[0]
        ):
            if len(train_records) >= self._max_train_records:
                train_records.pop(-1)
            train_records.append(self._last_train_record)
        return [
            *(record.copy() for _, record in train_records),
            *(record.copy() for record in self._other_records),
        ]

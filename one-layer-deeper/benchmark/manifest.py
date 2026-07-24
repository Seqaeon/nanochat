"""Validated, evaluator-owned benchmark manifests."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from data import DataConfig


@dataclass(frozen=True)
class RuntimeSpec:
    device: str
    dtype: str
    amp: bool
    compile: bool
    total_training_time_seconds: float
    max_steps: int
    seeds: tuple[int, ...]
    grad_clip: float | None
    log_every: int


@dataclass(frozen=True)
class ModelStateSpec:
    maximum_elements: int


@dataclass(frozen=True)
class BenchmarkManifest:
    name: str
    data: DataConfig
    runtime: RuntimeSpec
    model_state: ModelStateSpec


def _require_keys(value: dict[str, Any], expected: set[str], *, where: str) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing or extra:
        pieces = []
        if missing:
            pieces.append(f"missing={sorted(missing)}")
        if extra:
            pieces.append(f"unknown={sorted(extra)}")
        raise ValueError(f"invalid {where}: {', '.join(pieces)}")


def load_manifest(path: str | Path) -> BenchmarkManifest:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text())
    _require_keys(
        payload,
        {
            "name",
            "data",
            "runtime",
            "model_state",
        },
        where="manifest",
    )

    runtime_payload = payload["runtime"]
    _require_keys(
        runtime_payload,
        {
            "device",
            "dtype",
            "amp",
            "compile",
            "total_training_time_seconds",
            "max_steps",
            "seeds",
            "grad_clip",
            "log_every",
        },
        where="runtime",
    )
    runtime = RuntimeSpec(
        device=str(runtime_payload["device"]),
        dtype=str(runtime_payload["dtype"]),
        amp=bool(runtime_payload["amp"]),
        compile=bool(runtime_payload["compile"]),
        total_training_time_seconds=float(
            runtime_payload["total_training_time_seconds"]
        ),
        max_steps=int(runtime_payload["max_steps"]),
        seeds=tuple(int(seed) for seed in runtime_payload["seeds"]),
        grad_clip=(
            None
            if runtime_payload["grad_clip"] is None
            else float(runtime_payload["grad_clip"])
        ),
        log_every=int(runtime_payload["log_every"]),
    )
    if runtime.dtype not in {"float32", "bfloat16"}:
        raise ValueError("runtime.dtype must be float32 or bfloat16")
    if (
        runtime.total_training_time_seconds <= 0
        or runtime.max_steps < 1
        or runtime.log_every < 1
    ):
        raise ValueError(
            "total_training_time_seconds, max_steps, and log_every must be positive"
        )
    if not runtime.seeds or len(set(runtime.seeds)) != len(runtime.seeds):
        raise ValueError("runtime.seeds must be a non-empty list of unique integers")
    if runtime.grad_clip is not None and runtime.grad_clip <= 0:
        raise ValueError("runtime.grad_clip must be positive when provided")

    state_payload = payload["model_state"]
    _require_keys(
        state_payload,
        {"maximum_elements"},
        where="model_state",
    )
    model_state = ModelStateSpec(
        maximum_elements=int(state_payload["maximum_elements"])
    )
    if model_state.maximum_elements < 1:
        raise ValueError("model state maximum must be positive")

    data = DataConfig(**payload["data"])

    return BenchmarkManifest(
        name=str(payload["name"]),
        data=data,
        runtime=runtime,
        model_state=model_state,
    )

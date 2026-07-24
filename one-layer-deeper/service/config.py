"""Environment-backed service configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    database_url: str
    evaluator_backend: str
    benchmark_manifest_dir: str
    modal_app_name: str
    modal_function_name: str
    max_submission_bytes: int
    public_url: str
    github_client_id: str
    github_client_secret: str

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            database_url=os.environ.get(
                "DATABASE_URL",
                "postgresql://benchmark:benchmark@localhost:55432/benchmark",
            ),
            evaluator_backend=os.environ.get("EVALUATOR_BACKEND", "local"),
            benchmark_manifest_dir=os.environ.get(
                "BENCHMARK_MANIFEST_DIR",
                "benchmark/manifests",
            ),
            modal_app_name=os.environ.get("MODAL_APP_NAME", "one-layer-benchmark-runner"),
            modal_function_name=os.environ.get(
                "MODAL_FUNCTION_NAME",
                "evaluate_submission",
            ),
            max_submission_bytes=int(os.environ.get("MAX_SUBMISSION_BYTES", str(256 * 1024))),
            public_url=os.environ.get(
                "PUBLIC_URL",
                "http://127.0.0.1:8000",
            ).rstrip("/"),
            github_client_id=os.environ.get("GITHUB_CLIENT_ID", ""),
            github_client_secret=os.environ.get("GITHUB_CLIENT_SECRET", ""),
        )
        if settings.evaluator_backend not in {"local", "modal"}:
            raise ValueError("EVALUATOR_BACKEND must be local or modal")
        if settings.max_submission_bytes < 1:
            raise ValueError("service limits must be positive")
        return settings

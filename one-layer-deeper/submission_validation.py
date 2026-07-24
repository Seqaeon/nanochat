"""Pure source-policy checks shared by the CLI, service, and evaluator."""

from __future__ import annotations

import ast


FORBIDDEN_SUBMISSION_IMPORTS = {"model", "optim"}


def validate_submission_source(
    filename: str,
    source: str,
    max_bytes: int,
    *,
    required_filename: str | None = "submission.py",
) -> str:
    basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if required_filename is not None and basename != required_filename:
        raise ValueError(f"submission file must be named {required_filename}")
    if not basename.lower().endswith(".py"):
        raise ValueError("submit exactly one .py file")
    if not source.strip():
        raise ValueError("submission file is empty")
    if len(source.encode("utf-8")) > max_bytes:
        raise ValueError(f"submission exceeds the {max_bytes // 1024} KiB limit")
    try:
        tree = ast.parse(source, filename=basename)
    except SyntaxError as exc:
        raise ValueError(f"submission is not valid Python: {exc.msg}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported = [node.module]
        else:
            continue
        for name in imported:
            root = name.partition(".")[0]
            if root in FORBIDDEN_SUBMISSION_IMPORTS:
                raise ValueError(
                    f"submission must be self-contained and may not import {root}"
                )
    return basename

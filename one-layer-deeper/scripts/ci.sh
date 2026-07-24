#!/usr/bin/env bash

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

uv_bin=${UV_BIN:-uv}
if ! command -v "$uv_bin" >/dev/null 2>&1; then
  echo "CI requires uv: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

run_cli_and_package() {
  "$uv_bin" run python -m unittest tests.test_cli tests.test_scores tests.test_release
  "$uv_bin" run one-layer --help
  "$uv_bin" run one-layer validate submissions/baseline_adamw/submission.py

  ci_dist_dir=$(mktemp -d)
  trap 'rm -rf -- "$ci_dist_dir"' EXIT
  "$uv_bin" build --out-dir "$ci_dist_dir"
  "$uv_bin" run python - "$ci_dist_dir" <<'PY'
from pathlib import Path
import sys
from zipfile import ZipFile

wheel = next(Path(sys.argv[1]).glob("*.whl"))
with ZipFile(wheel) as archive:
    names = set(archive.namelist())
assert "client/cli.py" in names
assert "client/catalog.py" in names
assert "benchmark/manifests/smoke_cpu.json" in names
assert any(name.endswith(".dist-info/licenses/LICENSE") for name in names)
assert "service/app.py" in names
assert "service/static/style.css" in names
assert "service/static/favicon.svg" in names
PY
}

run_benchmark() {
  "$uv_bin" run python -m unittest discover -s tests
  "$uv_bin" run python -m benchmark.runner \
    --manifest benchmark/manifests/smoke_cpu.json \
    --submission-file submissions/baseline_adamw/submission.py
}

case "${1:-all}" in
  all)
    "$uv_bin" sync --locked
    run_cli_and_package
    run_benchmark
    ;;
  cli-and-package)
    "$uv_bin" sync --locked
    run_cli_and_package
    ;;
  benchmark)
    "$uv_bin" sync --locked
    run_benchmark
    ;;
  *)
    echo "usage: $0 [all|cli-and-package|benchmark]" >&2
    exit 2
    ;;
esac

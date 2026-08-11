#!/usr/bin/env bash
# Build the canonical measurement env (.venv23) from requirements-venv23.txt.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -e .venv23 ]; then
  echo "refusing to overwrite existing .venv23 — remove it first to rebuild" >&2
  exit 1
fi

uv venv --python 3.12 .venv23
VIRTUAL_ENV="$PWD/.venv23" uv pip install -r requirements-venv23.txt

echo
echo "done. Install the reinfors release wheel into it:"
echo "  cd ../reinfors && VIRTUAL_ENV=../reinfors-benchmarks/.venv23 uvx maturin develop --release -m crates/reinfors-py/Cargo.toml"
echo "then verify with: .venv23/bin/python benchmarks/openspiel/preflight.py --expect-tag <tag>"

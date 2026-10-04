#!/usr/bin/env bash
# Recompile the paper without make.
#   ./build.sh            compile main.pdf
#   ./build.sh --figures  regenerate figures first
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v tectonic >/dev/null; then
    echo "tectonic not found: https://tectonic-typesetting.github.io" >&2
    exit 1
fi

if [[ "${1:-}" == "--figures" ]] || ! ls figures/*.pdf >/dev/null 2>&1; then
    uv run --project .. python make_figures.py
fi

tectonic --keep-logs main.tex
echo "-> $(pwd)/main.pdf"

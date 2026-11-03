#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi

if [ -f .venv/Scripts/activate ]; then
    VENV_BIN=.venv/Scripts      # Windows (Git Bash, MSYS)
else
    VENV_BIN=.venv/bin          # Linux, macOS
fi

"$VENV_BIN/python" -m pip install -e .

echo -e "ACTIVATE VENV:\nsource $VENV_BIN/activate"

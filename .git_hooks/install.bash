#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" &>/dev/null && pwd)"
REPOSITORY_DIR="$(dirname "${SCRIPT_DIR}")"

# Install pre-commit if not detected
if [[ ! -x "$(command -v pre-commit)" ]]; then
    pip install --user pre-commit
fi

# Install local git hooks for this repository
cd "${REPOSITORY_DIR}"
pre-commit install --install-hooks --config "${REPOSITORY_DIR}/.pre-commit-config.yaml"
cd - >/dev/null

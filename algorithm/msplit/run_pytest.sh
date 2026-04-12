#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_root="$(cd "${repo_dir}/../.." && pwd)"
venv_python="${workspace_root}/.venv/bin/python"

if [[ ! -x "${venv_python}" ]]; then
    echo "expected venv python at ${venv_python}" >&2
    exit 1
fi

exec "${venv_python}" -m pytest "$@"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PATH="${PROJECT_ROOT}/.venv"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: no se encontró Python en '${PYTHON_BIN}'." >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c 'import sys; sys.exit(sys.version_info < (3, 14))'; then
  echo "Error: se requiere Python 3.14 o superior." >&2
  exit 1
fi

directories=(
  "config"
  "data/raw"
  "data/interim"
  "data/processed"
  "data/external"
  "models/trained"
  "models/checkpoints"
  "reports/figures"
  "src"
  "tests"
  "docs"
  "scripts"
)

for directory in "${directories[@]}"; do
  mkdir -p "${PROJECT_ROOT}/${directory}"
done

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
fi

"${VENV_PATH}/bin/python" -m pip install --requirement "${PROJECT_ROOT}/requirements.txt"

for directory in "${directories[@]}"; do
  if [[ ! -d "${PROJECT_ROOT}/${directory}" ]]; then
    echo "Error: no se pudo crear '${directory}'." >&2
    exit 1
  fi
done

echo "Entorno listo en ${VENV_PATH}; estructura del proyecto verificada."

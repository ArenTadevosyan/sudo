#!/usr/bin/env bash
set -euo pipefail

USE_VENV="${USE_VENV:-1}"

if [[ "$USE_VENV" == "1" ]]; then
  if python3 -m venv .venv; then
    source .venv/bin/activate
  else
    echo "Could not create .venv. Continuing with the current Python environment."
  fi
else
  echo "USE_VENV=0, using the current Python environment."
fi

python -m pip install --upgrade pip
pip install -r requirements.txt

cargo test
python -m unittest discover -s tests
python tools/export_dataset.py

echo "Language model: python train_tiny_lm.py --device cuda"
echo "Vision detector: python tools/prepare_vision_dataset.py init --classes person car && python train_vision.py --device cuda --amp"

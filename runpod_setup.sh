#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

cargo test
python -m unittest discover -s tests
python tools/export_dataset.py

echo "Ready. Train with: python train_tiny_lm.py --device cuda"


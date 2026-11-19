#!/usr/bin/env bash
# Install the sn-gamestate baseline (pinned) with its own Python 3.9 env.
set -euo pipefail
cd "$(dirname "$0")"

SNGS_COMMIT=1c958345067218297d221e45e1a6405f975f83e0
SNGS=third_party/sn-gamestate

command -v uv >/dev/null || pip install -q uv
[ -d "$SNGS/.git" ] || git clone -q https://github.com/SoccerNet/sn-gamestate.git "$SNGS"
git -C "$SNGS" checkout -q "$SNGS_COMMIT"

cd "$SNGS"
uv venv -q --python 3.9 --allow-existing
uv pip install -q -e .
uv run mim install -q mmcv==2.0.1
# The calibration plugin doesn't package pnlcalib, so put its folder on the path.
echo "$(pwd)/plugins/calibration" > .venv/lib/python3.9/site-packages/pnlcalib_path.pth
echo "Ready: $(pwd)/.venv"

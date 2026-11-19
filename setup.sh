#!/usr/bin/env bash
# Install the sn-gamestate baseline (pinned) into its own Python 3.9 env.
# sn-gamestate needs Python 3.9 (torch 1.13), whatever Python the machine has.
set -euo pipefail
cd "$(dirname "$0")"

SNGS_COMMIT=1c958345067218297d221e45e1a6405f975f83e0
SNGS=third_party/sn-gamestate

command -v uv >/dev/null || pip install -q uv
[ -d "$SNGS/.git" ] || git clone -q https://github.com/SoccerNet/sn-gamestate.git "$SNGS"
git -C "$SNGS" checkout -q "$SNGS_COMMIT"

cd "$SNGS"
unset UV_SYSTEM_PYTHON UV_PYTHON VIRTUAL_ENV  # if set (Colab may), these send installs to the system Python
uv python install 3.9
[ -x .venv/bin/python3.9 ] || { rm -rf .venv; uv venv -q --python 3.9 .venv; }
uv pip install -q --python .venv/bin/python -e . pip wheel "setuptools<70"
.venv/bin/mim install -q mmcv==2.0.1
# The calibration plugin doesn't package pnlcalib, so put its folder on the path.
echo "$(pwd)/plugins/calibration" > .venv/lib/python3.9/site-packages/pnlcalib_path.pth
.venv/bin/python -c "import sys, torch, sn_gamestate, pnlcalib; print('Ready: Python', sys.version.split()[0], 'torch', torch.__version__)"

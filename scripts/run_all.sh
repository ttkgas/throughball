#!/usr/bin/env bash
# The best run so far on SNGS-021, stages 1 to 4, skipping what's already done.
# Each stage's outputs are copied to SAVE_DIR (e.g. a mounted Drive folder) as soon as it finishes,
# since Colab drops idle runtimes and everything under outputs/ goes with them.
# Usage: bash scripts/run_all.sh [SAVE_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
SAVE=${1:-}
save() { [ -z "$SAVE" ] || { mkdir -p "$SAVE"; cp -r "outputs/stage$1" "$SAVE/"; echo "== stage $1 saved to $SAVE"; }; }
TL=third_party/sn-gamestate/.venv/bin
newest() { ls -t outputs/throughball-setup/*/*/states/*.pklz | head -1; }

[ -x $TL/tracklab ] || bash setup.sh
$TL/python scripts/get_clip.py SNGS-021
if [ ! -f outputs/stage1/021/cameras.json ]; then  # the 100-frame window we render
  MPLBACKEND=Agg $TL/tracklab --config-dir configs -cn throughball_setup
  $TL/python scripts/export_stage1.py "$(newest)" 021
fi
if [ ! -f outputs/stage1/021_full/cameras.json ]; then  # the whole clip, for the background
  MPLBACKEND=Agg $TL/tracklab --config-dir configs -cn throughball_setup dataset.nframes=-1
  $TL/python scripts/export_stage1.py "$(newest)" 021_full
fi
save 1

pip install -q gsplat ultralytics imageio-ffmpeg ninja
ls outputs/stage3/021/pov_*/cameras.json >/dev/null 2>&1 || python -u scripts/gaussians_pov.py --vid 021 --bg 021_full --fill 10
save 3

pip install -q -U diffusers transformers accelerate sentencepiece imageio-ffmpeg
[ -f outputs/stage4/021/compare.mp4 ] || python -u scripts/generate_pov.py --vid 021
save 4

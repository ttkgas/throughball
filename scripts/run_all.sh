#!/usr/bin/env bash
# The best run so far on one clip (SNGS-021 unless CLIP is set), stages 1 to 4, skipping what's already done.
# Each stage's outputs are copied to SAVE_DIR (e.g. a mounted Drive folder) as soon as it finishes,
# since Colab drops idle runtimes and everything under outputs/ goes with them.
# Usage: [CLIP=024] bash scripts/run_all.sh [SAVE_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
SAVE=${1:-}; C=${CLIP:-021}; V="dataset.vids_dict.valid=[SNGS-$C]"
save() {  # never creates SAVE_DIR's parent, so an unmounted Drive is reported, not shadowed
  [ -z "$SAVE" ] && return; mkdir "$SAVE" 2>/dev/null || true
  cp -r "outputs/$1" "$SAVE/" && echo "== $1 saved to $SAVE" || echo "!! $1 not saved: is $SAVE's parent mounted?"
}
TL=third_party/sn-gamestate/.venv/bin
newest() { ls -t outputs/throughball-setup/*/*/states/*.pklz | head -1; }

[ -x $TL/tracklab ] || bash setup.sh
$TL/python scripts/get_clip.py SNGS-$C
if [ ! -f outputs/stage1/$C/cameras.json ]; then  # the 100-frame window we render
  MPLBACKEND=Agg $TL/tracklab --config-dir configs -cn throughball_setup "$V"
  $TL/python scripts/export_stage1.py "$(newest)" $C
fi
if [ ! -f outputs/stage1/${C}_full/cameras.json ]; then  # the whole clip, for the background
  MPLBACKEND=Agg $TL/tracklab --config-dir configs -cn throughball_setup dataset.nframes=-1 "$V"
  $TL/python scripts/export_stage1.py "$(newest)" ${C}_full
fi
save stage1

pip install -q gsplat ultralytics imageio-ffmpeg ninja
ls outputs/stage3/$C/pov_*/cameras.json >/dev/null 2>&1 || python -u scripts/gaussians_pov.py --vid $C --bg ${C}_full --fill 10
save stage3

pip install -q -U diffusers transformers accelerate sentencepiece imageio-ffmpeg
[ -f outputs/stage4/$C/compare.mp4 ] || python -u scripts/generate_pov.py --vid $C
save stage4

# GS-HOTA of the full game state pipeline (roles, teams, jersey numbers) on the whole clip
mkdir -p outputs/gsr/$C
[ -f outputs/gsr/$C/log.txt ] && grep -q GS-HOTA outputs/gsr/$C/log.txt || \
  MPLBACKEND=Agg $TL/tracklab --config-dir configs -cn throughball_gsr "$V" 2>&1 | tee outputs/gsr/$C/log.txt
save gsr

#!/usr/bin/env bash
# Stage 3-4 with the background fit from other clips of the same match too (SNGS-021 is game 2,
# whose valid clips are SNGS-021 to SNGS-038; scripts/same_match.py lists them).
# Needs run_all.sh's stage 1 exports (021, 021_full). Outputs go to outputs/stage{1,3,4}/..., *_match.
# Usage: bash scripts/match_bg.sh [SAVE_DIR] [clips, default "024 028 033 037"]
set -euo pipefail
cd "$(dirname "$0")/.."
SAVE=${1:-}; CLIPS=${2:-"024 028 033 037"}
TL=third_party/sn-gamestate/.venv/bin
newest() { ls -t outputs/throughball-setup/*/*/states/*.pklz | head -1; }
save() { [ -z "$SAVE" ] && return; mkdir "$SAVE" 2>/dev/null || true; mkdir -p "$SAVE/$(dirname "$1")"
  cp -r "outputs/$1" "$SAVE/$(dirname "$1")/" && echo "== $1 saved to $SAVE" || echo "!! $1 not saved: is $SAVE's parent mounted?"; }
BG=021_full
for c in $CLIPS; do
  $TL/python scripts/get_clip.py SNGS-$c
  if [ ! -f outputs/stage1/$c/cameras.json ]; then
    MPLBACKEND=Agg $TL/tracklab --config-dir configs -cn throughball_setup dataset.nframes=-1 "dataset.vids_dict.valid=[SNGS-$c]"
    $TL/python scripts/export_stage1.py "$(newest)" $c
    save stage1/$c
  fi
  BG=$BG,$c
done
ls outputs/stage3/021_match/pov_*/cameras.json >/dev/null 2>&1 || python -u scripts/gaussians_pov.py --vid 021 --bg $BG --fill 10 --out 021_match
save stage3/021_match
mkdir -p outputs/stage4/021_match; [ -f outputs/stage4/021/baseline.mp4 ] && cp -n outputs/stage4/021/baseline.mp4 outputs/stage4/021_match/
[ -f outputs/stage4/021_match/compare.mp4 ] || python -u scripts/generate_pov.py --vid 021_match --reuse-baseline
save stage4/021_match

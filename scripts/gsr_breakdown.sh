#!/usr/bin/env bash
# Re-score the saved GS-HOTA run with parts of the metric turned off, to see what costs the most:
# all attributes, without jersey numbers, without any attribute (pitch tracking only), and plain
# HOTA on image boxes. Evaluation only, from the state run_all.sh saved. Writes outputs/gsr/breakdown.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
TL=third_party/sn-gamestate/.venv/bin
S=$(ls -t "$PWD"/outputs/throughball-gsr/*/*/states/*.pklz | head -1)
run() {  # name, overrides
  r=$(MPLBACKEND=Agg $TL/tracklab --config-dir configs -cn throughball_gsr "pipeline=[]" state.load_file="$S" state.save_file=null "${@:2}" 2>&1 |
      grep -a -A1 '^HOTA:' | tail -1 | awk '{print "HOTA", $2, "DetA", $3, "AssA", $4}')
  echo "$1: $r" | tee -a outputs/gsr/breakdown.txt
}
: > outputs/gsr/breakdown.txt
E=dataset.track_eval
run "pitch, roles+teams+jerseys (GS-HOTA)"
run "pitch, roles+teams" $E.USE_JERSEY_NUMBERS=False
run "pitch, no attributes" $E.USE_JERSEY_NUMBERS=False $E.USE_TEAMS=False $E.USE_ROLES=False
run "image boxes, no attributes (HOTA)" $E.EVAL_SPACE=image $E.USE_JERSEY_NUMBERS=False $E.USE_TEAMS=False $E.USE_ROLES=False

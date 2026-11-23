"""List the SoccerNet GSR clips that come from the same match as a given clip.

Reads only the start of each clip's Labels-GameState.json inside valid.zip on Hugging Face
(where the "info" block sits), so it downloads a few MB. Other clips of the same match show the
same stadium and crowd from other moments, which is what stage 3 can borrow for unseen stands.
Usage: python scripts/same_match.py [SNGS-021] [valid]
"""
import json
import re
import sys
import zipfile

from huggingface_hub import HfFileSystem

clip = sys.argv[1] if len(sys.argv) > 1 else "SNGS-021"
split = sys.argv[2] if len(sys.argv) > 2 else "valid"
zf = zipfile.ZipFile(HfFileSystem().open(f"datasets/SoccerNet/SN-GSR-2024/{split}.zip", "rb", block_size=2**20))

infos = {}
for m in zf.infolist():
    if m.filename.endswith("Labels-GameState.json"):
        head = zf.open(m).read(4000).decode(errors="ignore")
        block = re.search(r'"info"\s*:\s*(\{[^{}]*\})', head)
        infos[m.filename.split("/")[-2]] = json.loads(block.group(1)) if block else {}

game = infos.get(clip, {}).get("game_id")
print(f"{clip}: {infos.get(clip)}")
for name, info in sorted(infos.items()):
    if name != clip and game is not None and info.get("game_id") == game:
        print(name, {k: info.get(k) for k in ("game_time_start", "game_time_stop", "action_class", "visibility")})
print(f"{len(infos)} clips read; game_id {game}")

"""Fetch one free SoccerNet GSR validation clip from Hugging Face.

The public, ungated copy lives at huggingface.co/datasets/SoccerNet/SN-GSR-2024
(the old KAUST download now answers 401). valid.zip is 11 GB, so this reads
only the one clip's files out of it with ranged requests instead of
downloading the whole zip. Usage: python scripts/get_clip.py [SNGS-021]
"""
import shutil
import sys
import zipfile
from pathlib import Path

from huggingface_hub import HfFileSystem

REPO_FILE = "datasets/SoccerNet/SN-GSR-2024/valid.zip"

clip = sys.argv[1] if len(sys.argv) > 1 else "SNGS-021"
out = Path("data/SoccerNetGS/valid") / clip

if (out / "Labels-GameState.json").exists():
    print(f"{clip} is already in {out}")
    sys.exit(0)

try:
    remote = HfFileSystem().open(REPO_FILE, "rb", block_size=8 * 2**20)
    zf = zipfile.ZipFile(remote)
except Exception as e:
    sys.exit(
        f"Could not open {REPO_FILE} on Hugging Face: {e}\n"
        "The download was refused or failed. It needs no login; if this keeps "
        "happening, check the dataset page at "
        "https://huggingface.co/datasets/SoccerNet/SN-GSR-2024"
    )

# Members may sit at "SNGS-021/..." or under a top folder like "valid/SNGS-021/...".
members = [m for m in zf.infolist() if f"{clip}/" in m.filename and not m.is_dir()]
if not members:
    sys.exit(f"{clip} is not in valid.zip. First entries: {zf.namelist()[:5]}")

print(f"Fetching {len(members)} files for {clip}...")
for i, m in enumerate(members, 1):
    rel = m.filename.split(f"{clip}/", 1)[1]
    dest = out / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(m) as src, open(dest, "wb") as dst:
        shutil.copyfileobj(src, dst)
    if i % 100 == 0:
        print(f"  {i}/{len(members)}")
print(f"{clip}: {len(members)} files in {out}")

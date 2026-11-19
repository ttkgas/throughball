"""Download the free SoccerNet GSR validation split and unpack one clip."""
import sys
import zipfile
from pathlib import Path

from SoccerNet.Downloader import SoccerNetDownloader

clip = sys.argv[1] if len(sys.argv) > 1 else "SNGS-021"
root = Path("data/SoccerNetGS")

SoccerNetDownloader(LocalDirectory=str(root)).downloadDataTask(task="gamestate-2024", split=["valid"])
with zipfile.ZipFile(root / "gamestate-2024" / "valid.zip") as zf:
    members = [m for m in zf.namelist() if m.startswith(f"{clip}/")]
    assert members, f"{clip} not in valid.zip. First entries: {zf.namelist()[:5]}"
    zf.extractall(root / "valid", members)
print(f"{clip}: {len(members)} files in {root / 'valid' / clip}")

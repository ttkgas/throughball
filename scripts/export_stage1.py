"""Export stage 1 (calibration + tracking) from a tracklab state file to plain files.

Writes, per video, under outputs/stage1/<video_id>/:
  cameras.json  one entry per frame: PnLCalib camera (focal lengths, principal point,
                position in meters, rotation matrix, pan/tilt/roll), or null if calibration failed
  players.csv   one row per tracked detection: frame, track_id, pitch x/y in meters
                (bottom-middle of the box, pitch centre at 0,0), and the image box
These are what stage 3 (dynamic 3D Gaussians) reads, so it never needs tracklab.
Usage: python scripts/export_stage1.py [path/to/state.pklz] [name]  (default: newest under outputs/; name: output folder, default the video id)
"""
import glob
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

state = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("outputs/**/*.pklz", recursive=True))[-1]
zf = zipfile.ZipFile(state)
for name in zf.namelist():
    if not name.endswith("_image.pkl"):
        continue
    vid = name[: -len("_image.pkl")]
    images = pd.read_pickle(zf.open(name)).sort_values("frame")
    dets = pd.read_pickle(zf.open(f"{vid}.pkl"))
    out = Path("outputs/stage1") / (sys.argv[2] if len(sys.argv) > 2 else str(images["video_id"].iloc[0]))
    out.mkdir(parents=True, exist_ok=True)

    cams = [{"frame": int(r.frame), "image_id": str(r.Index), "camera": r.parameters or None}
            for r in images.itertuples()]
    json.dump(cams, open(out / "cameras.json", "w"), default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o), indent=1)

    dets = dets[dets.track_id.notna()].join(images["frame"], on="image_id")
    pitch = dets.bbox_pitch.apply(lambda p: (p["x_bottom_middle"], p["y_bottom_middle"]) if isinstance(p, dict) else (np.nan, np.nan))
    l, t, w, h = np.stack(dets.bbox_ltwh.values).T
    pd.DataFrame({"frame": dets.frame, "track_id": dets.track_id.astype(int),
                  "x_m": [p[0] for p in pitch], "y_m": [p[1] for p in pitch],
                  "box_l": l, "box_t": t, "box_w": w, "box_h": h}).sort_values(["frame", "track_id"]).to_csv(out / "players.csv", index=False)

    ok = sum(c["camera"] is not None for c in cams)
    print(f"{out}: {ok}/{len(cams)} frames calibrated, {len(dets)} tracked detections, {dets.track_id.nunique()} tracks")

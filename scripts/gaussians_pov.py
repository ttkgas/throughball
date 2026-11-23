"""Stages 3-4: a 3D Gaussian scene from one broadcast clip, rendered from a player's eyes.

Reads stage 1 (outputs/stage1/<vid>/{cameras.json, players.csv}) and the clip's frames.
Scene, in SoccerNet pitch coordinates (meters, centre spot at 0, z points down):
  background  Gaussians on known geometry: the pitch plane, a stadium bowl around it, and the
              goal frames. Positions stay fixed; colours, opacities and sizes are fit to every
              frame with gsplat, players masked out. With --bg the background is fit from a longer
              stretch of the same broadcast and from other clips of the same match, so stands
              seen at other moments are real footage.
              Parts no frame saw keep a prior: grass with the standard line markings, and the
              stand across the pitch mirrored in (stadiums are roughly symmetric).
  players     per-frame Gaussians: each tracked player's segmented pixels placed on a vertical
              plane at their (smoothed) pitch position, one Gaussian per pixel. The plane turns
              to face whichever camera renders it, so from the broadcast camera it reproduces the
              frame, and from a player's eyes it shows the side the broadcast saw. With
              --players body they wrap an upright body instead, fixed in the broadcast's
              orientation, with the seen side mirrored onto the back, so players have depth
              and turn as you walk round them.
The first-person camera sits at the chosen player's eye height, looking toward where the
broadcast camera points (a stand-in for the ball, which stage 1 doesn't track).
Writes outputs/stage3/<vid>/: pov_<track>.mp4 (broadcast | re-render | first person | map),
preview_<track>.png, metrics.json, and pov_<track>/ with the raw first-person frames for stage 4
(rgb_*.png, unseen_*.png where 255 = nothing known there (no broadcast frame saw it, not pitch), depth_*.png in mm with 0 for sky, cameras.json).
Usage: python scripts/gaussians_pov.py [--vid 021] [--track ID] [--iters 3000] [--bg 021_full,024] [--fill 10] [--out 021_match] [--own-share 0.5] [--players body] [--pin-camera 60]
"""
import argparse
import glob
import json
import math
from pathlib import Path

import cv2
import imageio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from gsplat import rasterization
from scipy.ndimage import gaussian_filter1d
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--vid", default="021")
p.add_argument("--track", type=int, default=None, help="track id to see through (default: busiest player)")
p.add_argument("--iters", type=int, default=3000)
p.add_argument("--eye", type=float, default=1.7, help="eye height, meters")
p.add_argument("--fov", type=float, default=90, help="horizontal field of view, degrees")
p.add_argument("--bg", default=None, help="stage 1 exports to fit the background from, comma separated: a longer stretch of the same "
               "broadcast (e.g. 021_full) and other clips of the same match (e.g. 024,031), so stands and pitch seen at other "
               "moments fill what this window never saw. An export's clip is the part of its name before '_'")
p.add_argument("--out", default=None, help="output folder name under outputs/stage3 (default: --vid)")
p.add_argument("--bg-stride", type=int, default=3, help="use every n-th frame of --bg")
p.add_argument("--own-share", type=float, default=0, help="share of fit steps drawn from the first --bg export alone "
               "(the clip itself), so other clips add coverage without blurring it; 0 draws all frames alike")
p.add_argument("--players", choices=["plane", "body"], default="plane", help="plane: each player's pixels on a flat cutout "
               "turned to face the viewer; body: wrapped on a fixed upright body (an elliptic cylinder), the side the "
               "broadcast saw toward its camera and the same pixels mirrored onto the back")
p.add_argument("--pin-camera", type=float, default=0, metavar="PX", help="hold the broadcast camera's centre at its median position (from "
               "the first --bg export, else this clip) and re-solve each frame's rotation and focal length to keep where it "
               "puts the pitch; steadies cameras whose centre jitters in calibration, as zoomed-in shots do. Frames the fixed camera can't match within PX pixels (another camera) keep their "
               "own. A 2 m shift alone moves the pitch 10-15 px, so PX is well above 8")
p.add_argument("--fill", type=int, default=0, help="generate what no footage covers with SDXL inpainting on every n-th "
               "first-person frame, and bake it into the Gaussians' colours so it stays put from frame to frame (0: off)")
a = p.parse_args()
dev = "cuda"
torch.manual_seed(0)
S1, OUT = Path(f"outputs/stage1/{a.vid}"), Path(f"outputs/stage3/{a.out or a.vid}")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- cameras and frames ----------------
cams = json.load(open(S1 / "cameras.json"))
T = len(cams)
paths = sorted(glob.glob(f"data/SoccerNetGS/*/SNGS-{a.vid}/img1/*.jpg"))[:T]
frames = [cv2.imread(f)[..., ::-1].copy() for f in paths]  # RGB uint8
H, W = frames[0].shape[:2]
ok = [i for i, c in enumerate(cams) if c["camera"]]
cam = [cams[min(ok, key=lambda j: abs(j - i))]["camera"] for i in range(T)]  # fill failed frames


# close-ups and replays calibrate badly: a plausible broadcast camera is off the pitch, 3 to 60 m up, within 150 m of the centre spot
plausible = lambda c: c is not None and (lambda x, y, z: (abs(x) > 52.5 or abs(y) > 34) and 3 < -z < 60 and math.hypot(x, y) < 150)(*c["position_meters"])


def pin_centre(R, C, K, C0, px):
    """Same frames with the camera centre held at C0: rotation and focal length re-solved so each frame still puts the
    pitch where its own calibration did. Frames that can't be matched within px pixels (another camera) keep their own."""
    g = torch.stack(torch.meshgrid(torch.linspace(-52.5, 52.5, 22, device=dev), torch.linspace(-34, 34, 15, device=dev), indexing="ij"), -1)
    X = F.pad(g.reshape(-1, 2), (0, 1))  # pitch points, z = 0

    def proj(R, C, f):
        q = torch.einsum("tij,tnj->tni", R, X[None] - C[:, None])
        return q[..., :2] / q[..., 2:].clamp(min=1e-3) * f[:, None, None], q[..., 2]

    p0, z0 = proj(R, C, K[:, 0, 0])
    m = ((z0 > 1) & (p0.abs() < torch.tensor([W, H], device=dev) * .6).all(-1)).float()
    w = torch.zeros(len(R), 3, device=dev, requires_grad=True)
    lf = K[:, 0, 0].log().clone().requires_grad_()
    opt = torch.optim.Adam([w, lf], lr=1e-2)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, (1e-5 / 1e-2) ** (1 / 1500))  # anneal: a constant step jitters by ~lr * f pixels
    for _ in range(1500):
        S = torch.zeros(len(R), 3, 3, device=dev)
        S[:, 0, 1], S[:, 0, 2], S[:, 1, 2] = -w[:, 2], w[:, 1], -w[:, 0]
        R1 = torch.linalg.matrix_exp(S - S.transpose(1, 2)) @ R
        p, _ = proj(R1, C0.expand_as(C), lf.exp())
        err = (((p - p0) ** 2).sum(-1) * m).sum(1) / m.sum(1).clamp(min=1)
        opt.zero_grad(); err.sum().backward(); opt.step(); sched.step()
    good = (err.sqrt() < px) & (m.sum(1) >= 8)
    K1 = K.clone(); K1[:, :2, :2] *= (lf.exp() / K[:, 0, 0])[:, None, None]
    return torch.where(good[:, None, None], R1.detach(), R), torch.where(good[:, None], C0.expand_as(C), C), torch.where(good[:, None, None], K1.detach(), K), good, err.detach().sqrt()


def cameras(cam, pin=None):
    R = torch.tensor([c["rotation_matrix"] for c in cam], device=dev)
    C = torch.tensor([c["position_meters"] for c in cam], device=dev)
    K = torch.tensor([[[c["x_focal_length"], 0, c["principal_point"][0]], [0, c["y_focal_length"], c["principal_point"][1]], [0, 0, 1]] for c in cam], device=dev)
    K[:, :2] *= (W / 2 / K[:, 0, 2])[:, None, None]  # in case calibration ran at another resolution
    if pin is not None:
        R, C, K, good, rms = pin_centre(R, C, K, pin, a.pin_camera)
        q = rms.nan_to_num(1e9).quantile(torch.tensor([.1, .5, .9], device=dev)).tolist()
        print(f"pinned the camera centre in {int(good.sum())} of {len(cam)} frames (10/50/90% of frames within {q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f} px of their own calibration)")
    V = torch.eye(4, device=dev).repeat(len(cam), 1, 1)
    V[:, :3, :3], V[:, :3, 3] = R, -(R @ C[..., None])[..., 0]
    Kh = K.clone(); Kh[:, :2] /= 2  # half resolution, for fitting
    return R, C, K, V, Kh


PIN = None
if a.pin_camera:
    src = json.load(open(f"outputs/stage1/{a.bg.split(',')[0] if a.bg else a.vid}/cameras.json"))
    PIN = torch.tensor([c["camera"]["position_meters"] for c in src if plausible(c["camera"])], device=dev).median(0).values
R, C, K, V, Kh = cameras(cam, PIN)
h2, w2 = H // 2, W // 2
print(f"{T} frames {W}x{H}, {len(ok)} calibrated, camera at {C.median(0).values.tolist()} m")

# ---------------- players: segmentation masks and per-pixel Gaussians ----------------
df = pd.read_csv(S1 / "players.csv").dropna(subset=["x_m"])
df["t"] = df.frame.map({c["frame"]: i for i, c in enumerate(cams)})
seg = YOLO("yolo11x-seg.pt")
excl = torch.zeros(T, h2, w2, dtype=torch.bool, device=dev)  # pixels kept out of the background fit
shrink = lambda ex: F.max_pool2d(F.max_pool2d(ex[None, None].float(), 15, 1, 7), 2)[0, 0] > 0  # grow a little, half resolution
app = {}  # (track, t) -> per pixel: (u, height) on the player's plane, rgb, Gaussian size, plane normal (xy, away from the camera)
for t in range(T):
    r = seg.predict(np.ascontiguousarray(frames[t][..., ::-1]), classes=[0], conf=0.2, imgsz=1280, retina_masks=True, verbose=False)[0]
    m = r.masks.data > 0.5 if r.masks is not None else torch.zeros(0, H, W, dtype=torch.bool, device=dev)
    ex = m.any(0) if len(m) else torch.zeros(H, W, dtype=torch.bool, device=dev)
    img = torch.from_numpy(frames[t]).to(dev).float() / 255
    Kinv = torch.linalg.inv(K[t])
    for d in df[df.t == t].itertuples():
        x0, y0, x1, y1 = int(d.box_l), int(d.box_t), int(d.box_l + d.box_w), int(d.box_t + d.box_h)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        ex[max(y0 - 4, 0):y1 + 4, max(x0 - 4, 0):x1 + 4] = True
        if not len(m) or x1 <= x0 or y1 <= y0:
            continue
        sb = r.boxes.xyxy
        iw = (sb[:, 2].clamp(max=x1) - sb[:, 0].clamp(min=x0)).clamp(min=0)
        ih = (sb[:, 3].clamp(max=y1) - sb[:, 1].clamp(min=y0)).clamp(min=0)
        iou = iw * ih / ((sb[:, 2] - sb[:, 0]) * (sb[:, 3] - sb[:, 1]) + (x1 - x0) * (y1 - y0) - iw * ih)
        if iou.max() < 0.4:
            continue
        ys, xs = torch.nonzero(m[iou.argmax(), y0:y1, x0:x1], as_tuple=True)
        if len(xs) < 20:
            continue
        ys, xs = ys + y0, xs + x0
        ray = (R[t].T @ Kinv @ torch.stack([xs + .5, ys + .5, torch.ones(len(xs), device=dev)])).T  # world dirs, unit camera depth
        B = torch.tensor([d.x_m, d.y_m, 0.], device=dev)
        n = F.pad(F.normalize((B - C[t])[:2], dim=0), (0, 1))  # plane normal, horizontal, facing the camera
        s = ((B - C[t]) @ n) / (ray @ n)  # depth along each pixel ray to the player's plane
        X = C[t] + s[:, None] * ray - B
        app[d.track_id, t] = (X[:, :2] @ torch.stack([-n[1], n[0]]), -X[:, 2], img[ys, xs], (0.6 * s / K[t, 0, 0])[:, None].expand(-1, 3), n[:2])
    excl[t] = shrink(ex)
print(f"segmented players in {len(app)} of {len(df)} tracked detections")

tracks = {}  # track -> (frames, smoothed xy per frame, frames with appearance)
for k, g in df.groupby("track_id"):
    ts = np.arange(g.t.min(), g.t.max() + 1)
    xy = gaussian_filter1d(np.stack([np.interp(ts, g.t, g.x_m), np.interp(ts, g.t, g.y_m)], 1), 2, axis=0, mode="nearest")
    tracks[k] = (ts, xy, np.array(sorted(t for kk, t in app if kk == k)))


def players_at(t, P, skip=None):
    """Player Gaussians at frame t: each player's plane turned to face camera position P, or with --players body
    wrapped on an upright elliptic cylinder that keeps the broadcast's orientation, whatever camera renders it."""
    out = []
    for k, (ts, xy, seen) in tracks.items():
        if k == skip or not ts[0] <= t <= ts[-1] or not len(seen):
            continue
        tt = seen[np.abs(seen - t).argmin()]  # nearest frame where this player was segmented
        if abs(tt - t) > 12:
            continue
        u, hh, rgb, sg, nb = app[k, tt]
        B = xy[t - ts[0]]
        if a.players == "body":
            half = u.abs().quantile(.95).clamp(.15, .45)  # half width across the body, arms mostly out
            th = torch.asin((u / half).clamp(-1, 1))
            side = half * torch.sin(th)[:, None] * torch.stack([-nb[1], nb[0]])
            deep = (.6 * half).clamp(max=.2) * torch.cos(th)[:, None] * nb  # half depth front to back
            Bt = torch.tensor(B, dtype=torch.float32, device=dev)
            z = -hh[:, None]
            out.append((torch.cat([torch.cat([Bt + side - deep, z], 1), torch.cat([Bt + side + deep, z], 1)]),
                        rgb.repeat(2, 1), torch.ones(2 * len(u), device=dev), sg.repeat(2, 1)))
            continue
        dx, dy = B - P[:2]
        r = np.array([-dy, dx]) / max(math.hypot(dx, dy), 1e-6)
        out.append((torch.stack([B[0] + u * r[0], B[1] + u * r[1], -hh], 1), rgb, torch.ones_like(u), sg))
    if not out:
        return [torch.zeros(0, 3, device=dev), torch.zeros(0, 3, device=dev), torch.zeros(0, device=dev), torch.zeros(0, 3, device=dev)]
    return [torch.cat(x) for x in zip(*out)]


# ---------------- frames the background is fit from ----------------
tgt = torch.stack([torch.from_numpy(cv2.resize(f, (w2, h2), interpolation=cv2.INTER_AREA)) for f in frames]).to(dev)
bR, bC, bV, bKh, btgt, bexcl, bok = R, C, V, Kh, tgt, excl, ok
own = len(bok)  # frames of the first export, the ones --own-share favours
if a.bg:  # the same match over a longer stretch or other clips: people masked out, nothing else kept from them
    sel, btgt, bexcl = [], [], []
    for name in a.bg.split(","):
        bcams = json.load(open(f"outputs/stage1/{name}/cameras.json"))
        bdf = pd.read_csv(f"outputs/stage1/{name}/players.csv")
        allp = sorted(glob.glob(f"data/SoccerNetGS/*/SNGS-{name.split('_')[0]}/img1/*.jpg"))
        good = [i for i, c in enumerate(bcams) if plausible(c["camera"])]
        for i in good[::a.bg_stride]:
            f = cv2.imread(allp[i])  # cameras.json is in frame order, like the image files
            r = seg.predict(f, classes=[0], conf=0.2, imgsz=1280, retina_masks=True, verbose=False)[0]
            ex = r.masks.data.any(0) if r.masks is not None else torch.zeros(H, W, dtype=torch.bool, device=dev)
            for d in bdf[bdf.frame == bcams[i]["frame"]].itertuples():
                ex[max(int(d.box_t) - 4, 0):int(d.box_t + d.box_h) + 4, max(int(d.box_l) - 4, 0):int(d.box_l + d.box_w) + 4] = True
            bexcl.append(shrink(ex))
            btgt.append(torch.from_numpy(cv2.resize(f[..., ::-1], (w2, h2), interpolation=cv2.INTER_AREA).copy()).to(dev))
            sel.append(bcams[i]["camera"])
        own = len(sel) if name == a.bg.split(",")[0] else own
        print(f"background: {len(good[::a.bg_stride])} frames of {name} ({len(good)} of {len(bcams)} with a plausible camera, every {a.bg_stride}th)")
    bR, bC, _, bV, bKh = cameras(sel, PIN)
    btgt, bexcl, bok = torch.stack(btgt), torch.stack(bexcl), list(range(len(sel)))
bT = len(bok)

# ---------------- background Gaussians on known geometry ----------------
def pitch_lines():
    L = [[(-52.5, -34), (52.5, -34), (52.5, 34), (-52.5, 34), (-52.5, -34)], [(0, -34), (0, 34)]]
    th = np.linspace(0, 2 * np.pi, 73)
    L.append(list(zip(9.15 * np.cos(th), 9.15 * np.sin(th))))
    for s in (-1, 1):
        L.append([(s * 52.5, -20.16), (s * 36, -20.16), (s * 36, 20.16), (s * 52.5, 20.16)])
        L.append([(s * 52.5, -9.16), (s * 47, -9.16), (s * 47, 9.16), (s * 52.5, 9.16)])
        th = np.linspace(-1, 1, 21) * np.arccos(5.5 / 9.15)
        L.append(list(zip(s * (41.5 - 9.15 * np.cos(th)), 9.15 * np.sin(th))))
    return [np.array(l, dtype=np.float32) for l in L]


g = 0.08  # ground spacing, meters
gx, gy = torch.meshgrid(torch.arange(-57, 57, g, device=dev), torch.arange(-39, 39, g, device=dev), indexing="ij")
ground = torch.stack([gx.flatten(), gy.flatten(), torch.zeros(gx.numel(), device=dev)], 1)
on_line = torch.zeros(len(ground), dtype=torch.bool, device=dev)
for l in pitch_lines():
    a0, a1 = torch.from_numpy(l[:-1]).to(dev), torch.from_numpy(l[1:]).to(dev)
    for i in range(0, len(ground), 2 ** 20):
        q = ground[i:i + 2 ** 20, None, :2]
        w = ((q - a0) * (a1 - a0)).sum(-1) / ((a1 - a0) ** 2).sum(-1)
        on_line[i:i + 2 ** 20] |= ((q - a0 - w.clamp(0, 1)[..., None] * (a1 - a0)).norm(dim=-1) < 0.06).any(1)
for sx in (0, -41.5, 41.5):
    on_line |= (ground[:, :2] - torch.tensor([sx, 0.], device=dev)).norm(dim=1) < 0.11

b = 0.2  # stand spacing; the bowl rises at 33 degrees from 1 m boards 4.5 m (sides) / 4.5 m (ends) outside the lines
bx, by = torch.meshgrid(torch.arange(-96, 96, b, device=dev), torch.arange(-78, 78, b, device=dev), indexing="ij")
dist = torch.hypot((bx.abs() - 57).clamp(min=0), (by.abs() - 39).clamp(min=0))
hgt = 1 + dist * math.tan(math.radians(33))
keep = (dist > 0) & (hgt < 25)
bowl = torch.stack([bx[keep], by[keep], -hgt[keep]], 1)
per = torch.cat([torch.stack([torch.arange(-57, 57, .1, device=dev), torch.full((1140,), s * 39., device=dev)], 1) for s in (-1, 1)]
                + [torch.stack([torch.full((780,), s * 57., device=dev), torch.arange(-39, 39, .1, device=dev)], 1) for s in (-1, 1)])
hz = torch.arange(.05, 1, .1, device=dev)
boards = torch.cat([per.repeat_interleave(len(hz), 0), -hz.repeat(len(per))[:, None]], 1)

# stand points the broadcast camera sees from close by would hide the near touchline: left out of broadcast renders only
stand = torch.cat([bowl, boards])
near = torch.zeros(len(stand), dtype=torch.bool, device=dev)
for t in range(bT):
    Xc = (stand - bC[t]) @ bR[t].T
    near |= (Xc[:, 2] > 0) & (Xc[:, 2] < 25)
nb = len(bowl)

goal = []  # goal frames: posts and crossbar, 7.32 x 2.44 m
for sx in (-52.5, 52.5):
    for sy in (-3.66, 3.66):
        z = torch.arange(0, 2.44, .03, device=dev)
        goal.append(torch.stack([torch.full_like(z, sx), torch.full_like(z, sy), -z], 1))
    y = torch.arange(-3.66, 3.66, .03, device=dev)
    goal.append(torch.stack([torch.full_like(y, sx), y, torch.full_like(y, -2.44)], 1))
goal = torch.cat(goal)

means = torch.cat([ground, stand])
N, NG = len(means), len(ground)
kind = torch.cat([torch.zeros(NG), torch.ones(nb), 2 * torch.ones(len(stand) - nb)]).to(dev)  # 0 ground, 1 bowl, 2 boards
on_line = torch.cat([on_line, torch.zeros(N - NG, dtype=torch.bool, device=dev)])
hide = torch.cat([torch.zeros(NG, dtype=torch.bool, device=dev), near])  # near stand, left out from the broadcast camera
print(f"background: {NG} ground + {nb} stand + {len(stand) - nb} board Gaussians, {len(goal)} goal frame")

# ---------------- initial colours: average of every frame that sees each Gaussian ----------------
csum, cnt = torch.zeros(N, 3, device=dev), torch.zeros(N, device=dev)
for t in range(bT):
    Xc = ((means - bC[t]) @ bR[t].T) @ bKh[t].T
    z = Xc[:, 2].clamp(min=1e-6)
    u, v = (Xc[:, 0] / z).long(), (Xc[:, 1] / z).long()
    vis = (z > 0) & (u >= 0) & (u < w2) & (v >= 0) & (v < h2) & ~hide
    vis[vis.clone()] &= ~bexcl[t, v[vis], u[vis]]
    csum[vis] += btgt[t, v[vis], u[vis]].float() / 255
    cnt[vis] += 1
seen = cnt > 0
col = csum / cnt.clamp(min=1)[:, None]
grass = col[seen & (kind == 0) & ~on_line].median(0).values
col[~seen & (kind == 0)] = grass
col[~seen & (kind == 0) & on_line] = 0.9
# unseen stand: copy the stand seen across the pitch (stadiums are roughly symmetric), so it is real crowd, not noise
G = torch.full((round(192 / b) + 1, round(156 / b) + 1, 3), -1., device=dev)
cell = lambda xy: (((xy[:, 0] + 96) / b).round().long().clamp(0, G.shape[0] - 1), ((xy[:, 1] + 78) / b).round().long().clamp(0, G.shape[1] - 1))
sb = seen & (kind == 1)
G[cell(means[sb])] = col[sb]
for fx, fy in ((1, -1), (-1, 1), (-1, -1)):
    miss = (~seen & (kind == 1)).nonzero()[:, 0]
    got = G[cell(means[miss, :2] * torch.tensor([fx, fy], device=dev))]
    hit = got[:, 0] >= 0
    col[miss[hit]] = got[hit]
    seen[miss[hit]] = True  # only for the next flip; restored below
real = seen.clone()  # real footage behind the colour: seen, or mirrored from a seen stand
seen = cnt > 0
for k in (1, 2):  # still unseen stand: crowd colours sampled from the seen stand; unseen boards: their mean
    pool = col[seen & (kind == k)]
    miss = ~seen & (kind == k) & ((col == 0).all(1) if k == 1 else True)
    if len(pool):
        col[miss] = pool[torch.randint(len(pool), (int(miss.sum()),), device=dev)] if k == 1 else pool.mean(0)
print(f"seen by the broadcast: {seen.float().mean():.0%} of background Gaussians, {real.float().mean():.0%} with mirrored stands")
real |= kind == 0  # unseen pitch: grass and markings come from the known pitch layout, nothing to generate there

# ---------------- fit colours, opacities and sizes to the frames ----------------
c_l = torch.logit(col.clamp(.02, .98)).requires_grad_()
o_l = torch.full((N,), 3., device=dev).requires_grad_()
sc0 = torch.where(kind[:, None] == 0, torch.tensor([.6 * g, .6 * g, .005], device=dev), torch.tensor(.6 * b, device=dev).repeat(3))
sc0[kind == 2] = .06
s_l = sc0.log().requires_grad_()
sky_l = torch.zeros(3, device=dev).requires_grad_()
gcol, gop, gsc = torch.full((len(goal), 3), .95, device=dev), torch.ones(len(goal), device=dev), torch.full((len(goal), 3), .05, device=dev)
opt = torch.optim.Adam([{"params": [c_l], "lr": .02}, {"params": [o_l], "lr": .02}, {"params": [s_l], "lr": .002}, {"params": [sky_l], "lr": .01}])


def render(V1, K1, w, h, extra=None, unseen=False, broadcast=False):
    """RGB from camera V1, K1. With unseen=True: RGB is 1 where no footage stands behind the colour (prior grass, sky), and a 4th channel is depth."""
    c0 = (~real).float()[:, None].expand(-1, 3) if unseen else torch.sigmoid(c_l)
    if unseen and extra:
        extra = [extra[0], torch.zeros_like(extra[1]), *extra[2:]]
    bg = [x[~hide] if broadcast else x for x in (means, c0, torch.sigmoid(o_l), s_l.exp())]
    m, c, o, s = [torch.cat(x) for x in zip(bg, (goal, gcol * (not unseen), gop, gsc), *([extra] if extra else []))]
    q = torch.tensor([1., 0, 0, 0], device=dev).expand(len(m), 4).contiguous()
    img, _, _ = rasterization(m, q, s, o, c, V1[None], K1[None], w, h, render_mode="RGB+ED" if unseen else "RGB",
                              backgrounds=(torch.ones(3, device=dev) if unseen else torch.sigmoid(sky_l))[None], packed=False,
                              rasterize_mode="antialiased", near_plane=0.05)
    return img[0] if unseen else img[0].clamp(0, 1)


for it in range(a.iters + 1):
    t = bok[np.random.randint(own if np.random.rand() < a.own_share else len(bok))]
    keep = ~bexcl[t]
    loss = ((render(bV[t], bKh[t], w2, h2, broadcast=True) - btgt[t].float() / 255).abs().mean(-1) * keep).sum() / keep.sum()
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 500 == 0:
        print(f"fit {it}/{a.iters}: L1 {loss.item():.4f}")

# ---------------- first-person camera ----------------
A = torch.stack([C[t] - C[t, 2] / R[t, 2, 2] * R[t, 2] for t in range(T)]).cpu().numpy()[:, :2]  # where the broadcast points, on the pitch
A = gaussian_filter1d(A, 3, axis=0, mode="nearest")
if a.track is None:  # busiest player: tracked through most of the clip, closest on average to the action
    long = [k for k, (ts, _, s) in tracks.items() if len(ts) >= .8 * T and len(s)] or list(tracks)
    a.track = min(long, key=lambda k: np.linalg.norm(tracks[k][1] - A[tracks[k][0]], axis=1).mean())
ts, xy, _ = tracks[a.track]
eye = xy[np.clip(np.arange(T) - ts[0], 0, len(ts) - 1)]
fwd = gaussian_filter1d(A - eye, 4, axis=0, mode="nearest")
fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
pw, ph, tilt = 1280, 720, math.radians(8)
fpx = pw / 2 / math.tan(math.radians(a.fov) / 2)
Kp = torch.tensor([[fpx, 0, pw / 2], [0, fpx, ph / 2], [0, 0, 1]], device=dev)
print(f"first person: track {a.track}, in {len(ts)} of {T} frames")


def pov_view(t):
    E = np.array([*eye[t], -a.eye])
    z = np.array([*(fwd[t] * math.cos(tilt)), math.sin(tilt)])  # forward, tilted a little down (z is down)
    y = np.array([0, 0, 1.]) - z[2] * z; y /= np.linalg.norm(y)
    Rp = np.stack([np.cross(y, z), y, z])
    Vp = np.eye(4); Vp[:3, :3], Vp[:3, 3] = Rp, -Rp @ E
    return torch.tensor(Vp, dtype=torch.float32, device=dev), E


# ---------------- fill: generated keyframes baked into the scene ----------------
if a.fill:
    from diffusers import AutoPipelineForInpainting
    from PIL import Image

    pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                                                     torch_dtype=torch.float16, variant="fp16").to(dev)
    pipe.set_progress_bar_config(disable=True)
    prompt = "packed football stadium stands full of spectators, green grass pitch, floodlights, night sky, realistic broadcast photo"
    keys = []
    with torch.no_grad():
        for t in range(0, T, a.fill):
            Vp, _ = pov_view(t)
            img, un = render(Vp, Kp, pw, ph), render(Vp, Kp, pw, ph, unseen=True)[..., 0] > .5
            if un.float().mean() < .01:
                continue
            m = cv2.dilate(un.cpu().numpy().astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
            out = pipe(prompt, negative_prompt="cartoon, painting, blurry, text, watermark, people close up", width=1024, height=576,
                       image=Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)).resize((1024, 576)),
                       mask_image=Image.fromarray(m).resize((1024, 576)), strength=0.99, num_inference_steps=25,
                       generator=torch.Generator(dev).manual_seed(t)).images[0]
            keys.append((Vp, torch.from_numpy(np.array(out.resize((pw // 2, ph // 2)))).to(dev).float() / 255,
                         F.interpolate(un[None, None].float(), size=(ph // 2, pw // 2))[0, 0] > .5))
    del pipe
    torch.cuda.empty_cache()
    Kp2 = Kp.clone(); Kp2[:2] /= 2
    opt = torch.optim.Adam([c_l], lr=.03)  # colours only, and only where no footage stands behind them
    for it in range(600):
        Vp, gt, m = keys[np.random.randint(len(keys))]
        loss = ((render(Vp, Kp2, pw // 2, ph // 2) - gt).abs().mean(-1) * m).sum() / m.sum()
        opt.zero_grad(); loss.backward(); c_l.grad[real] = 0; opt.step()
    print(f"fill: {len(keys)} generated keyframes baked into the unseen Gaussians (L1 {loss.item():.3f})")

# ---------------- map, video, metrics ----------------
LINES = pitch_lines()


def minimap(t):
    s, ox, oy = 7.5, 480, 270
    img = np.full((540, 960, 3), (30, 90, 40), np.uint8)
    P = lambda q: (int(ox + q[0] * s), int(oy + q[1] * s))
    for l in LINES:
        cv2.polylines(img, [np.array([P(q) for q in l])], False, (220, 220, 220), 1, cv2.LINE_AA)
    for k, (kts, kxy, _) in tracks.items():
        if kts[0] <= t <= kts[-1]:
            cv2.circle(img, P(kxy[t - kts[0]]), 7 if k == a.track else 4, (230, 40, 40) if k == a.track else (240, 240, 240), -1, cv2.LINE_AA)
    for side in (-1, 1):
        ang = math.atan2(fwd[t, 1], fwd[t, 0]) + side * math.radians(a.fov) / 2
        cv2.line(img, P(eye[t]), P(eye[t] + 30 * np.array([math.cos(ang), math.sin(ang)])), (230, 40, 40), 1, cv2.LINE_AA)
    cv2.drawMarker(img, P(A[t]), (250, 220, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
    return img


def label(img, text):
    cv2.putText(img, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 2, cv2.LINE_AA)
    return img


writer = imageio.get_writer(OUT / f"pov_{a.track}.mp4", fps=25, codec="libx264", quality=8, macro_block_size=8)
psnr, psnr_bg, cams = [], [], []
FR = OUT / f"pov_{a.track}"  # raw first-person frames: rgb, unseen mask (255 = no frame saw it), depth in mm
FR.mkdir(exist_ok=True)
with torch.no_grad():
    for t in range(T):
        Cb = C[t].cpu().numpy()
        rec = render(V[t], K[t], W, H, players_at(t, Cb), broadcast=True)
        gt = torch.from_numpy(frames[t]).to(dev).float() / 255
        err = ((rec - gt) ** 2).mean(-1)
        bgm = ~F.interpolate(excl[t][None, None].float(), size=(H, W))[0, 0].bool()
        psnr.append(-10 * math.log10(err.mean().item()))
        psnr_bg.append(-10 * math.log10(err[bgm].mean().item()))
        Vp, E = pov_view(t)
        pl = players_at(t, E, skip=a.track)
        pov = render(Vp, Kp, pw, ph, pl)
        aux = render(Vp, Kp, pw, ph, pl, unseen=True)  # for stage 4: what to generate, and depth
        cv2.imwrite(str(FR / f"rgb_{t:04d}.png"), (pov.cpu().numpy()[..., ::-1] * 255).astype(np.uint8))
        cv2.imwrite(str(FR / f"unseen_{t:04d}.png"), (aux[..., 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        cv2.imwrite(str(FR / f"depth_{t:04d}.png"), (aux[..., 3].clamp(0, 65.535).cpu().numpy() * 1000).astype(np.uint16))
        cams.append(Vp.cpu().numpy().tolist())
        tiles = [frames[t], (rec.cpu().numpy() * 255).astype(np.uint8), (pov.cpu().numpy() * 255).astype(np.uint8), minimap(t)]
        tiles = [label(cv2.resize(x, (960, 540), interpolation=cv2.INTER_AREA), n) for x, n in
                 zip(tiles, ["broadcast", "3D Gaussians, broadcast camera", f"first person, player {a.track}", "map"])]
        frame = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
        writer.append_data(frame)
        if t == T // 2:
            cv2.imwrite(str(OUT / f"preview_{a.track}.png"), frame[..., ::-1])
writer.close()
metrics = {"track": int(a.track), "frames": T, "psnr_broadcast": float(np.mean(psnr)), "psnr_background_only": float(np.mean(psnr_bg)),
           "gaussians_background": N, "background_seen": float(seen.float().mean()), "players": a.players}
json.dump(metrics, open(OUT / "metrics.json", "w"), indent=1)
json.dump({"track": int(a.track), "K": Kp.cpu().numpy().tolist(), "w": pw, "h": ph, "view": cams}, open(FR / "cameras.json", "w"))
print(metrics)
print(f"wrote {OUT}/pov_{a.track}.mp4")

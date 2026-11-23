"""Stage 4: first-person video by geometry-guided generation, against a pure video-generation baseline.

Reads stage 3's raw first-person frames (outputs/stage3/<vid>/pov_<track>/: rgb, unseen mask, depth,
cameras.json) and the clip's broadcast frames.
  guided    each Gaussian render frame goes through SDXL inpainting: the parts no broadcast frame saw
            (sky, stands behind the camera, far pitch) are generated, optionally a light pass over the whole
            frame (--refine) makes the render look more like footage. Geometry, players and camera come
            from stage 3, so the pitch lines stay where the 3D scene puts them.
  baseline  LTX-Video, image to video: the first broadcast frame and a text prompt asking for the
            player's view. No geometry, no camera.
Scores (there is no real first-person footage to compare with):
  line_f1   pitch markings found in the output (white, low saturation pixels) against the markings
            projected through the first-person camera, within 4 px. Tests whether the view is
            where the player stands and looks.
  flicker   mean absolute change between consecutive frames (0-255). Lower is steadier.
Writes outputs/stage4/<vid>/: compare.mp4 (render | guided | baseline), guided.mp4, baseline.mp4,
preview.png, metrics.json.
Usage: python scripts/generate_pov.py [--vid 021] [--refine 0] [--no-baseline]
"""
import argparse
import glob
import json
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--vid", default="021")
p.add_argument("--refine", type=float, default=0, help="strength of a whole-frame SDXL pass, 0 to skip (0.3 halved line_f1 and tripled flicker)")
p.add_argument("--steps", type=int, default=25)
p.add_argument("--no-baseline", action="store_true")
a = p.parse_args()

src = sorted(glob.glob(f"outputs/stage3/{a.vid}/pov_*/cameras.json"))[-1]
FR = Path(src).parent
cam = json.load(open(src))
OUT = Path(f"outputs/stage4/{a.vid}")
OUT.mkdir(parents=True, exist_ok=True)
T, W, H = len(cam["view"]), cam["w"], cam["h"]
rgb = [np.ascontiguousarray(cv2.imread(str(FR / f"rgb_{t:04d}.png"))[..., ::-1]) for t in range(T)]
unseen = [cv2.imread(str(FR / f"unseen_{t:04d}.png"), 0) for t in range(T)]
print(f"{T} first-person frames of player {cam['track']} at {W}x{H}")


# ---------------- scores ----------------
def pitch_lines():
    L = [[(-52.5, -34), (52.5, -34), (52.5, 34), (-52.5, 34), (-52.5, -34)], [(0, -34), (0, 34)]]
    th = np.linspace(0, 2 * np.pi, 145)
    L.append(list(zip(9.15 * np.cos(th), 9.15 * np.sin(th))))
    for s in (-1, 1):
        L.append([(s * 52.5, -20.16), (s * 36, -20.16), (s * 36, 20.16), (s * 52.5, 20.16)])
        L.append([(s * 52.5, -9.16), (s * 47, -9.16), (s * 47, 9.16), (s * 52.5, 9.16)])
        th = np.linspace(-1, 1, 41) * np.arccos(5.5 / 9.15)
        L.append(list(zip(s * (41.5 - 9.15 * np.cos(th)), 9.15 * np.sin(th))))
    pts = []
    for l in L:  # resample every 5 cm so the projection is a solid curve
        l = np.array(l)
        for p0, p1 in zip(l[:-1], l[1:]):
            n = max(int(np.linalg.norm(p1 - p0) / .05), 1)
            pts.append(p0 + np.linspace(0, 1, n, endpoint=False)[:, None] * (p1 - p0))
    pts = np.concatenate(pts)
    return np.c_[pts, np.zeros(len(pts))]


LINES = pitch_lines()
K = np.array(cam["K"])


def line_mask(t, w, h):
    V = np.array(cam["view"][t])
    X = LINES @ V[:3, :3].T + V[:3, 3]
    X = X[X[:, 2] > .5]
    uv = (X @ K.T)[:, :2] / X[:, 2:] * [w / W, h / H]
    m = np.zeros((h, w), np.uint8)
    uv = np.round(uv).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    m[uv[ok, 1], uv[ok, 0]] = 1
    return m


def white(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    grass = cv2.inRange(hsv, (30, 40, 30), (90, 255, 255))
    near_grass = cv2.dilate(grass, np.ones((15, 15), np.uint8)) > 0
    tophat = cv2.morphologyEx(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.MORPH_TOPHAT, np.ones((9, 9), np.uint8))
    return ((tophat > 25) & (hsv[..., 1] < 90) & near_grass).astype(np.uint8)


def scores(frames):
    f1, k4 = [], np.ones((9, 9), np.uint8)
    for t, f in enumerate(frames):
        h, w = f.shape[:2]
        L, D = line_mask(t, w, h), white(f)
        if L.sum() < 50:
            continue
        rec = (L & cv2.dilate(D, k4)).sum() / L.sum()
        pre = (D & cv2.dilate(L, k4)).sum() / max(D.sum(), 1)
        f1.append(2 * rec * pre / max(rec + pre, 1e-6))
    flick = np.mean([np.abs(x.astype(np.int16) - y).mean() for x, y in zip(frames[1:], frames)])
    return {"line_f1": float(np.mean(f1)) if f1 else None, "flicker": float(flick)}


def save(name, frames):
    with imageio.get_writer(OUT / f"{name}.mp4", fps=25, codec="libx264", quality=8, macro_block_size=8) as wr:
        for f in frames:
            wr.append_data(f)


metrics = {"track": cam["track"], "render": scores(rgb)}
print("render", metrics["render"])

# ---------------- guided: SDXL inpainting on the Gaussian render ----------------
from diffusers import AutoPipelineForInpainting

pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                                                 torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.set_progress_bar_config(disable=True)
prompt = ("first-person view from a professional football player's eyes during a match, green grass pitch with white "
          "line markings, packed stadium stands with spectators, floodlights, realistic broadcast photo")
neg = "cartoon, painting, blurry, text, watermark, distorted, deformed people"
gw, gh = 1024, 576
full = Image.new("L", (gw, gh), 255)
guided = []
for t in range(T):
    g = torch.Generator("cuda").manual_seed(0)  # same noise every frame keeps the fill steadier
    img = Image.fromarray(rgb[t]).resize((gw, gh), Image.LANCZOS)
    m = cv2.dilate(unseen[t], np.ones((9, 9), np.uint8))
    if m.mean() > 2:
        img = pipe(prompt, negative_prompt=neg, image=img, mask_image=Image.fromarray(m).resize((gw, gh)), width=gw, height=gh,
                   strength=0.99, num_inference_steps=a.steps, guidance_scale=7, generator=g).images[0]
    if a.refine > 0:
        img = pipe(prompt, negative_prompt=neg, image=img, mask_image=full, width=gw, height=gh,
                   strength=a.refine, num_inference_steps=a.steps, guidance_scale=6, generator=g).images[0]
    guided.append(np.array(img.resize((W, H), Image.LANCZOS)))
    if t % 10 == 0:
        print(f"guided {t}/{T}", flush=True)
del pipe
torch.cuda.empty_cache()
save("guided", guided)
metrics["guided"] = scores(guided)
print("guided", metrics["guided"])
json.dump(metrics, open(OUT / "metrics.json", "w"), indent=1)

# ---------------- baseline: LTX-Video from the first broadcast frame ----------------
baseline = None
if not a.no_baseline:
    from diffusers import LTXImageToVideoPipeline

    b0 = sorted(glob.glob(f"data/SoccerNetGS/*/SNGS-{a.vid}/img1/*.jpg"))[0]
    vp = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
    vp.enable_model_cpu_offload()
    nf = min(T - (T - 1) % 8, 97)  # LTX wants 8k+1 frames
    out = vp(image=Image.open(b0).convert("RGB"), prompt=prompt.replace("realistic broadcast photo", "the camera moves with the player, realistic video"),
             negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted", width=1024, height=576,
             num_frames=nf, num_inference_steps=40, generator=torch.Generator("cuda").manual_seed(0)).frames[0]
    baseline = [np.array(f.resize((W, H), Image.LANCZOS)) for f in out]
    del vp
    torch.cuda.empty_cache()
    save("baseline", baseline)
    metrics["baseline"] = scores(baseline)
    print("baseline", metrics["baseline"])


# ---------------- side by side ----------------
def label(img, text):
    img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


n = len(baseline) if baseline else T
cols = [("Gaussian render", rgb), ("geometry-guided", guided)] + ([("video generation only", baseline)] if baseline else [])
comp = [np.hstack([label(f[t], name) for name, f in cols]) for t in range(n)]
save("compare", comp)
cv2.imwrite(str(OUT / "preview.png"), comp[n // 2][..., ::-1])
json.dump(metrics, open(OUT / "metrics.json", "w"), indent=1)
print(metrics)

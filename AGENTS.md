# AGENTS.md

Shared context for coding agents and for us. Read this before starting work.
When a fact here changes, update this file in the same commit.

## Goal
Generate a first-person view of a soccer game from a chosen player's position,
starting from broadcast video and player tracking data.
Computer vision course project, Tejas and Sanika.

## Scope
The full version (rendering angles no camera ever saw) is out of scope.
Scoped version:
- Recover the broadcast camera using the feed and known pitch geometry.
- Drop a virtual camera on a chosen player and render what they would see.
- Stretch: use earlier footage of the same match (stadium layout, who is on the
  pitch) to fill in parts of the scene the current shot doesn't cover.

What's new versus existing soccer work: building the 3D scene from game
information and putting a camera inside a chosen player.
We extend existing code rather than reimplementing a paper.

## Base code
SoccerNet's Game State Reconstruction baseline, sn-gamestate, built on TrackLab.
`setup.sh` clones it into `third_party/sn-gamestate` at a pinned commit and
installs it into its own Python 3.9 env (it requires 3.9 and torch 1.13).
Don't edit `third_party/`. Our code and configs live in this repo and plug into
TrackLab with `--config-dir configs`.

## Pipeline
1. Camera calibration: PnLCalib (`modules/pitch` + `modules/calibration`).
   Gives per-frame camera parameters and each player's position on the pitch.
2. Player detection and tracking: YOLO + ByteTrack (no re-id for now).
   Stages 1-2 run together in tracklab. `scripts/export_stage1.py` turns its state file into
   `outputs/stage1/<video_id>/cameras.json` (per-frame camera) and `players.csv`
   (per-frame track id and pitch position in meters). Stage 3 reads only these.
   Scored with GS-HOTA by `configs/throughball_gsr.yaml`, which adds sn-gamestate's re-id,
   jersey number, role and team modules (GS-HOTA needs them) and runs the whole clip.
   SNGS-021 (2026-11-23, L4): GS-HOTA 31.2. `scripts/gsr_breakdown.sh` re-scores it with parts
   of the metric off: 39.8 without jerseys, 53.7 with no attributes (tracking on the pitch),
   53.6 as plain HOTA on image boxes. So calibration costs nothing; roles and teams cost 14
   points and jersey numbers 9.
3. Dynamic 3D scene built with dynamic 3D Gaussians: `scripts/gaussians_pov.py`
   (`notebooks/01_gaussians_pov.ipynb`). Static pitch-and-stands Gaussians fitted to the
   broadcast frames (`--bg` fits them from a longer export, e.g. the whole clip), plus one
   Gaussian billboard per tracked player. Renders a first-person view for one track and a
   broadcast check, and writes raw frames, unseen masks, depth and cameras for stage 4.
   `--fill N` inpaints every Nth first-person frame and bakes it into the unseen Gaussians.
   Best run so far (SNGS-021): stage 1 on all 750 frames (`nframes: -1`), exported with
   `export_stage1.py <state> 021_full`, then `--bg 021_full --fill 10`.
   `bash scripts/run_all.sh [SAVE_DIR]` redoes stages 1 to 4 this way, then the GS-HOTA run, and copies each to SAVE_DIR.
   `--bg` also takes other clips of the same match (SNGS-021 to 038 are all game 2;
   `scripts/same_match.py` finds them), keeping only plausible broadcast cameras;
   `bash scripts/match_bg.sh` runs that with four of them (with `--own-share 0.5 --iters 15000`) into `*_match` outputs.
   SNGS-021 with 024, 028, 033, 037 (2026-11-23, L4): background seen 77% (83% with mirrored
   stands) vs 61% (65%). Drawing all frames alike blurs 021 (PSNR 19.8 vs 22.3; 20.4 even at
   15000 iters); `--own-share 0.5 --iters 15000` keeps 021 sharp (PSNR 22.2) with the same 77%.
   Guided first-person line F1 0.33 vs 0.35, flicker 1.87 vs 1.85: more real crowd for a
   small cost on the pitch lines, likely from other clips' calibration error.
   `--players body` puts each player on an upright elliptic cylinder fixed in the broadcast's
   orientation (seen side toward the broadcast camera, mirrored onto the back) instead of a
   cutout that turns to face every viewer. SNGS-021 (2026-11-23): broadcast PSNR 22.25 vs 22.31,
   guided line F1 0.36 vs 0.35, flicker 1.99 vs 1.85; players look narrower side-on, as bodies do.
   Both together (`--bg 021_full,024,028,033,037 --own-share 0.5 --iters 15000 --players body`):
   PSNR 22.1, background seen 77%, guided line F1 0.32, flicker 1.91.
4. First-person rendering with geometry-guided generation: `scripts/generate_pov.py`
   (`notebooks/02_generate_pov.ipynb`). SDXL inpaints what no broadcast frame saw.
   Baseline: LTX-Video from the first broadcast frame. Scored by pitch-line F1 against the
   projected markings and by frame-to-frame flicker (no real first-person footage exists).

Run on a Colab GPU runtime by default; tracklab picks CUDA on its own. Stages 1-3 still
run on CPU, just slowly (100 frames took 65 minutes on a Colab CPU).

## Data
SoccerNet Game State Reconstruction (GSR), free download only (no NDA data).
`scripts/get_clip.py` fetches one free validation clip from the public Hugging Face copy (SoccerNet/SN-GSR-2024, no login), reading only that clip out of the 11 GB valid.zip
into `data/SoccerNetGS/valid/`. Data never goes in git.

## Evaluation
- Score against SoccerNet GSR ground truth with its official metric, GS-HOTA.
- The proposal names two metrics. The second (working name EgoPitch) still
  needs a real definition: [decide Thursday]
- Small user study, with our roommates as participants.

## Compute
- Google Colab first. Tejas has Colab Pro for Education. Sanika is checking
  whether a T4 runtime is enough before buying Pro.
- Not using the campus cluster.

## Minimum deliverable and fallback
Minimum: the pipeline running end to end on CPU, without the generation step.
If generation fails or there's no GPU, we demo the minimum and document what
we tried. The course accepts partial results when the process is documented.

## Known risks
- Version conflicts across the repos we build on and between pipeline stages.
- Camera jitter in broadcast footage.
- No GPU when we need one.

## Working with coding agents (proposed, agree Thursday)
- We each run our own Claude Code session. We can't share a chat, so this file
  and the repo are the shared context. Read this file and recent commits first.
- One branch per task: <name>/<topic>. Merge to main by PR.
  PR review by the other person required: [decide Thursday]
- Configs go in `configs/`, scripts in `scripts/`, Colab notebooks in `notebooks/`.
  Clear notebook outputs before committing.
- Never commit data, model weights, outputs or keys.
- An agent that changes a fact in this file updates it in the same commit, and
  adds a dated line under Decisions.
- Course policy on AI-written code: [decide Thursday: check before relying on it]
- Report and write-up text is written in our own words.

## Who does what
- Repo setup: Tejas
- Stage owners: [decide Thursday]

## Decisions
- 2026-11-16: Chose this project over deepfake detection. Proposal submitted on Canvas.
- 2026-11-19: Build on sn-gamestate. Colab first. Free SoccerNet downloads only.
  Setup counts as done when one notebook calibrates a clip and plots players on the pitch on CPU.
- 2026-11-20: GPU runtime is the default (Tejas). CPU was too slow for day-to-day runs.
- 2026-11-23: Stage 3 fits the background from the whole clip and mirrors the seen stand
  into unseen ones; generated fill is baked into the 3D scene, not painted per frame
  (a per-frame whole-image SDXL pass halved line F1 and tripled flicker).
- 2026-11-23: Stage 4 baseline is LTX-Video image-to-video. Only stands and sky are generated;
  the pitch and its markings come from the known layout. Line F1 counts only whites on the
  pitch (found from the camera alone, same region for every method).

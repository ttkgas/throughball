# throughball

First-person view of a soccer game from a chosen player, built from broadcast video.
Computer vision course project. See [AGENTS.md](AGENTS.md) for the plan and conventions.

## Get started (Colab)
Open [notebooks/00_setup.ipynb](notebooks/00_setup.ipynb) in Colab and run all cells.
It installs the sn-gamestate baseline, downloads one free SoccerNet GSR clip, calibrates
the camera with PnLCalib and plots the players on the pitch. Use a GPU runtime
(Runtime > Change runtime type); CPU works but is slow.

## Layout
- `setup.sh`: installs sn-gamestate (pinned) into `third_party/` with a Python 3.9 env
- `scripts/get_clip.py`: fetches one free GSR validation clip from Hugging Face (no login)
- `configs/`: our TrackLab configs, used with `--config-dir configs`
- `notebooks/`: Colab notebooks

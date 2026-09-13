# fly brain mario

Real fruit-fly connectome (MaleCNS v1.0 via neuPrint API) driving Mario.

## quickstart

```
pip install -r requirements.txt
cp .env.example .env
```

Put your neuPrint token in `.env` (get it at neuprint.janelia.org, never commit it).

```
DISPLAY=:0 .venv/bin/python play_real_head.py   # visible: real NES Mario, fly brain
SDL_VIDEODRIVER=dummy .venv/bin/python train_real_rl_v3.py 30  # headless RL, fly W frozen
```

## training datasets

Two inputs the brain learns from:

```
python train_data.py connectome
python train_data.py gameplay
python train_data.py train
```

Outputs in `data/` (gitignored, too big for github):

- `connectome_subset.npz` — real PPL101 wiring + adjacency from the API
- `episodes.npz` — recorded frames, actions, rewards
- `readout.npy` — supervised visual-to-action map the brain loads at play time
```

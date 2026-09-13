# fly brain mario

Real fruit-fly connectome (MaleCNS v1.0 via neuPrint API) driving Mario.

## quickstart

```
pip install -r requirements.txt
cp .env.example .env
```

Put your neuPrint token in `.env` (get it at neuprint.janelia.org, never commit it).

```
python play.py
python fly_mario_real.py
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

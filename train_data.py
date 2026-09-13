import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

def fetch_connectome(out="data/connectome_subset.npz", max_neurons=200):
    import neuprint as neu
    host = os.getenv("NEUPRINT_HOST", "https://neuprint.janelia.org")
    dataset = os.getenv("NEUPRINT_DATASET", "male-cns:v1.0")
    token = os.getenv("NEUPRINT_TOKEN", "")
    if not token:
        raise SystemExit("set NEUPRINT_TOKEN in .env first")
    client = neu.Client(host, dataset=dataset, token=token)
    neu.set_default_client(client)
    crit = neu.NeuronCriteria(type=".*PPL101.*", client=client)
    df, _ = neu.fetch_neurons(crit)
    ids = df["bodyId"].tolist()[:max_neurons]
    adj = neu.fetch_adjacencies(ids, ids)
    os.makedirs("data", exist_ok=True)
    np.savez_compressed(out, body_ids=np.array(ids), edges=adj.to_numpy() if hasattr(adj, "to_numpy") else np.array(adj))
    print(f"saved {len(ids)} neurons -> {out}")
    return out

def record_gameplay(out="data/episodes.npz", steps=2000):
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    base = gym_super_mario_bros.make("SuperMarioBros-v0").unwrapped
    env = JoypadSpace(base, [[], ["right"], ["right", "A"]])
    obs = env.reset()
    frames, actions, rewards = [], [], []
    act = 1
    for _ in range(steps):
        import cv2
        small = cv2.resize(np.asarray(obs), (16, 10)).mean(axis=2) / 255.0
        frames.append(small.astype(np.float32))
        if np.random.rand() < 0.1:
            act = np.random.randint(3)
        obs, reward, done, _ = env.step(act)
        actions.append(act)
        rewards.append(reward)
        if done:
            obs = env.reset()
    os.makedirs("data", exist_ok=True)
    np.savez_compressed(out, frames=np.array(frames), actions=np.array(actions), rewards=np.array(rewards))
    print(f"saved {steps} steps -> {out}")
    return out

def train_readout(ep_path="data/episodes.npz", out="data/readout.npz"):
    d = np.load(ep_path)
    X = d["frames"].reshape(len(d["frames"]), -1)
    y = d["actions"]
    n_feat = X.shape[1]
    n_act = int(y.max()) + 1
    Y = np.zeros((len(y), n_act), dtype=np.float32)
    Y[np.arange(len(y)), y] = 1.0
    W, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    os.makedirs("data", exist_ok=True)
    np.save(out, W.astype(np.float32))
    print(f"saved readout {W.shape} -> {out}")
    return out

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "connectome"
    if mode == "connectome":
        fetch_connectome()
    elif mode == "gameplay":
        record_gameplay()
    elif mode == "train":
        train_readout()
    elif mode == "all":
        fetch_connectome()
        record_gameplay()
        train_readout()

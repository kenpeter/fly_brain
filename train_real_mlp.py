"""Real Mario stage-1, option (b): frozen fly W + small MLP readout.
h(64, fly hidden spikes) -> tanh(32) -> logits(7 Mario buttons).
Only MLP trains. W never touched (asserted).
Run: .venv/bin/python train_real_mlp.py [episodes]
Saves: data/readout_mlp.npz (best), data/mlp_log.npy
"""
import os
import sys
import numpy as np
import cv2

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from fly_brain import FlyBrain

ACTIONS = [["right"], ["right", "A"], ["right", "B"], ["right", "A", "B"],
           ["A"], ["left"], []]
N_ACT, N_HID, N_MLP = len(ACTIONS), 64, 32
REPEAT, GAMMA = 6, 0.99


def zoom_vis(frame):
    f = np.asarray(frame)
    h, w, _ = f.shape
    crop = f[int(h * 0.35):, int(w * 0.30):]
    return (cv2.resize(crop, (16, 10)).mean(axis=2) / 255.0).reshape(-1).astype(np.float32)


def softmax(x):
    x = x - x.max()
    e = np.exp(np.clip(x, -20, 20))
    return e / (e.sum() + 1e-8)


class MLP:
    def __init__(self, rng):
        self.W1 = (rng.normal(0, 0.3, (N_HID, N_MLP))).astype(np.float32)
        self.b1 = np.zeros(N_MLP, dtype=np.float32)
        self.W2 = (rng.normal(0, 0.2, (N_MLP, N_ACT))).astype(np.float32)
        self.b2 = np.zeros(N_ACT, dtype=np.float32)
        # warm start: bias sprint-right + jumps so it explores like v1's best region
        self.b2[:] = np.array([0.3, 0.3, 0.4, 0.2, 0.0, -0.5, -0.5], dtype=np.float32)

    def forward(self, h):
        z = h @ self.W1 + self.b1
        u = np.tanh(z)
        return z, u, u @ self.W2 + self.b2

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]


def run_episode(env, brain, mlp, rng, max_dec=1000, explore=0.10):
    obs = env.reset()
    brain.v[:] = 0
    Hs, Zs, Us, As, Ps, Rs = [], [], [], [], [], []
    max_x, prev_x, steps, dead, flag = 0, 0, 0, False, False
    info = {}
    for _ in range(max_dec):
        _, spikes = brain.step(zoom_vis(obs), dopamine=0.0)
        h = spikes[brain.n_visual:brain.n_visual + brain.n_hidden].astype(np.float32)
        z, u, logits = mlp.forward(h)
        probs = softmax(logits)
        a = int(rng.integers(N_ACT)) if rng.random() < explore else int(rng.choice(N_ACT, p=probs))
        Hs.append(h)
        Zs.append(z)
        Us.append(u)
        As.append(a)
        Ps.append(probs)
        for _ in range(REPEAT):
            out = env.step(a)
            if len(out) == 5:
                obs, _, term, trunc, info = out
                done = term or trunc
            else:
                obs, _, done, info = out
            steps += 1
            if done:
                break
        x = int(info.get("x_pos", prev_x))
        prev_x, max_x = x, max(max_x, x)
        r = (x - prev_x) * 0.0  # placeholder, recomputed below correctly
        Rs.append([x, done])
        if done:
            dead, flag = True, bool(info.get("flag_get", False))
            break
    # rebuild per-step rewards from x trace
    H = np.array(Hs, dtype=np.float32)
    Z = np.array(Zs, dtype=np.float32)
    U = np.array(Us, dtype=np.float32)
    A = np.array(As)
    P = np.array(Ps, dtype=np.float32)
    xs = np.array([r[0] for r in Rs])
    dx = np.diff(np.concatenate([[0], xs]))
    R = dx * 0.05 - 0.01
    if dead:
        R[-1] += 50.0 if flag else -10.0
    G, g = np.zeros(len(R), dtype=np.float32), 0.0
    for t in range(len(R) - 1, -1, -1):
        g = R[t] + GAMMA * g
        G[t] = g
    return H, Z, U, A, P, G, max_x, steps, dead, flag


def main():
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rng = np.random.default_rng(31)
    base = gym_super_mario_bros.make("SuperMarioBros-v0").unwrapped
    env = JoypadSpace(base, [list(a) for a in ACTIONS])
    brain = FlyBrain(n_visual=160, n_hidden=64, n_motor=N_ACT)
    brain.W = np.load("data/connectome_5group.npz")["W"].astype(np.float32)
    brain.plasticity = 0.0
    W0 = brain.W.copy()
    mlp = MLP(rng)
    lr, ent = 0.02, 0.01
    best_x, log = -1, []
    for ep in range(n_ep):
        H, Z, U, A, P, G, max_x, steps, dead, flag = run_episode(env, brain, mlp, rng)
        log.append((float(max_x), int(dead), int(flag)))
        adv = G - G.mean()
        Oh = np.zeros_like(P)
        Oh[np.arange(len(A)), A] = 1.0
        D = ((Oh - P) * adv[:, None]) / len(A)  # d logits
        # entropy bonus: dH/dlogit = p * (-logp - H)
        logP = np.log(P + 1e-8)
        Hent = -(P * logP).sum(axis=1, keepdims=True)
        D_ent = P * (-logP - Hent) / len(A)
        gW2 = U.T @ (D + ent * D_ent)
        gb2 = D.sum(axis=0)
        DU = D @ mlp.W2.T * (1.0 - U ** 2)
        gW1 = H.T @ DU / 1.0
        gb1 = DU.sum(axis=0)
        # entropy pull on W2 too (keep it simple: on logits via D already + small reg)
        mlp.W2 += lr * gW2
        mlp.b2 += lr * gb2
        mlp.W1 += lr * np.clip(gW1, -1, 1)
        mlp.b1 += lr * np.clip(gb1, -1, 1)
        for p in mlp.params():
            np.clip(p, -3.0, 3.0, out=p)
        if max_x > best_x:
            best_x = max_x
            np.savez_compressed("data/readout_mlp.npz", W1=mlp.W1, b1=mlp.b1,
                                W2=mlp.W2, b2=mlp.b2)
        print(f"ep{ep:02d} x={max_x} Gmean={G.mean():+.2f} steps={steps} "
              f"dead={dead} flag={flag} best_x={best_x}", flush=True)
        if flag:
            print("STAGE CLEAR!", flush=True)
            break
    assert np.array_equal(brain.W, W0), "fly W must stay frozen"
    np.save("data/mlp_log.npy", np.array(log))
    print(f"DONE frozen-ok best_x={best_x} (flag needs ~3200)")


if __name__ == "__main__":
    main()

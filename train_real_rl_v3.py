"""Real Mario stage-1 training v3: per-step dense returns + zoom vision, fly W frozen.
- Zoom-ahead 160-dim encoding (threats bigger).
- Per-decision reward: dx*0.5 - step cost - death@sprint, discounted returns.
- Readout R only. W never touched.
Run: .venv/bin/python train_real_rl_v3.py [episodes]
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
N_ACT = len(ACTIONS)
REPEAT = 6
GAMMA = 0.99


def zoom_vis(frame):
    f = np.asarray(frame)
    h, w, _ = f.shape
    crop = f[int(h * 0.35):, int(w * 0.30):]
    small = cv2.resize(crop, (16, 10))
    return (small.mean(axis=2) / 255.0).reshape(-1).astype(np.float32)


def softmax(x):
    x = x - x.max()
    e = np.exp(np.clip(x, -20, 20))
    return e / (e.sum() + 1e-8)


def run_episode(env, brain, R, rng, max_dec=1000, explore=0.10):
    obs = env.reset()
    brain.v[:] = 0
    Hs, As, Ps, Rs = [], [], [], []
    max_x, prev_x, steps, dead, flag = 0, 0, 0, False, False
    info = {}
    for _ in range(max_dec):
        _, spikes = brain.step(zoom_vis(obs), dopamine=0.0)
        h = spikes[brain.n_visual:brain.n_visual + brain.n_hidden].astype(np.float32)
        probs = softmax(h @ R)
        a = int(rng.integers(N_ACT)) if rng.random() < explore else int(rng.choice(N_ACT, p=probs))
        Hs.append(h)
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
        dx = x - prev_x
        prev_x = x
        max_x = max(max_x, x)
        r = dx * 0.05 - 0.01
        if done:
            dead = True
            flag = bool(info.get("flag_get", False))
            r += 50.0 if flag else -10.0
            Rs.append(r)
            break
        Rs.append(r)
    # discounted returns
    G, g = np.zeros(len(Rs), dtype=np.float32), 0.0
    for t in range(len(Rs) - 1, -1, -1):
        g = Rs[t] + GAMMA * g
        G[t] = g
    return (np.array(Hs, dtype=np.float32), np.array(As), np.array(Ps), G,
            max_x, steps, dead, flag)


def main():
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rng = np.random.default_rng(21)
    base = gym_super_mario_bros.make("SuperMarioBros-v0").unwrapped
    env = JoypadSpace(base, [list(a) for a in ACTIONS])
    brain = FlyBrain(n_visual=160, n_hidden=64, n_motor=N_ACT)
    brain.W = np.load("data/connectome_5group.npz")["W"].astype(np.float32)
    brain.plasticity = 0.0
    W0 = brain.W.copy()
    if os.path.exists("data/readout_real_rl_v1.npy"):
        R = np.load("data/readout_real_rl_v1.npy").astype(np.float32)
        print("warm start from v1 best (x=1494)", flush=True)
    else:
        R = (brain.W[160:224, 224:231] * 0.5).copy()
    lr, ent = 0.03, 0.01
    best_x, log = -1, []
    for ep in range(n_ep):
        H, A, P, G, max_x, steps, dead, flag = run_episode(env, brain, R, rng)
        log.append((float(max_x), int(dead), int(flag)))
        adv = G - G.mean()
        Oh = np.zeros_like(P)
        Oh[np.arange(len(A)), A] = 1.0
        PG = ((Oh - P) * adv[:, None]).T @ H / len(A)
        EG = -((np.log(P + 1e-8) + 1.0)).T @ H / len(A)
        R = np.clip(R + lr * (PG.T + ent * EG.T), -3.0, 3.0)
        if max_x > best_x:
            best_x = max_x
            np.save("data/readout_real_rl.npy", R.astype(np.float32))
        print(f"ep{ep:02d} x={max_x} Gmean={G.mean():+.2f} steps={steps} "
              f"dead={dead} flag={flag} best_x={best_x}", flush=True)
        if flag:
            print("STAGE CLEAR!", flush=True)
            break
    assert np.array_equal(brain.W, W0)
    np.save("data/real_rl_log_v3.npy", np.array(log))
    print(f"DONE frozen-ok best_x={best_x} (flag needs ~3200)")


if __name__ == "__main__":
    main()

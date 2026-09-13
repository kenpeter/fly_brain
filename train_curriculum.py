"""Stage 1-1 ONLY, max-fly: the ONLY plastic weights are the fly's own
motor block W[160:224,224:231] (MBON-equivalent). Optic->hidden +
hidden->hidden frozen forever. Pristine backup: data/connectome_5group_pristine.npz
Run: .venv/bin/python train_curriculum.py [episodes]
Saves: data/motorblock_1_1.npy, data/curriculum.npy
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
H0, H1, M0 = 160, 224, 224
REPEAT, GAMMA = 3, 0.99  # finer jump timing (was 6)


def zoom_vis(frame):
    # FOVEA: tight ahead-at-feet band (threats ~2x bigger in same 160 dims).
    # Sky above + behind-Mario discarded; same frozen optic IDs driven.
    f = np.asarray(frame)
    h, w, _ = f.shape
    crop = f[int(h * 0.45):int(h * 0.95), int(w * 0.25):int(w * 0.75)]
    return (cv2.resize(crop, (16, 10)).mean(axis=2) / 255.0).reshape(-1).astype(np.float32)


def softmax(x):
    x = x - x.max()
    e = np.exp(np.clip(x, -20, 20))
    return e / (e.sum() + 1e-8)


def run_episode(env, brain, rng, max_dec=1000, explore=0.10):
    obs = env.reset()
    brain.v[:] = 0
    Hs, As, Ps, Xs = [], [], [], []
    max_x, prev_x, steps, dead, flag = 0, 0, 0, False, False
    info = {}
    for _ in range(max_dec):
        Wm = brain.W[H0:H1, M0:M0 + N_ACT]
        _, spikes, _ = brain.step(zoom_vis(obs), dopamine=0.0)
        h = spikes[brain.n_visual:brain.n_visual + 64].astype(np.float32)
        probs = softmax(h @ Wm)
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
            # stay in 1-1: leaving the stage ends the episode
            if (int(info.get("world", 1)), int(info.get("stage", 1))) != (1, 1):
                done = True
            if done:
                break
        x = int(info.get("x_pos", prev_x))
        prev_x, max_x = x, max(max_x, x)
        Xs.append(x)
        if done:
            dead = True
            flag = bool(info.get("flag_get", False))
            break
    H = np.array(Hs, dtype=np.float32)
    A = np.array(As)
    P = np.array(Ps, dtype=np.float32)
    dx = np.diff(np.concatenate([[0], np.array(Xs)]))
    R = dx * 0.05 - 0.01
    if dead:
        R[-1] += 50.0 if flag else -10.0
    G, g = np.zeros(len(R), dtype=np.float32), 0.0
    for t in range(len(R) - 1, -1, -1):
        g = R[t] + GAMMA * g
        G[t] = g
    return H, A, P, G, max_x, steps, dead, flag


def main():
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    rng = np.random.default_rng(41)
    base = gym_super_mario_bros.make("SuperMarioBros-v0").unwrapped
    env = JoypadSpace(base, [list(a) for a in ACTIONS])
    brain = FlyBrain(n_visual=160, n_hidden=64, n_motor=N_ACT)
    brain.W = np.load("data/connectome_5group.npz")["W"].astype(np.float32)
    brain.plasticity = 0.0
    pristine = np.load("data/connectome_5group_pristine.npz")["W"].astype(np.float32)
    mb_path = "data/motorblock_1_1.npy"
    if os.path.exists(mb_path):
        brain.W[H0:H1, M0:M0 + N_ACT] = np.load(mb_path).astype(np.float32)
        print("resumed motor block", flush=True)
    lr, ent = 0.03, 0.01
    best_x, clears, log = -1, 0, []
    for ep in range(n_ep):
        H, A, P, G, max_x, steps, dead, flag = run_episode(env, brain, rng)
        Wm = brain.W[H0:H1, M0:M0 + N_ACT]
        adv = G - G.mean()
        Oh = np.zeros_like(P)
        Oh[np.arange(len(A)), A] = 1.0
        D = ((Oh - P) * adv[:, None]) / len(A)
        logP = np.log(P + 1e-8)
        He = -(P * logP).sum(axis=1, keepdims=True)
        D_ent = P * (-logP - He) / len(A)
        Wm += lr * (H.T @ (D + ent * D_ent))
        np.clip(Wm, -3.0, 3.0, out=Wm)
        if max_x > best_x:
            best_x = max_x
            np.save(mb_path, Wm.astype(np.float32))
        if flag:
            clears += 1
        chk = brain.W.copy()
        chk[H0:H1, M0:M0 + N_ACT] = pristine[H0:H1, M0:M0 + N_ACT]
        frozen_ok = bool(np.array_equal(chk, pristine))
        log.append((ep, max_x, int(dead), int(flag)))
        np.save("data/curriculum.npy", np.array(log))
        print(f"ep{ep:02d} x={max_x} Gmean={G.mean():+.2f} steps={steps} "
              f"dead={dead} flag={flag} best={best_x} clears={clears} frozen={frozen_ok}",
              flush=True)
        if clears >= 2:
            print("1-1 RELIABLE (2 flags).", flush=True)
            break
    print(f"DONE best_x={best_x} clears={clears} (flag needs ~3200)")


if __name__ == "__main__":
    main()

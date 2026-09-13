"""Vectorized 1-1 RL, max-fly: N parallel Mario envs (threads), ONE shared fly.
Frozen W (53k real), only shared motor block W[160:224,224:231] learns from
batched diverse trajectories. Fovea eyes, REPEAT 3.
Run: .venv/bin/python train_vec.py [waves] [nenvs]
Saves: data/motorblock_1_1.npy, data/vec_log.npy
"""
import os
import sys
import time
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from fly_brain import FlyBrain

ACTIONS = [["right"], ["right", "A"], ["right", "B"], ["right", "A", "B"],
           ["A"], ["left"], []]
N_ACT = len(ACTIONS)
H0, H1, M0 = 160, 224, 224
REPEAT, GAMMA = 3, 0.99
REPEAT_JUMP = 8  # A-button actions hold longer = higher jumps (tall pipes)
JUMP_ACTS = {1, 3, 4}  # actions containing 'A'


def fovea(frame):
    f = np.asarray(frame)
    h, w, _ = f.shape
    crop = f[int(h * 0.45):int(h * 0.95), int(w * 0.25):int(w * 0.75)]
    return (cv2.resize(crop, (16, 10)).mean(axis=2) / 255.0).reshape(-1).astype(np.float32)


def softmax(x):
    x = x - x.max()
    e = np.exp(np.clip(x, -20, 20))
    return e / (e.sum() + 1e-8)


def make_env():
    base = gym_super_mario_bros.make("SuperMarioBros-v0").unwrapped
    return JoypadSpace(base, [list(a) for a in ACTIONS])


def run_one(seed, Wm, max_dec=1000, explore=0.10):
    """Own env + own voltages, SHARED read-only motor block. Returns trace."""
    rng = np.random.default_rng(seed)
    env = make_env()
    b = FlyBrain(n_visual=160, n_hidden=64, n_motor=N_ACT)
    b.W = np.load("data/connectome_5group.npz")["W"].astype(np.float32)
    b.plasticity = 0.0
    obs = env.reset()
    Hs, As, Ps, Rs = [], [], [], []
    max_x, prev_x, steps, dead, flag = 0, 0, 0, False, False
    best, still, prev_score = 0, 0, 0
    info = {}
    for _ in range(max_dec):
        _, spikes, _ = b.step(fovea(obs), dopamine=0.0)
        h = spikes[b.n_visual:b.n_visual + 64].astype(np.float32)
        probs = softmax(h @ Wm)
        a = int(rng.integers(N_ACT)) if rng.random() < explore else int(rng.choice(N_ACT, p=probs))
        Hs.append(h)
        As.append(a)
        Ps.append(probs)
        rep = REPEAT_JUMP if a in JUMP_ACTS else REPEAT
        peak_h, dscore = 0.0, 0
        for _ in range(rep):
            out = env.step(a)
            if len(out) == 5:
                obs, _, term, trunc, info = out
                done = term or trunc
            else:
                obs, _, done, info = out
            steps += 1
            peak_h = max(peak_h, 79 - int(info.get("y_pos", 79)))  # height above ground
            dscore += int(info.get("score", 0)) - prev_score
            prev_score = int(info.get("score", 0))
            if (int(info.get("world", 1)), int(info.get("stage", 1))) != (1, 1):
                done = True
            if done:
                break
        x = int(info.get("x_pos", prev_x))
        moved = x - prev_x
        prev_x = x
        # GitHub patterns: new-best-only progress (backtracking free),
        # jump-for-height, score = kills/coins, stuck penalty
        r = max(0, x - best) * 0.05 - 0.01
        best = max(best, x)
        max_x = best
        r += min(peak_h, 60) * 0.03 if moved > -1 else 0.0  # jump while advancing
        r += dscore * 0.01  # stomps (+100) / coins
        still = still + 1 if moved == 0 else 0
        if still > 8:
            r -= 0.5  # wedged (pipe pocket) -> move
        Rs.append(r)
        if done:
            dead, flag = True, bool(info.get("flag_get", False))
            break
    env.close()
    H = np.array(Hs, dtype=np.float32)
    A = np.array(As)
    P = np.array(Ps, dtype=np.float32)
    R = np.array(Rs, dtype=np.float32)  # already shaped per-decision above
    if dead:
        R[-1] += 50.0 if flag else -10.0
    G, g = np.zeros(len(R), dtype=np.float32), 0.0
    for t in range(len(R) - 1, -1, -1):
        g = R[t] + GAMMA * g
        G[t] = g
    return H, A, P, G, max_x, steps, dead, flag


def main():
    n_waves = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    W = np.load("data/connectome_5group.npz")["W"].astype(np.float32)
    pristine = np.load("data/connectome_5group_pristine.npz")["W"].astype(np.float32)
    mb_path = "data/motorblock_1_1.npy"
    Wm = np.load(mb_path).astype(np.float32) if os.path.exists(mb_path) else W[H0:H1, M0:M0 + N_ACT].copy()
    print(f"start motor block from {mb_path if os.path.exists(mb_path) else 'pristine init'}", flush=True)
    lr, ent, seed0 = 0.03, 0.01, 1000
    best_x, clears, log = -1, 0, []
    live = open("data/vec_progress.log", "a", buffering=1)  # live per-episode log
    live.write(f"--- run {time.strftime('%F %T')} waves={n_waves} envs={n_envs} ---\n")
    pool = ThreadPoolExecutor(max_workers=n_envs)
    for wv in range(n_waves):
        t0 = time.time()
        Wm_ro = Wm.copy()  # workers read-only snapshot
        fut2id = {pool.submit(run_one, seed0 + wv * n_envs + i, Wm_ro): i for i in range(n_envs)}
        trajs, xs, fl = [], [], 0
        num = np.zeros_like(Wm)  # batched gradient into shared motor block
        for f in as_completed(fut2id):
            H, A, P, G, max_x, steps, dead, flag = f.result()
            trajs.append((H, A, P, G, max_x, steps, dead, flag))
            xs.append(max_x)
            fl += flag
            line = (f"wave{wv:02d} ep{fut2id[f]:02d} x={max_x} steps={steps} "
                    f"dead={dead} flag={flag} t={time.time()-t0:.0f}s")
            print(line, flush=True)
            live.write(line + "\n")
            adv = G - G.mean()
            Oh = np.zeros_like(P)
            Oh[np.arange(len(A)), A] = 1.0
            D = ((Oh - P) * adv[:, None]) / len(A)
            logP = np.log(P + 1e-8)
            He = -(P * logP).sum(axis=1, keepdims=True)
            num += H.T @ (D + ent * P * (-logP - He) / len(A))
        Wm = np.clip(Wm + lr * num / n_envs, -3.0, 3.0)
        wave_best = max(xs)
        if wave_best > best_x:
            best_x = wave_best
            np.save(mb_path, Wm.astype(np.float32))
        clears += fl
        # frozen check on shared W
        chk = W.copy()
        frozen_ok = bool(np.array_equal(chk, pristine))
        log.append((wv, float(np.mean(xs)), float(wave_best), int(fl)))
        np.save("data/vec_log.npy", np.array(log))
        print(f"wave{wv:02d} mean_x={np.mean(xs):.0f} best_x={wave_best} flags={fl} "
              f"overall_best={best_x} clears={clears} frozen={frozen_ok} "
              f"{time.time()-t0:.0f}s", flush=True)
        seed0 += 100000
        if clears >= 2:
            print("1-1 RELIABLE.", flush=True)
            break
    print(f"DONE best_x={best_x} clears={clears}")


if __name__ == "__main__":
    main()

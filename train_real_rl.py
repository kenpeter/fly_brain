"""Real NES Mario Stage 1-1 training: freeze fly W, train readout R only.
State: 160-dim frame. Hidden: 64 real (CX16+MB32+DN16). Actions: 7 Mario buttons.
Reward: +dx (x_pos progress), death -5, flag +50.
Saves: data/readout_real_rl.npy (best), data/real_rl_log.npy
Run: .venv/bin/python train_real_rl.py [episodes]
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
           ["A"], ["left"], []]  # [] = noop
N_ACT = len(ACTIONS)
REPEAT = 6  # longer button hold = higher jumps (needed for pipes/goombas)
DEATH_PEN = 20.0  # must exceed any progress reward: dying far < surviving near
STEP_COST = 0.0008  # time pressure: standing still scores nothing
ENT_COEF = 0.02  # keep jump actions alive (policy collapsed to always-right before)


def frame_to_vis(frame):
    small = cv2.resize(np.asarray(frame), (16, 10))
    gray = small.mean(axis=2) / 255.0
    return gray.reshape(-1).astype(np.float32)


def softmax(x):
    x = x - x.max()
    e = np.exp(np.clip(x, -20, 20))
    return e / (e.sum() + 1e-8)


def run_episode(env, brain, R, rng, max_dec=1200, explore=0.12):
    obs = env.reset()
    brain.v[:] = 0
    Hs, As, Ps = [], [], []
    max_x, prev_x, steps = 0, 0, 0
    dead, flag = False, False
    for _ in range(max_dec):
        vis = frame_to_vis(obs)
        _, spikes = brain.step(vis, dopamine=0.0)  # frozen W
        h = spikes[brain.n_visual:brain.n_visual + brain.n_hidden].astype(np.float32)
        logits = h @ R
        probs = softmax(logits)
        if rng.random() < explore:
            a = int(rng.integers(N_ACT))
        else:
            a = int(rng.choice(N_ACT, p=probs))
        Hs.append(h)
        As.append(a)
        Ps.append(probs)
        for _ in range(REPEAT):
            out = env.step(a)
            if len(out) == 5:
                obs, rew, term, trunc, info = out
                done = term or trunc
            else:
                obs, rew, done, info = out
            steps += 1
            x = int(info.get("x_pos", prev_x))
            if x > max_x:
                max_x = x
            prev_x = x
            if done:
                break
        if done:
            dead = True
            flag = bool(info.get("flag_get", False))
            break
    H = np.array(Hs, dtype=np.float32)
    A = np.array(As, dtype=np.int64)
    P = np.array(Ps, dtype=np.float32)
    # episodic return: progress minus big death penalty minus time (anti-stall)
    ret = (max_x * 0.01 - (DEATH_PEN if (dead and not flag) else 0.0)
           - steps * STEP_COST + (50.0 if flag else 0.0))
    return H, A, P, ret, max_x, steps, dead, flag


def main():
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rng = np.random.default_rng(11)
    base = gym_super_mario_bros.make("SuperMarioBros-v0").unwrapped
    env = JoypadSpace(base, [list(a) for a in ACTIONS])
    brain = FlyBrain(n_visual=160, n_hidden=64, n_motor=N_ACT)
    d = np.load("data/connectome_5group.npz")
    assert d["W"].shape == (231, 231), d["W"].shape
    brain.W = d["W"].astype(np.float32)
    brain.plasticity = 0.0
    W0 = brain.W.copy()
    R = (brain.W[160:224, 224:231] * 0.5).copy()  # init from real DN->motor
    if os.getenv("FRESH", "") != "1" and os.path.exists("data/readout_real_rl.npy"):
        R = np.load("data/readout_real_rl.npy").astype(np.float32)
        print("resumed R from data/readout_real_rl.npy", flush=True)
    bl = 6.0  # carry baseline from previous 25-ep run (ended ~5.99)
    lr, bl, gamma = 0.05, bl, 0.92
    prev = np.load("data/real_rl_log.npy").tolist() if os.path.exists("data/real_rl_log.npy") else []
    best_x, best_R, log = -1, R.copy(), list(prev)
    if log:
        best_x = max(r[0] for r in log)
        print(f"prev best_x={best_x} over {len(log)} eps", flush=True)
    for ep in range(n_ep):
        H, A, P, ret, max_x, steps, dead, flag = run_episode(env, brain, R, rng)
        log.append((float(max_x), float(ret), int(dead), int(flag)))
        bl = gamma * bl + (1 - gamma) * ret
        adv = ret - bl
        # policy gradient: (onehot - p) * adv * h  -> R, plus entropy bonus
        Oh = np.zeros_like(P)
        Oh[np.arange(len(A)), A] = 1.0
        G = ((Oh - P) * adv).T @ H / max(len(A), 1)  # (7,64)
        ent_G = -((np.log(P + 1e-8) + 1.0)).T @ H / max(len(A), 1)
        R += lr * (G.T + ENT_COEF * ent_G.T)
        R[:] = np.clip(R, -3.0, 3.0)
        if max_x > best_x:
            best_x, best_R = max_x, R.copy()
            np.save("data/readout_real_rl.npy", best_R.astype(np.float32))
        np.save("data/real_rl_log.npy", np.array(log))
        print(f"ep{ep:02d} x={max_x} ret={ret:.2f} base={bl:.2f} "
              f"steps={steps} dead={dead} flag={flag} best_x={best_x}", flush=True)
        if flag:
            print("STAGE CLEAR!", flush=True)
            break
    assert np.array_equal(brain.W, W0), "fly W must stay frozen"
    print(f"DONE frozen-ok best_x={best_x} (need ~3000+ for flag)")


if __name__ == "__main__":
    main()

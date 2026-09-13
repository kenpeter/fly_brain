"""Teacher warm-start: scripted full-res jump teacher demonstrates 1-1,
fly watches through its own 160-dim eyes; clone (hidden spikes -> action)
into the in-fly motor block. Then RL resumes from there.
Run: .venv/bin/python teach_clone.py [teacher_episodes]
Saves: data/motorblock_1_1.npy (cloned), keeps data/motorblock_rl40.npy backup
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
REPEAT = 6
SPRINT, JUMP = 2, 3


def zoom_vis(frame):
    f = np.asarray(frame)
    h, w, _ = f.shape
    crop = f[int(h * 0.35):, int(w * 0.30):]
    return (cv2.resize(crop, (16, 10)).mean(axis=2) / 255.0).reshape(-1).astype(np.float32)


def teacher_action(frame, hold, stall=0):
    """Full-res eyes (teacher only): jump if threat/gap ahead, else sprint.
    stall>0: run-up recovery (back off, then sprint-jump) for tall pipes."""
    if stall > 24:  # finished backing off -> full run-up jump
        return JUMP, 22, 0
    if stall > 0:  # backing off the wall
        return 5, 0, stall + 1
    if hold > 0:
        return JUMP, hold - 1, 0
    f = np.asarray(frame)
    h, w, _ = f.shape
    box = f[int(h * 0.35):int(h * 0.90), int(w * 0.30):int(w * 0.95)].astype(int)
    R, G, B = box[:, :, 0], box[:, :, 1], box[:, :, 2]
    black = ((R < 80) & (G < 80) & (B < 80)).mean()  # outlines: goomba, pipe rim, ?-blocks
    enemy = ((R > 130) & (G > 50) & (G < 150) & (B < 90)).mean()  # goomba brown
    strip = f[int(h * 0.88):int(h * 0.97), int(w * 0.30):int(w * 0.95)]
    ground = (strip.mean(axis=2) > 100).mean()
    if black > 0.022 or enemy > 0.06 or ground < 0.70:
        return JUMP, 20, 0
    return SPRINT, 0, 0


def main():
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    base = gym_super_mario_bros.make("SuperMarioBros-v0").unwrapped
    env = JoypadSpace(base, [list(a) for a in ACTIONS])
    brain = FlyBrain(n_visual=160, n_hidden=64, n_motor=N_ACT)
    brain.W = np.load("data/connectome_5group.npz")["W"].astype(np.float32)
    brain.plasticity = 0.0
    Hs, As = [], []
    best_tx = 0
    for ep in range(n_ep):
        obs = env.reset()
        brain.v[:] = 0
        hold, stall, maxx, lastx, still = 0, 0, 0, 0, 0
        for _ in range(1000):
            _, spikes, _ = brain.step(zoom_vis(obs), dopamine=0.0)
            h = spikes[brain.n_visual:brain.n_visual + 64].astype(np.float32)
            a, hold, stall = teacher_action(obs, hold, stall)
            Hs.append(h)
            As.append(a)
            for _ in range(REPEAT):
                out = env.step(a)
                obs, _, done, info = out[0], out[1], out[2], out[3]
                if (int(info.get("world", 1)), int(info.get("stage", 1))) != (1, 1):
                    done = True
                if done:
                    break
            x = int(info.get("x_pos", 0))
            maxx = max(maxx, x)
            still = still + 1 if x == lastx else 0
            lastx = x
            if still > 12 and stall == 0 and not done:
                stall = 1  # wedged: start run-up recovery
            if done:
                break
        best_tx = max(best_tx, maxx)
        print(f"teacher ep{ep} x={maxx} flag={info.get('flag_get', False)}", flush=True)
    H = np.array(Hs, dtype=np.float32)
    A = np.array(As)
    print(f"teacher best x={best_tx}, demos={len(A)}")
    # clone: least-squares hidden -> onehot(action)
    Oh = np.zeros((len(A), N_ACT), dtype=np.float32)
    Oh[np.arange(len(A)), A] = 1.0
    Wc, _, _, _ = np.linalg.lstsq(H, Oh, rcond=None)
    Wc = np.clip(Wc * 2.0, -3.0, 3.0).astype(np.float32)
    np.save("data/motorblock_1_1.npy", Wc)
    # greedy check with cloned block
    brain.W[H0:H1, M0:M0 + N_ACT] = Wc
    obs = env.reset()
    brain.v[:] = 0
    maxx = 0
    info = {}
    for _ in range(1000):
        _, spikes, _ = brain.step(zoom_vis(obs), dopamine=0.0)
        h = spikes[brain.n_visual:brain.n_visual + 64].astype(np.float32)
        a = int(np.argmax(h @ Wc))
        for _ in range(REPEAT):
            out = env.step(a)
            obs, _, done, info = out[0], out[1], out[2], out[3]
            if (int(info.get("world", 1)), int(info.get("stage", 1))) != (1, 1):
                done = True
            if done:
                break
        maxx = max(maxx, int(info.get("x_pos", 0)))
        if done:
            break
    print(f"CLONED greedy: x={maxx} flag={info.get('flag_get', False)}")


if __name__ == "__main__":
    main()

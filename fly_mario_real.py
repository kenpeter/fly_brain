import os
import time
import numpy as np
import cv2
from dotenv import load_dotenv
load_dotenv()
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from fly_brain import FlyBrain

ACTIONS = [
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
    ["A"],
    ["left"],
    ["noop"],
]

def frame_to_vis(frame, n):
    small = cv2.resize(frame, (16, 10))
    gray = small.mean(axis=2) / 255.0
    v = gray.reshape(-1).astype(np.float32)
    if len(v) > n:
        v = v[:n]
    if len(v) < n:
        v = np.pad(v, (0, n - len(v)))
    return v

def main():
    brain = FlyBrain(n_visual=160, n_hidden=64, n_motor=len(ACTIONS))
    ok, msg = brain.load_real_subset()
    print(msg, flush=True)
    base = gym_super_mario_bros.make("SuperMarioBros-v0").unwrapped
    env = JoypadSpace(base, [list(a) if a != ["noop"] else [] for a in ACTIONS])
    obs = env.reset()
    env.render()
    deaths = 0
    prev_lives = 3
    while True:
        vis = frame_to_vis(np.asarray(obs), brain.n_visual)
        motor, _ = brain.step(vis, dopamine=0.0)
        act = int(np.argmax(motor)) if motor.sum() > 0 else 0
        if np.random.rand() < 0.05:
            act = np.random.randint(len(ACTIONS))
        out = env.step(act)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = terminated or truncated
        else:
            obs, reward, done, info = out
        env.render()
        lives = info.get("life", prev_lives)
        if reward is not None and reward < -5:
            for _ in range(6):
                brain.step(np.zeros(brain.n_visual, dtype=np.float32), dopamine=-1.0)
        if done:
            deaths += 1
            print(f"mario done deaths={deaths} info={info}", flush=True)
            for _ in range(8):
                brain.step(np.zeros(brain.n_visual, dtype=np.float32), dopamine=-1.0)
            time.sleep(0.5)
            obs = env.reset()
            prev_lives = 3
        else:
            prev_lives = lives

if __name__ == "__main__":
    main()

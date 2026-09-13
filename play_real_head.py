"""Visible real-Mario player: NES frames -> pygame window on DISPLAY, fly brain acts.
Run: DISPLAY=:0 .venv/bin/python play_real_head.py [seconds]
Saves shots to /tmp/opencode/live_real_*.png
"""
import os
import sys
import time
import numpy as np
import cv2

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import pygame
from fly_brain import FlyBrain

ACTIONS = [["right"], ["right", "A"], ["right", "B"], ["right", "A", "B"],
           ["A"], ["left"], []]


def zoom_vis(frame):
    f = np.asarray(frame)
    h, w, _ = f.shape
    crop = f[int(h * 0.45):int(h * 0.95), int(w * 0.25):int(w * 0.75)]
    return (cv2.resize(crop, (16, 10)).mean(axis=2) / 255.0).reshape(-1).astype(np.float32)


def softmax(x):
    x = x - x.max()
    e = np.exp(np.clip(x, -20, 20))
    return e / (e.sum() + 1e-8)


def main():
    secs = int(sys.argv[1]) if len(sys.argv) > 1 else 75
    pygame.init()
    win = pygame.display.set_mode((512, 480))
    pygame.display.set_caption("fly brain real mario")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)
    base = gym_super_mario_bros.make("SuperMarioBros-v0").unwrapped
    env = JoypadSpace(base, [list(a) for a in ACTIONS])
    brain = FlyBrain(n_visual=160, n_hidden=64, n_motor=7)
    brain.W = np.load("data/connectome_5group.npz")["W"].astype(np.float32)
    brain.W[160:224, 224:231] = np.load("data/motorblock_1_1.npy").astype(np.float32)
    brain.plasticity = 0.0
    print("in-fly motor block + fovea eyes", flush=True)
    rng = np.random.default_rng(9)
    obs = env.reset()
    brain.v[:] = 0
    t0, t, deaths, maxx, shots = time.time(), 0, 0, 0, 0
    while time.time() - t0 < secs:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return
        _, sp, _ = brain.step(zoom_vis(obs), dopamine=0.0)
        p = softmax(sp[160:224].astype(np.float32) @ brain.W[160:224, 224:231])
        a = int(rng.choice(7, p=p))
        rep = 8 if a in {1, 3, 4} else 3  # hold A longer = higher jumps
        for _ in range(rep):
            out = env.step(a)
            obs, _, done, info = out[0], out[1], out[2], out[3]
            if done:
                break
        maxx = max(maxx, int(info.get("x_pos", 0)))
        fr = np.asarray(obs)
        surf = pygame.surfarray.make_surface(np.transpose(fr, (1, 0, 2)))
        surf = pygame.transform.scale(surf, (512, 480))
        win.blit(surf, (0, 0))
        txt = font.render(f"x={info.get('x_pos',0)} max={maxx} deaths={deaths} act={ACTIONS[a]}", True, (255, 255, 255))
        win.blit(txt, (8, 8))
        pygame.display.flip()
        if t % 150 == 0:
            pygame.image.save(win, f"/tmp/opencode/live_real_{shots}.png")
            shots += 1
        t += 1
        clock.tick(60)
        if done:
            deaths += 1
            print(f"died x={info.get('x_pos')} maxx={maxx} deaths={deaths}", flush=True)
            obs = env.reset()
            brain.v[:] = 0
    print(f"LIVE done maxx={maxx} deaths={deaths}", flush=True)


if __name__ == "__main__":
    main()

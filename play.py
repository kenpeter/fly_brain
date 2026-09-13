import os
import time
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from mario_clone import SideScroller
from fly_brain import FlyBrain

def main():
    env = SideScroller()
    brain = FlyBrain(n_visual=64, n_hidden=64, n_motor=2)
    ok, msg = brain.load_real_subset()
    print(msg)
    obs = env.reset()
    deaths = 0
    while True:
        for e in __import__("pygame").event.get():
            if e.type == __import__("pygame").QUIT:
                return
        vis = env.frame_small(w=16, h=8)
        if len(vis) > brain.n_visual:
            vis = vis[:brain.n_visual]
        if len(vis) < brain.n_visual:
            vis = np.pad(vis, (0, brain.n_visual - len(vis)))
        motor, _ = brain.step(vis, dopamine=0.0)
        act_right = bool(motor[0] > 0 or True)
        act_jump = bool(motor[1] > 0)
        if np.random.rand() < 0.02:
            act_jump = not act_jump
        env.physics(act_right, act_jump)
        env.draw()
        env.clock.tick(60)
        if env.dead:
            deaths += 1
            print(f"died score={env.score} deaths={deaths} -> dopamine punish")
            for _ in range(8):
                brain.step(np.zeros_like(vis), dopamine=-1.0)
            time.sleep(0.4)
            env.reset()

if __name__ == "__main__":
    main()

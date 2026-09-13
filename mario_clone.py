import pygame
import numpy as np

W, H = 640, 360
GROUND = H - 60

class SideScroller:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("fly mario clone")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.x = 40.0
        self.y = float(GROUND)
        self.vy = 0.0
        self.world = 0.0
        self.speed = 2.2
        self.dead = False
        self.steps = 0
        self.pipes = [300, 620, 1050, 1500, 1950]
        self.gaps = [850, 1750]
        self.score = 0
        return self.obs()

    def obs(self):
        arr = pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
        return arr

    def physics(self, act_right, act_jump):
        if self.dead:
            return
        if act_right:
            self.world += self.speed
            self.x = min(self.x + 0.4, W * 0.4)
            self.score += 1
        if act_jump and self.y >= GROUND - 1:
            self.vy = -11.5
        self.vy += 0.6
        self.y += self.vy
        if self.y > GROUND:
            self.y = float(GROUND)
            self.vy = 0.0
        px = self.world + self.x
        for p in self.pipes:
            if abs(px - p) < 14 and self.y > GROUND - 48:
                self.dead = True
        for g in self.gaps:
            if abs(px - g) < 18 and self.y >= GROUND - 1:
                self.dead = True
        self.steps += 1
        if self.steps > 4000:
            self.dead = True

    def draw(self):
        s = self.screen
        s.fill((92, 148, 252))
        pygame.draw.rect(s, (60, 180, 80), (0, GROUND + 18, W, H - GROUND))
        pygame.draw.rect(s, (120, 90, 40), (0, GROUND + 14, W, 8))
        for p in self.pipes:
            sx = p - self.world
            if -30 < sx < W + 30:
                pygame.draw.rect(s, (20, 160, 40), (sx - 12, GROUND - 48, 24, 64))
                pygame.draw.rect(s, (10, 120, 30), (sx - 15, GROUND - 56, 30, 12))
        for g in self.gaps:
            sx = g - self.world
            if -40 < sx < W + 40:
                pygame.draw.rect(s, (92, 148, 252), (sx - 18, GROUND + 14, 60, 30))
        pygame.draw.rect(s, (230, 40, 30), (int(self.x), int(self.y) - 22, 20, 22))
        pygame.display.flip()

    def frame_small(self, w=16, h=8):
        import cv2
        big = pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
        small = cv2.resize(big, (w, h))
        gray = small.mean(axis=2) / 255.0
        ahead = gray[:, 6:].reshape(-1)
        return ahead.astype("float32")

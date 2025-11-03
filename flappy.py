import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

# constants
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
PIPE_WIDTH = 52
PIPE_GAP = 150
BASE_HEIGHT = 100
GRAVITY = 0.5
FLAP_STRENGTH = -8
BIRD_WIDTH = 34
BIRD_HEIGHT = 24

class FlappyBirdEnv(gym.Env):
    def __init__(self):
        super(FlappyBirdEnv, self).__init__()

        # Action space: 0 = do nothing, 1 = flap
        self.action_space = spaces.Discrete(2)

        # Observation space: [bird_y, bird_velocity, pipe_x, pipe_y]
        high = np.array([SCREEN_HEIGHT, 10, SCREEN_WIDTH, SCREEN_HEIGHT], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird RL")
        self.clock = pygame.time.Clock()

        # game variables
        self.reset_game_vars()

        # load bird image
        self.bird_img = pygame.image.load("bird.png").convert_alpha()
        self.bird_img = pygame.transform.scale(self.bird_img, (BIRD_WIDTH, BIRD_HEIGHT))

    def reset_game_vars(self):
        self.bird_y = SCREEN_HEIGHT // 2
        self.bird_vel = 0
        self.pipe_x = SCREEN_WIDTH
        self.pipe_y = random.randint(150, SCREEN_HEIGHT - 150)
        self.score = 0
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_game_vars()
        obs = np.array([self.bird_y, self.bird_vel, self.pipe_x, self.pipe_y], dtype=np.float32)
        return obs, {}

    def step(self, action):
        reward = 0.1
        self.bird_vel += GRAVITY
        if action == 1:
            self.bird_vel = FLAP_STRENGTH

        self.bird_y += self.bird_vel
        self.pipe_x -= 3

        # pipe passed
        if self.pipe_x + PIPE_WIDTH < 50:
            self.pipe_x = SCREEN_WIDTH
            self.pipe_y = random.randint(150, SCREEN_HEIGHT - 150)
            self.score += 1
            reward += 10

        # collision
        if (
            self.bird_y < 0
            or self.bird_y + BIRD_HEIGHT > SCREEN_HEIGHT - BASE_HEIGHT
            or (50 + BIRD_WIDTH > self.pipe_x and 50 < self.pipe_x + PIPE_WIDTH and
                (self.bird_y < self.pipe_y - PIPE_GAP // 2 or
                 self.bird_y + BIRD_HEIGHT > self.pipe_y + PIPE_GAP // 2))
        ):
            reward = -100
            self.done = True

        obs = np.array([self.bird_y, self.bird_vel, self.pipe_x, self.pipe_y], dtype=np.float32)
        return obs, reward, self.done, False, {}

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # background
        self.screen.fill((135, 206, 235))

        # pipes
        pygame.draw.rect(self.screen, (0, 255, 0),
                         (self.pipe_x, 0, PIPE_WIDTH, self.pipe_y - PIPE_GAP // 2))
        pygame.draw.rect(self.screen, (0, 255, 0),
                         (self.pipe_x, self.pipe_y + PIPE_GAP // 2, PIPE_WIDTH, SCREEN_HEIGHT))

        # bird
        self.screen.blit(self.bird_img, (50, self.bird_y))

        # ground
        pygame.draw.rect(self.screen, (222, 184, 135),
                         (0, SCREEN_HEIGHT - BASE_HEIGHT, SCREEN_WIDTH, BASE_HEIGHT))

        # show score
        font = pygame.font.SysFont("Arial", 30)
        score_text = font.render(f"Score: {self.score}", True, (255, 0, 0))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)

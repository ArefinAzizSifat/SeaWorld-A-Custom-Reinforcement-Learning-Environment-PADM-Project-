import sys
import pygame
import numpy as np
import gymnasium as gym


# Custom Environment:
# -------------------
class SeaWorldEnv(gym.Env):
    def __init__(self, grid_size=5, cell_size=100):
        super().__init__()

        self.grid_size = grid_size
        self.cell_size = cell_size
        self.goal = np.array([4, 4])  # Goal position (small fish)
        self.danger_states = []       # List of danger (obstacle) cells
        self.state = None
        self.reward = 0
        self.done = False
        self.info = {}

        # Define action & observation space
        self.action_space = gym.spaces.Discrete(4)  # up, down, right, left
        self.observation_space = gym.spaces.Box(
            low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.cell_size * self.grid_size, self.cell_size * self.grid_size)
        )
        pygame.display.set_caption("SeaWorld Game")

        # Load and scale images
        base_path = "/Users/aarefinasifat/Desktop/THI/SoSe 25/Principles of Autonomy and Decision Making/Assignment/Assisgnment1/Final/Assignment2/Pictures"
        self.background_img = pygame.transform.scale(
            pygame.image.load(f"{base_path}/background.png").convert(),
            (self.cell_size * self.grid_size, self.cell_size * self.grid_size)
        )
        self.bigfish_img = pygame.transform.scale(
            pygame.image.load(f"{base_path}/bigfish.png"),
            (self.cell_size, self.cell_size)
        )
        self.fish_img = pygame.transform.scale(
            pygame.image.load(f"{base_path}/smallfish.png"),
            (self.cell_size, self.cell_size)
        )
        self.obstacle_img = pygame.transform.scale(
            pygame.image.load(f"{base_path}/danger.png"),
            (self.cell_size, self.cell_size)
        )

    def add_danger(self, coord):
        self.danger_states.append(np.array(coord, dtype=int))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0, 0], dtype=int)
        self.done = False
        self.reward = 0
        self.info = {
            "Distance to fish": np.linalg.norm(self.state - self.goal)
        }
        return self.state.copy(), self.info

    def step(self, action):
        # Move agent
        if action == 0 and self.state[0] > 0:
            self.state[0] -= 1  # Up
        elif action == 1 and self.state[0] < self.grid_size - 1:
            self.state[0] += 1  # Down
        elif action == 2 and self.state[1] < self.grid_size - 1:
            self.state[1] += 1  # Right
        elif action == 3 and self.state[1] > 0:
            self.state[1] -= 1  # Left

        # Default step reward
        self.reward = -0.5
        self.done = False

        # Check goal or danger
        if np.array_equal(self.state, self.goal):
            self.reward = 30
            self.done = True
        elif any(np.array_equal(self.state, d) for d in self.danger_states):
            self.reward = -5
            self.done = True

        self.info = {
            "Distance to fish": np.linalg.norm(self.state - self.goal)
        }

        return self.state.copy(), self.reward, self.done, self.info

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.blit(self.background_img, (0, 0))

        # Draw grid lines
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y *
                                   self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        # Draw goal
        gx, gy = self.goal
        self.screen.blit(
            self.fish_img, (gy * self.cell_size, gx * self.cell_size))

        # Draw dangers
        for dx, dy in self.danger_states:
            self.screen.blit(self.obstacle_img,
                             (dy * self.cell_size, dx * self.cell_size))

        # Draw agent
        ax, ay = self.state
        self.screen.blit(self.bigfish_img,
                         (ay * self.cell_size, ax * self.cell_size))

        pygame.display.flip()
        pygame.time.wait(250)

    def close(self):
        pygame.quit()
        super().close()


# create environment with danger states
def create_env():
    env = SeaWorldEnv(grid_size=5)
    env.add_danger((0, 3))
    env.add_danger((2, 1))
    env.add_danger((2, 3))
    return env

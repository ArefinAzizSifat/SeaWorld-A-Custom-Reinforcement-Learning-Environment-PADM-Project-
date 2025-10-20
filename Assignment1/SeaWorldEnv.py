import sys
import pygame
import numpy as np
import gymnasium as gym


# Custom Environment:
class SeaWorldEnv(gym.Env):
    def __init__(self, grid_size) -> None:
        super().__init__()

        self.state = None
        self.done = False
        self.info = {}
        self.reward = 0
        self.cell_size = 100
        self.grid_size = grid_size
        self.goal = np.array([4, 4])  # Fish location
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=4, shape=(2,), dtype=np.int32)

        self.danger_states = []

        # Display:
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.cell_size*self.grid_size, self.cell_size*self.grid_size))
        pygame.display.set_caption("SeaWorld Game")

        # Load images:
        self.background_img = pygame.image.load(
            "/Users/aarefinasifat/Desktop/THI/SoSe 25/Principles of Autonomy and Decision Making/Assignment/Assisgnment1/Final/Assignment1/Pictures/background.png").convert()
        self.background_img = pygame.transform.scale(
            self.background_img, (self.cell_size*self.grid_size, self.cell_size*self.grid_size))
        self.bigfish_img = pygame.image.load(
            "/Users/aarefinasifat/Desktop/THI/SoSe 25/Principles of Autonomy and Decision Making/Assignment/Assisgnment1/Final/Assignment1/Pictures/bigfish.png")
        self.bigfish_img = pygame.transform.scale(
            self.bigfish_img, (self.cell_size, self.cell_size))
        self.fish_img = pygame.image.load(
            "/Users/aarefinasifat/Desktop/THI/SoSe 25/Principles of Autonomy and Decision Making/Assignment/Assisgnment1/Final/Assignment1/Pictures/smallfish.png")
        self.fish_img = pygame.transform.scale(
            self.fish_img, (self.cell_size, self.cell_size))
        self.obstacle_img = pygame.image.load(
            "/Users/aarefinasifat/Desktop/THI/SoSe 25/Principles of Autonomy and Decision Making/Assignment/Assisgnment1/Final/Assignment1/Pictures/danger.png")
        self.obstacle_img = pygame.transform.scale(
            self.obstacle_img, (self.cell_size, self.cell_size))

    def add_danger(self, coordinates):
        self.danger_states.append(np.array(coordinates))

    def reset(self):
        self.state = np.array([0, 0])
        self.done = False
        self.reward = 0

        self.info["Distance to fish"] = np.sqrt(
            (self.state[0]-self.goal[0])**2 + (self.state[1]-self.goal[1])**2
        )

        return self.state, self.info

    def step(self, action):
        # Move
        if action == 0 and self.state[0] > 0:              # Up
            self.state[0] -= 1
        elif action == 1 and self.state[0] < self.grid_size-1:  # Down
            self.state[0] += 1
        elif action == 2 and self.state[1] < self.grid_size-1:  # Right
            self.state[1] += 1
        elif action == 3 and self.state[1] > 0:             # Left
            self.state[1] -= 1

        # Default step reward
        self.reward = -0.5
        self.done = False

        # Check termination
        if np.array_equal(self.state, self.goal):
            self.done = True
            self.reward = 30
        elif any(np.array_equal(self.state, d) for d in self.danger_states):
            self.done = True
            self.reward = -5

        self.info["Distance to fish"] = np.sqrt(
            (self.state[0] - self.goal[0]) ** 2 +
            (self.state[1] - self.goal[1]) ** 2
        )

        return self.state, self.done, self.reward, self.info

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.blit(self.background_img, (0, 0))

        # Draw grid
        for col in range(self.grid_size):
            for row in range(self.grid_size):
                rect = pygame.Rect(col*self.cell_size, row*self.cell_size,
                                   self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        # Draw fish (goal)
        self.screen.blit(
            self.fish_img, (self.goal[1]*self.cell_size, self.goal[0]*self.cell_size))

        # Draw dangers
        for danger in self.danger_states:
            self.screen.blit(
                self.obstacle_img, (danger[1]*self.cell_size, danger[0]*self.cell_size))

        # Draw bigfish (agent)
        self.screen.blit(
            self.bigfish_img, (self.state[1]*self.cell_size, self.state[0]*self.cell_size))

        pygame.display.flip()
        pygame.time.wait(250)

    def close(self):
        pygame.quit()


# Run the environment:
if __name__ == "__main__":
    env = SeaWorldEnv(grid_size=5)

    # Add dangers
    env.add_danger((0, 3))
    env.add_danger((2, 1))
    env.add_danger((2, 3))

    for _ in range(50):
        state, info = env.reset()
        print("Initial State:", state, "Distance to Fish:",
              float(info["Distance to fish"]))

        for _ in range(20):
            action = env.action_space.sample()
            state, done, reward, info = env.step(action)
            env.render()
            print("State:", state, "Done:", done, "Reward:", reward,
                  "Distance to Fish:", float(info["Distance to fish"]))

            if done:
                break

    env.close()

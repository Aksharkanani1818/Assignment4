import sys
import pygame
import numpy as np
import gymnasium as gym

# Step 0: Import images for the environment
# -------
# Paths to your images
agent_img_path = r'D:\Ingolstadt\PADM\q_learning_akk5551 (1)\q_learning_akk5551\agent.gif'
goal_img_path = r'D:\Ingolstadt\PADM\q_learning_akk5551 (1)\q_learning_akk5551\goal.webp'
obstacle_img_path = r'D:\Ingolstadt\PADM\q_learning_akk5551 (1)\q_learning_akk5551\obstacles.gif'
cctv_img_path = r'D:\Ingolstadt\PADM\q_learning_akk5551 (1)\q_learning_akk5551\cctv.gif'
coin_gif = r"D:\Ingolstadt\PADM\q_learning_akk5551 (1)\q_learning_akk5551\coin.gif"


# Step 1: Define your own custom environment
# -------
class ChidEnv(gym.Env):
    def __init__(self, grid_size=6, goal_coordinates=(5, 5)) -> None:
        super(ChidEnv, self).__init__()
        self.grid_size = grid_size
        self.cell_size = 100
        self.state = np.array([0, 0])
        self.reward = 0
        self.coin_01 = False
        self.coin_02 = False
        self.coin_03 = False
        self.in_cctv = False
        self.cctv_transfer = []
        self.coin = []
        self.info = {}
        self.goal = np.array(goal_coordinates)
        self.cctv = []
        self.done = False
        self.hell_states = []

        # Load images
        self.agent_image = pygame.transform.scale(pygame.image.load(agent_img_path), (self.cell_size, self.cell_size))
        self.goal_image = pygame.transform.scale(pygame.image.load(goal_img_path), (self.cell_size, self.cell_size))
        self.hell_image = pygame.transform.scale(pygame.image.load(obstacle_img_path), (self.cell_size, self.cell_size))
        self.cctv_image = pygame.transform.scale(pygame.image.load(cctv_img_path), (self.cell_size, self.cell_size))
        self.coin_image = pygame.transform.scale(pygame.image.load(coin_gif),(self.cell_size,self.cell_size))

        # Action-space:
        self.action_space = gym.spaces.Discrete(4)
        
        # Observation space:
        self.observation_space = gym.spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32)

        # Initialize the Pygame window
        pygame.init()
        self.screen = pygame.display.set_mode((self.cell_size * grid_size, self.cell_size * grid_size))

    def reset(self):
        self.state = np.array([0, 0])
        self.in_cctv = False
        self.reward = 0
        self.done = False
        self.coin_01 = False
        self.coin_02 = False
        self.coin_03 = False
        self.info = {'Distance to goal': np.linalg.norm(self.state - self.goal)}
        return self.state, self.info

    def add_hell_states(self, hell_state_coordinates):
        self.hell_states.append(np.array(hell_state_coordinates))

    def add_cctv_states(self, cctv_coordinates):
        self.cctv.append(np.array(cctv_coordinates))
    def add_coin(self, coin):
        self.coin.append(np.array(coin))

    def step(self, action):
        if action == 0 and self.state[0] > 0:  # Up
            self.state[0] -= 1
        if action == 1 and self.state[0] < self.grid_size - 1:  # Down
            self.state[0] += 1
        if action == 2 and self.state[1] < self.grid_size - 1:  # Right
            self.state[1] += 1
        if action == 3 and self.state[1] > 0:  # Left
            self.state[1] -= 1
        
        # Update reward, check for goal state or hell state
        if np.array_equal(self.state, self.goal):
            self.reward = 30
            self.done = True
        elif any(np.array_equal(self.state, hell) for hell in self.hell_states):
            self.reward = -5
            self.done = True
        elif any(np.array_equal(self.state, coin) for coin in self.coin):

            if not self.coin_01 and np.array_equal(self.state, self.coin[0]):
                self.coin_01 = True
                self.reward = +3
            elif self.coin_01 and np.array_equal(self.state, self.coin[0]):
                self.reward += 0

            if not self.coin_02 and np.array_equal(self.state, self.coin[1]):
                self.coin_02 = True
                self.reward = +3
            elif self.coin_02 and np.array_equal(self.state, self.coin[1]):
                self.reward += 0

            if not self.coin_03 and np.array_equal(self.state, self.coin[2]):
                self.coin_03 = True
                self.reward = +3
            elif self.coin_03 and np.array_equal(self.state, self.coin[2]):
                self.reward += 0


        elif any(np.array_equal(self.state, cctv) for cctv in self.cctv):
            #print('im going')
            self.in_cctv = False
            
            if np.array_equal(self.state, self.cctv[0]):
                self.in_cctv = True
                self.cctv_transfer = self.cctv[0]
            elif np.array_equal(self.state, self.cctv[1]):
                self.in_cctv = True
                self.cctv_transfer = self.cctv[1]
            
            self.reward -= 2
            self.state[0] = 0
            self.state[1] = 0
        else:
            self.reward = -1  # Negative reward for each step

        self.info['Distance to goal'] = np.linalg.norm(self.state - self.goal)
        return self.state, self.reward, self.done, self.info, self.cctv_transfer, self.in_cctv

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((255, 255, 255))  # Fill the background with white

        # Draw the grid lines
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        # Draw the hell-states, goal, and agent using their images
        for hell in self.hell_states:
            self.screen.blit(self.hell_image, (hell[1] * self.cell_size, hell[0] * self.cell_size))
        self.screen.blit(self.goal_image, (self.goal[1] * self.cell_size, self.goal[0] * self.cell_size))
        self.screen.blit(self.agent_image, (self.state[1] * self.cell_size, self.state[0] * self.cell_size))

        for c in self.coin:
            self.screen.blit(self.coin_image, (c[1] * self.cell_size, c[0] * self.cell_size))


        for cctv in self.cctv:
            self.screen.blit(self.cctv_image, (cctv[1] * self.cell_size, cctv[0] * self.cell_size))


        pygame.display.flip()  # Update the full display

    def close(self):
        pygame.quit()

def create_env(goal_coordinates, hell_state_coordinates, cctv_coordinates, coin):
  env = ChidEnv(goal_coordinates=goal_coordinates)

  for i in range(len(hell_state_coordinates)):
    env.add_hell_states(hell_state_coordinates=hell_state_coordinates[i])

  for i in range(len(cctv_coordinates)):
    env.add_cctv_states(cctv_coordinates=cctv_coordinates[i])
    for i in range(len(coin)):
        env.add_coin(coin=coin[i])

  return env
# Step 2: Instantiate and use your custom environment
# -------
# my_env = ChidEnv()
# hell_states = [(1, 1), (2, 3), (3, 1), (4, 4), (0, 5)]  # Add more if needed
# for hs in hell_states:
#     my_env.add_hell_states(hell_state_coordinates=hs)

# # Manual control of the environment
# observation, info = my_env.reset()
# print(f"Initial state: {observation}, Info: {info}")

# try:
#     while True:
#         my_env.render()
#         keys = pygame.key.get_pressed()
#         action = None
#         if keys[pygame.K_UP]:
#             action = 0
#         elif keys[pygame.K_DOWN]:
#             action = 1
#         elif keys[pygame.K_RIGHT]:
#             action = 2
#         elif keys[pygame.K_LEFT]:
#             action = 3
        
#         if action is not None:
#             observation, reward, done, info = my_env.step(action)
#             print(f"Observation: {observation}, Reward: {reward}, Done: {done}, Info: {info}")
        
#         if 'done' in locals() and done:
#             print("Game Over!")
#             break

#         # Limit the loop to 10 times per second
#         pygame.time.Clock().tick(10)

# except Exception as e:
#     print(e)
# finally:
#     my_env.close()

import registration
import gymnasium as gym
import sys
import numpy as np
import cv2
from collections import deque
import time

main_env = gym.make("Heal-v1", render_mode="human")
main_env.reset()

env = main_env.env.env.env
initial_position = None
model = env._model
data = env._data
action_trajectory = np.load("../training_script/act_trajectory 3.npy")

print(f"action trajectory size {action_trajectory.shape}")


for i in range(100):
	action = [-1.5730853391174673, 0.15484474915392638, -0.051969636236800565, -0.24907753891966092, 0.2833406040512775, -6.95920944727502e-07, 1.4214986626857298]
	obs, reward , terminated, truncated, info = env.step(action)
	# env.render()
	frame = env.render()
	# frame = np.hstack([frame['wrist_cam'], frame['bev_cam']])
	# cv2.imshow(f'Frame', frame)
	# cv2.waitKey(1)
      
class MovingAverageFilter:
    def __init__(self, window=5):
        self.window = window
        self.buffer = deque(maxlen=window)

    def filter(self, action):
        self.buffer.append(action)
        # Compute moving average: axis=0 to average over timesteps
        return np.mean(self.buffer, axis=0)
    
action_filter = MovingAverageFilter(window=10)


for i in range(2000):

    obs, reward , terminated, truncated, info = env.step(action)
    time.sleep(0.005)

for action in action_trajectory:
	
    action[-1] = 255*action[-1]

    action = action_filter.filter(action)
	
    obs, reward , terminated, truncated, info = env.step(action)
	
    state = obs[:6].tolist()
	
    state.append(obs[-2])
	
    print(state)
    input("test")

    frame = env.render()
    time.sleep(0.001)



    
	
    # frame = np.hstack([frame['wrist_cam'], frame['bev_cam'], frame['front']])
	
    # cv2.imshow(f'Frame', frame)
	
    # cv2.waitKey(1)
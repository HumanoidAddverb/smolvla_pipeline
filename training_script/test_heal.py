# '''
# # Author : Rajesh Kumar (rajesh.kumar01@addverb.com)
# # Details : This is the OpenAI SAC example for the heal robot 
# '''

# import registration
# import gymnasium as gym
# import cv2

# from env.heal.ik_solver import IKSolver

# # from agents.sac.sac import SAC
# # from agents.train import Train
# # from configs import Configs as cfgs

# import numpy as np
# # from utils.env_utils import make_env_and_datasets
# import time

# # env = gym.make("Heal-v1", render_mode="rgb_array", max_episode_steps=200)
# # env = gym.make('Heal-v1', render_mode="rgb_array")
# env = gym.make("Heal-v1", render_mode="human")
# print(env.spec.max_episode_steps)
# env.reset()
# n_steps = 0
# ik_solver = IKSolver(env)
# while True:
#     action = np.random.uniform(-0.1, 0.1, 6) * 0
#     action[0] = 0.1
#     # action = np.ones(6)
#     obs, truncated = env.step(action)
#     # print(f'Truncation flag: {truncated}')
#     c_pose = ik_solver.get_current_position()
#     print(f'Current_pose: {c_pose}')
#     # del_pos = np.array([00, 0.0, 0.05])
#     # ik_solution = env.compute_ik(del_pos)
#     # print(ik_solution)
#     env.render()
#     # obs, ik_solution = env.render()
#     # print(f'IK solution: {ik_solution}')
#     # cv2.imshow('obs', cv2.flip(obs, 0))
#     # cv2.waitKey(1)
#     time.sleep(0.001)





'''
# Author : Rajesh Kumar (rajesh.kumar01@addverb.com)
# Details : This is the OpenAI SAC example for the heal robot 
'''

import registration
import gymnasium as gym
import sys

from trajectory_follow import IKSolver

sys.path.append("/home/gonnayaswanth/heal_gym_openai")
env = gym.make("Heal-v1", render_mode="human")
print(env.spec.max_episode_steps)
env.reset()
n_steps = 0
ik_solver = IKSolver(env)
compute_once = True
while True:
    ik_solver.follow_trajectory()
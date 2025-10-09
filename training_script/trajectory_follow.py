"""
	Date: April 01, 2025
	Description: Script to follow a predetermined trajectory
	specified in Cartesian Coordinates for the Heal Robots
	Author: Anant Garg (anant.garg@addverb.com)
"""

import mujoco
import ikpy.chain
import numpy as np
import cv2

class IKSolver:
	def __init__(self, env) -> None:
		self.env = env.env.env.env
		self.initial_position = None
		self.model = self.env._model
		self.data = self.env._data
		
	def follow_trajectory(self):
		for i in range(10000):

			action = np.zeros(7)
			action[0] = -1.57
			# action[0] = -1.0
			obs, reward , terminated, truncated, info = self.env.step(action)
			# self.env.render()
			frame = self.env.render()
			# frame = np.hstack([frame['wrist_cam'], frame['bev_cam']])
			# cv2.imshow(f'Frame', frame)
			# cv2.waitKey(1)

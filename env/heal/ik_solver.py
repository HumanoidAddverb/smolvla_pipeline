"""
	Date: March 27, 2025
	Description: Script to compute IK Solution for Heal Robot
	Author: Anant Garg (anant.garg@addverb.com)
"""

import mujoco
import ikpy.chain
import numpy as np

class IKSolver:
	def __init__(self, env) -> None:
		self.env = env.env.env.env
		self.initial_position = None
		self.model = self.env._model
		self.data = self.env._data
		urdf_file_path = f'../env/descriptions/heal_robot/urdf/heal_anant_single_arm.urdf'
		self.chain = ikpy.chain.Chain.from_urdf_file(urdf_file_path)

	def get_current_position(self):
		link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "heal/end_effector")
		current_position = self.data.xpos[link_id]
		return current_position

	def compute_ik(self, target_position):
		# Computes inverse kinematics given a target cartesian position (tx, ty, tz)
		if self.initial_position is None:
			link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "heal/end_effector")
			self.initial_position = self.data.xpos[link_id]
			# target_position = self.initial_position
		try:
			# self.initial_position = self.data.xpos[link_id]
			# print(f'Initial Pose: {self.initial_position} | Target Pose: {target_position}')
			initial_position = self.initial_position
			# initial_position = np.array([0.0, 0.0, 1.0])
			# ik_solution = self.chain.inverse_kinematics(target_position-initial_position, initial_position)
			# target_position = np.array([-0.001, -0.045411697, 0.88470996])
			# target_position = np.array([-0.001, -0.045411697, 1.08470996])
			target_orientation=np.array([0, 0, 1])
			ik_solution = self.chain.inverse_kinematics(target_position=target_position, target_orientation=target_orientation, orientation_mode='Z')

			fk_solution = self.chain.forward_kinematics(ik_solution)
			return ik_solution, fk_solution, True
		except Exception as e:
			print(f'IK computation failed with following error: {str(e)}')
			return np.zeros(6), np.zeros(6), False
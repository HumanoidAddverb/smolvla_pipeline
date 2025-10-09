import mujoco
import numpy as np
from dm_control import mjcf
from gymnasium.spaces import Box

from env.heal.manispace_env import ManiSpaceEnv
import ikpy.chain
import lie
import sys


class CubeEnv(ManiSpaceEnv):
	def __init__(self, *args, **kwargs):
		self.observation_space = Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float64)
		urdf_file_path = f'../env/descriptions/heal_robot/urdf/heal_anant_single_arm.urdf'
		self.chain = ikpy.chain.Chain.from_urdf_file(urdf_file_path)
		super().__init__(*args, **kwargs)

		self._num_cubes = 0

		self.last_x_pose = np.array([0,0,0])
		self._ob_type = 'pixel'

	def render(self):
		return super().render()
	
	def compute_observation(self):
		return self.get_pixel_observation()
	
	def add_objects(self, xml_path=None, arena_mjcf=None):
		assert xml_path is not None, "XML path of the object cannot be None"
		assert arena_mjcf is not None, "Arena MJCF cannot be None"
		object_mjcf = mjcf.from_path((self._desc_dir / 'objects' / f'{xml_path}').as_posix())
		arena_mjcf.include_copy(object_mjcf)

	def initialize_episode(self):
		self._data.qpos[self._arm_joint_ids] = self._home_qpos
		mujoco.mj_kinematics(self._model, self._data)

		self.initialize_arm()
		for i in range(1):
			# xy = self.np_random.uniform(low=[0.2, -0.3], high=[0.25, -0.3])
			xy = (0.2, -0.35)
			obj_pos = (*xy, 0.22)
			self._data.joint(f'cube_joint_{i}').qpos[:3] = obj_pos

		for i in range(self._num_cubes):
			xy = self.np_random.uniform(*self._object_sampling_bounds)
			obj_pos = (*xy, 0.22)
			self._data.joint(f'object_joint_{i}').qpos[:3] = obj_pos
		# Set a new target.
		self.pre_step()


	def initialize_arm(self):
		# Sample initial effector position and orientation.
		# eff_pos = self.np_random.uniform(*self._arm_sampling_bounds)
		eff_pos = np.array([0.4, 0, 0.5])
		self.last_x_pose = eff_pos
		target_orientation=np.array([0, 0, 1])
		ik_solution = self.chain.inverse_kinematics(target_position=eff_pos, target_orientation=target_orientation, orientation_mode='Z')[1:]
		ik_solution_ = np.zeros(7)
		ik_solution_[:6] = ik_solution
		ik_solution_[-1] = 0
		self._data.qpos[:ik_solution_.shape[0]] = ik_solution_
		mujoco.mj_forward(self._model, self._data)


	def add_object_info(self, ob_info):
		# Cube positions and orientations.
		for i in range(self._num_cubes):
			ob_info[f'privileged/block_{i}_pos'] = self._data.joint(f'object_joint_{i}').qpos[:3].copy()
			ob_info[f'privileged/block_{i}_quat'] = self._data.joint(f'object_joint_{i}').qpos[3:].copy()
			ob_info[f'privileged/block_{i}_yaw'] = np.array(
				[lie.SO3(wxyz=self._data.joint(f'object_joint_{i}').qpos[3:]).compute_yaw_radians()]
			)


	def compute_observation(self):
		if self._ob_type == 'pixels':
			return self.get_pixel_observation()
		else:
			xyz_center = np.array([0.425, 0.0, 0.0])
			xyz_scaler = 10.0
			gripper_scaler = 3.0

			ob_info = self.compute_ob_info()
			ob = [
				ob_info['proprio/joint_pos'],
				ob_info['proprio/joint_vel'],
				(ob_info['proprio/effector_pos'] - xyz_center) * xyz_scaler,
				np.cos(ob_info['proprio/effector_yaw']),
				np.sin(ob_info['proprio/effector_yaw']),
				ob_info['proprio/gripper_opening'] * gripper_scaler,
				ob_info['proprio/gripper_contact'],
			]
			# for i in range(self._num_cubes):
			# 	ob.extend(
			# 		[
			# 			(ob_info[f'privileged/block_{i}_pos'] - xyz_center) * xyz_scaler,
			# 			ob_info[f'privileged/block_{i}_quat'],
			# 			np.cos(ob_info[f'privileged/block_{i}_yaw']),
			# 			np.sin(ob_info[f'privileged/block_{i}_yaw']),
			# 		]
			# 	)

			return np.concatenate(ob)


	def _compute_successes(self):
		"""Compute object successes."""
		cube_successes = []
		for i in range(self._num_cubes):
			obj_pos = self._data.joint(f'object_joint_{i}').qpos[:3]
			tar_pos = self._data.mocap_pos[self._cube_target_mocap_ids[i]]
			if np.linalg.norm(obj_pos - tar_pos) <= 0.04:
				cube_successes.append(True)
			else:
				cube_successes.append(False)
		return cube_successes

	def compute_reward(self, ob, action):
		successes = self._compute_successes()
		reward = float(sum(successes) - len(successes))
		return reward

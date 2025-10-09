import numpy as np
from gymnasium.spaces import Box
from pathlib import Path
from dm_control import mjcf
import mujoco

from env.heal.env import CustomMuJoCoEnv
from env import mjcf_utils

import lie


class ManiSpaceEnv(CustomMuJoCoEnv):

	def __init__(
	self,
		ob_type='states',
		physics_timestep=0.002,
		control_timestep=0.05,
		terminate_at_goal=True,
		mode='task',
		visualize_info=True,
		pixel_transparent_arm=True,
		reward_task_id=None,
		use_oracle_rep=False,
		**kwargs,
	):
		super().__init__(
			physics_timestep=physics_timestep,
			control_timestep=control_timestep,
			**kwargs,
		)
		self._desc_dir = Path(__file__).resolve().parent.parent / 'descriptions'
		action_range = np.array([0.05, 0.05, 0.05, 0.3, 1.0])
		self.action_low = -action_range
		self.action_high = action_range

		self._home_qpos = np.asarray([-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
		self._arm_sampling_bounds = np.asarray([[0.25, -0.35, 0.20], [0.6, 0.35, 0.35]])
		self._object_sampling_bounds = np.asarray([[0.3, -0.4], [0.4, -0.3]])

		self._colors = dict(
            red=np.array([0.96, 0.26, 0.33, 1.0]),
            orange=np.array([1.0, 0.69, 0.21, 1.0]),
            green=np.array([0.06, 0.74, 0.21, 1.0]),
            blue=np.array([0.35, 0.55, 0.91, 1.0]),
            purple=np.array([0.61, 0.28, 0.82, 1.0]),
            lightred=np.array([0.99, 0.85, 0.86, 1.0]),
            lightorange=np.array([1.0, 0.94, 0.84, 1.0]),
            lightgreen=np.array([0.77, 0.95, 0.81, 1.0]),
            lightblue=np.array([0.86, 0.9, 0.98, 1.0]),
            lightpurple=np.array([0.91, 0.84, 0.96, 1.0]),
            white=np.array([0.9, 0.9, 0.9, 1.0]),
            lightgray=np.array([0.7, 0.7, 0.7, 1.0]),
            gray=np.array([0.5, 0.5, 0.5, 1.0]),
            darkgray=np.array([0.3, 0.3, 0.3, 1.0]),
            black=np.array([0.1, 0.1, 0.1, 1.0]),
        )

		self._terminate_at_goal = terminate_at_goal

		self._success = False

		# self.observation_space = Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)
		# self.action_space = Box(low=-np.ones(6), high=np.ones(6), shape=(6,), dtype=np.float32)

		

	def normalize_action(self, action):
		"""Normalize the action to the range [-1, 1]."""
		action = 2 * (action - self.action_low) / (self.action_high - self.action_low) - 1
		return np.clip(action, -1, 1)
	
	def unnormalize_action(self, action):
		return 0.5 * (action + 1) * (self.action_high - self.action_low) + self.action_low
	
	
	def compute_ik(self, target_position, target_orientation=np.array([0,0,1])):
		ik_solution = self.chain.inverse_kinematics(target_position=target_position, target_orientation=target_orientation, orientation_mode='Z')[1:]
		return ik_solution
		
		
	def build_mjcf_model(self):
		arena_mjcf = mjcf.from_path((self._desc_dir / 'floor_wall.xml').as_posix())
		arena_mjcf.model = 'heal_arena'

		arena_mjcf.statistic.center = (0.3, 0, 0.15)
		arena_mjcf.statistic.extent = 0.7
		getattr(arena_mjcf.visual, 'global').elevation = -20
		getattr(arena_mjcf.visual, 'global').azimuth = 180
		arena_mjcf.statistic.meansize = 0.04
		arena_mjcf.visual.map.znear = 0.1
		arena_mjcf.visual.map.zfar = 10.0

		# Add the Heal Robot
		heal_mjcf = mjcf.from_path((self._desc_dir / 'heal_robot' / 'mjmodel.xml'), escape_separators=True)
		# # Attach the gripper with the manipulator
		gripper_mjcf = mjcf.from_path((self._desc_dir / 'robotiq_2f85' / '2f85.xml'), escape_separators=True)
		gripper_mjcf.model = 'robotiq'
		mjcf_utils.attach(heal_mjcf, gripper_mjcf, 'attachment_site')

		# Attach Heal to the scene
		mjcf_utils.attach(arena_mjcf, heal_mjcf)

		# TO DO: Add bounding Boxes for debugging and understanding the workspace and object sampling areas during simulation.

		xml_path = f'glass.xml'
		self.add_objects(xml_path, arena_mjcf)

		self._arm_jnts = mjcf_utils.safe_find_all(
            heal_mjcf,
            'joint',
            exclude_attachments=True,
        )

		self._arm_acts = mjcf_utils.safe_find_all(
            heal_mjcf,
            'actuator',
            exclude_attachments=True,
        )

		self._gripper_jnts = mjcf_utils.safe_find_all(gripper_mjcf, 'joint', exclude_attachments=True)
		self._gripper_acts = mjcf_utils.safe_find_all(gripper_mjcf, 'actuator', exclude_attachments=True)

		cameras = {
			'front': {
				'pos': (1.287, -0.300, 0.509),
				'xyaxes': (0.000, 1.000, 0.000, -0.342, 0.000, 0.940),
			},
			'front_pixels': {
				'pos': (1.553, 0.0, 1.139),
				'xyaxes': (0.000, 1.000, 0.000, -0.628, 0.001, 0.778),
			},
			'top_bev': {
				'pos': (0.0, -0.2, 1.2),
				'xyaxes': (0.00, -1.000, 0.00, 1.00, 0.00, 0.00),
			},
		}
		for camera_name, camera_kwargs in cameras.items():
			arena_mjcf.worldbody.add('camera', name=camera_name, **camera_kwargs)
		return arena_mjcf
	
	def post_compilation(self):
		# Arm joint and actuator IDs.
		arm_joint_names = [j.full_identifier for j in self._arm_jnts]
		self._arm_joint_ids = np.asarray([self._model.joint(name).id for name in arm_joint_names])
		actuator_names = [a.full_identifier for a in self._arm_acts]
		self._arm_actuator_ids = np.asarray([self._model.actuator(name).id for name in actuator_names])
		gripper_actuator_names = [a.full_identifier for a in self._gripper_acts]
		self._gripper_actuator_ids = np.asarray([self._model.actuator(name).id for name in gripper_actuator_names])
		self._gripper_opening_joint_id = self._model.joint('heal/robotiq/right_driver_joint').id

		# # Site IDs.
		self._pinch_site_id = self._model.site('heal/robotiq/pinch').id
		self._attach_site_id = self._model.site('heal/attachment_site').id


	def post_step(self):
		# Check if the cubes are in the target positions.
		cube_successes = self._compute_successes()
		if self._mode == 'data_collection':
			self._success = cube_successes[self._target_block]
		else:
			self._success = all(cube_successes)

	def compute_ob_info(self):
		ob_info = {}

		# Proprioceptive observations
		ob_info['proprio/joint_pos'] = self._data.qpos[self._arm_joint_ids].copy()
		ob_info['proprio/joint_vel'] = self._data.qvel[self._arm_joint_ids].copy()
		ob_info['proprio/effector_pos'] = self._data.site_xpos[self._pinch_site_id].copy()
		ob_info['proprio/effector_yaw'] = np.array(
			[lie.SO3.from_matrix(self._data.site_xmat[self._pinch_site_id].copy().reshape(3, 3)).compute_yaw_radians()]
		)
		ob_info['proprio/gripper_opening'] = np.array(
			np.clip([self._data.qpos[self._gripper_opening_joint_id] / 0.8], 0, 1)
		)
		ob_info['proprio/gripper_vel'] = self._data.qvel[[self._gripper_opening_joint_id]].copy()
		ob_info['proprio/gripper_contact'] = np.array(
			[np.clip(np.linalg.norm(self._data.body('heal/robotiq/right_pad').cfrc_ext) / 50, 0, 1)]
		)

		self.add_object_info(ob_info)

		ob_info['prev_qpos'] = self._prev_qpos.copy()
		ob_info['prev_qvel'] = self._prev_qvel.copy()
		ob_info['qpos'] = self._data.qpos.copy()
		ob_info['qvel'] = self._data.qvel.copy()
		ob_info['control'] = self._data.ctrl.copy()
		ob_info['time'] = np.array([self._data.time])

		return ob_info

	def get_step_info(self):
		ob_info = self.compute_ob_info()
		ob_info['success'] = self._success
		return ob_info
	
	def terminate_episode(self):
		if self._terminate_at_goal:
			return self._success
		else:
			return False
	
	def add_object_info(self, ob_info):
		# Cube positions and orientations.
		for i in range(self._num_cubes):
			ob_info[f'privileged/block_{i}_pos'] = self._data.joint(f'object_joint_{i}').qpos[:3].copy()
			ob_info[f'privileged/block_{i}_quat'] = self._data.joint(f'object_joint_{i}').qpos[3:].copy()
			ob_info[f'privileged/block_{i}_yaw'] = np.array(
				[lie.SO3(wxyz=self._data.joint(f'object_joint_{i}').qpos[3:]).compute_yaw_radians()]
			)

		if self._mode == 'data_collection':
			# Target cube info.
			ob_info['privileged/target_task'] = self._target_task

			target_mocap_id = self._cube_target_mocap_ids[self._target_block]
			ob_info['privileged/target_block'] = self._target_block
			ob_info['privileged/target_block_pos'] = self._data.mocap_pos[target_mocap_id].copy()
			ob_info['privileged/target_block_yaw'] = np.array(
				[lie.SO3(wxyz=self._data.mocap_quat[target_mocap_id]).compute_yaw_radians()]
			)
	
	def compute_observation(self):
		return self.get_pixel_observation()
	
	def get_pixel_observation(self):
		frame = self.render()
		return frame
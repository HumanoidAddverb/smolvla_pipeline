import numpy as np
import abc
from typing import Any, Callable, Optional, SupportsFloat
import cv2

import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from dm_control import mjcf

from env import mjcf_utils

import ikpy.chain

import pyroki as pk
from pyroki_snippets import solve_ik
from yourdfpy import URDF


class CustomMuJoCoEnv(gym.Env, abc.ABC):
	metadata = {
		"render_modes": [
			"human",
			"rgb_array",
			"depth_array",
		],
		"render_fps": 100,
	}

	def __init__(
		self,
		physics_timestep: float = 0.002,
		control_timestep: float = 0.02,
		render_mode: Optional[str] = None,
		width: int = 640,
		height: int = 480,
		**kwargs,
	):
		self.max_episode_steps = 200
		self.current_step = 0
		self.global_step = 0
		self._mjcf_model: Optional[mjcf.RootElement] = None
		self._model: Optional[mujoco.MjModel] = None
		self._data: Optional[mujoco.MjData] = None

		self._passive_viewer_handle = None
		self._render_height = height
		self._render_width = width
		self._renderer: Optional[mujoco.Renderer] = None
		self._scene_option = mujoco.MjvOption()
		self._camera = mujoco.MjvCamera()

		self.render_mode = render_mode
		self.viewer = None
		self.set_timesteps(
			physics_timestep=float(physics_timestep),
			control_timestep=float(control_timestep),
		)

		urdf_file_path = f'/heal_robot/urdf/heal_anant_single_arm.urdf'
		self.urdf = URDF.load(urdf_file_path)
		self.robot = pk.Robot.from_urdf(self.urdf)

		

	def step(self, action):
		# Added on April 5, 2025. Calculate the joint angles and gripper position based on action
		# provided in terms of (del x, del y, del z, yaw, gripper_action)
		# action = self.compute_action(action)
		self._data.ctrl[:7] = action 
		mujoco.mj_step(self._model, self._data, nstep=self._n_steps)

		self.current_step += 1
		self.global_step += 1

		terminated = self.terminate_episode() # IMPLEMENT THE FUNCTION
		truncated = self.truncate_episode() # IMPLEMENT THE FUNCTION
		ob = self.compute_observation() # IMPLEMENT THE FUNCTION
		reward = self.compute_reward(ob, action) # IMPLEMENT THE FUNCTION
		info = self.get_step_info() # IMPLEMENT THE FUNCTION

		return ob, reward, terminated, truncated, info
	
	def reset(self, seed: int = None, options = None, **kwargs):
		super().reset(seed=seed, options=options, **kwargs)
		if self._mjcf_model is None:
			self._mjcf_model = self.build_mjcf_model()
		self.compile_model_and_data()

		self.initialize_episode()
		mujoco.mj_forward(self._model, self._data)

		ob = self.compute_observation()
		return ob

	def pre_step(self):
		self._prev_qpos = self._data.qpos.copy()
		self._prev_qvel = self._data.qvel.copy()
		self._prev_ob_info = self.compute_ob_info()

	def compute_action(self, action):
		self.last_x_pose += action[:3]

		target_x_pose = self.last_x_pose
		target_wxyz = np.array([1,0,0,0])
		ik_solution = solve_ik(
			robot=self.robot,
			target_link_name="end_effector",
			target_position=target_x_pose,
			target_wxyz=target_wxyz
		)
		target_q_pose = np.zeros(7)
		target_q_pose[:6] = ik_solution
		target_q_pose[-2] = action[3] + self._data.qpos[5]
		target_q_pose[-1] = int(255 / 2 * (action[-1] + 1))
		action = target_q_pose
     

		for i in range(6):
			target_velocity = 10*(target_q_pose[i] - self._data.qpos[i])
			torque = 2*(target_velocity - self._data.qvel[i])
			action[i] = self._data.qfrc_bias[i] + torque

		return action
	

	def get_current_position(self):
		link_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "heal/end_effector")
		current_position = self._data.xpos[link_id].copy()
		print(f'Current Position: {current_position}')
		return current_position


	def set_timesteps(self, physics_timestep: float, control_timestep: float) -> None:
		"""Set the physics and control timesteps for the environment.

		The physics timestep will be assigned to the MjModel during compilation. The control timestep is used to
		determine the number of physics steps to take per control step.
		"""
		# Check timesteps divisible.
		n_steps = control_timestep / physics_timestep
		rounded_n_steps = int(round(n_steps))
		if abs(n_steps - rounded_n_steps) > 1e-6:
			raise ValueError(
				f'Control timestep {control_timestep} should be an integer multiple of '
				f'physics timestep {physics_timestep}.'
			)

		self._physics_timestep = physics_timestep
		self._control_timestep = control_timestep
		self._n_steps = rounded_n_steps

	def render(self):
		width, height = self._render_width, self._render_height
		if self.render_mode == "human":
			if self.viewer is None:
				self.viewer = mujoco.viewer.launch_passive(self._model, self._data)
			else:
				self.viewer.sync()

		elif self.render_mode == "rgb_array":
			# Initialize OpenGL context once
			if not hasattr(self, "gl_context"):
				self.gl_context = mujoco.GLContext(width, height)
				self.gl_context.make_current()
			# Initialize rendering context once
			if not hasattr(self, "mjr_context"):
				self.mjr_context = mujoco.MjrContext(self._model, mujoco.mjtFontScale.mjFONTSCALE_150)
			# Common setup
			option = mujoco.MjvOption()
			viewport = mujoco.MjrRect(0, 0, width, height)
			rgb_images = []
			camera_names = ["heal/wrist_cam", "top_bev", "front"]  # Replace with your actual camera names
			for cam_name in camera_names:
				cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
				if cam_id == -1:
					raise ValueError(f"Camera '{cam_name}' not found!")
				# Set up the camera
				camera = mujoco.MjvCamera()
				camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
				camera.fixedcamid = cam_id
				# Create a new scene per camera
				scene = mujoco.MjvScene(self._model, maxgeom=1000)
				mujoco.mjv_updateScene(
					self._model,
					self._data,
					option,
					None,
					camera,
					mujoco.mjtCatBit.mjCAT_ALL,
					scene,
				)
				# Allocate pixel buffers
				rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
				depth_buffer = np.zeros((height, width), dtype=np.float32)
				# Render and read pixels
				mujoco.mjr_render(viewport, scene, self.mjr_context)
				mujoco.mjr_readPixels(rgb_array, depth_buffer, viewport, self.mjr_context)
				# Convert RGB to BGR (for OpenCV)
				rgb_images.append(rgb_array[:, :, ::-1])
			# Return stacked horizontally or as a list

			rgb_images = [cv2.flip(image, 0) for image in rgb_images]
			return {
				'wrist_cam': rgb_images[0],
				'bev_cam': rgb_images[1],
				'front': rgb_images[2]
			}

			# stacked_image = np.hstack(rgb_images)
			# return stacked_image  # or return rgb_images if you prefer separate


	@abc.abstractmethod
	def compute_observation(self) -> Any:
		# Computes the observation at each timestep
		raise NotImplementedError


	@abc.abstractmethod
	def build_mjcf_model(self) -> mjcf.RootElement:
		# Builds the MJCF model for the environment using the 'mjcf' library
		raise NotImplementedError

	@property
	def action_space(self):
		"""Return the action space for the environment.

		By default, this returns a Box matching the actuators defined in the model. Override this method to provide a
		custom action space.
		"""
		if self._model is None:
			self.reset()
		is_limited = self._model.actuator_ctrllimited.ravel().astype(bool)
		ctrlrange = self._model.actuator_ctrlrange
		return gym.spaces.Box(
			low=np.where(is_limited, ctrlrange[:, 0], -mujoco.mjMAXVAL),
			high=np.where(is_limited, ctrlrange[:, 1], mujoco.mjMAXVAL),
			dtype=np.float32,
		)

	def terminate_episode(self) -> bool:
		# Determine whether the episode should be terminated
		raise NotImplementedError
	
	def truncate_episode(self) -> bool:
		return self.current_step > self.max_episode_steps

	def compile_model_and_data(self):
		# Compile the MJCF model and MjModel and MjData objects
		getattr(self._mjcf_model.visual, 'global').offwidth = self._render_width
		getattr(self._mjcf_model.visual, 'global').offheight = self._render_height

		self._model = mujoco.MjModel.from_xml_string(
			xml=mjcf_utils.to_string(self._mjcf_model),
			assets=mjcf_utils.get_assets(self._mjcf_model)
		)
		self._data = mujoco.MjData(self._model)

		# Assign the physics timestep.
		self._model.opt.timestep = self._physics_timestep

		mujoco.mj_resetData(self._model, self._data)
		mujoco.mj_forward(self._model, self._data)

		# Make sure the passive viewer is up-to-date.
		if self._passive_viewer_handle is not None:
			self._passive_viewer_handle._sim().load(self._model, self._data, '')

		# Re-initialize the renderer.
		if self._renderer is not None:
			self._renderer.close()
			self._initialize_renderer()

		self.post_compilation()

	def launch_passive_viewer(self, *args, **kwargs):
		"""Launch a passive viewer for the environment."""
		if self._passive_viewer_handle is not None:
			raise ValueError('Passive viewer already launched.')
		if self._model is None or self._data is None:
			raise ValueError('Call `reset` before launching the passive viewer.')
		self._passive_viewer_handle = mujoco.viewer.launch_passive(
			self._model,
			self._data,
			show_left_ui=kwargs.pop('show_left_ui', False),
			show_right_ui=kwargs.pop('show_right_ui', False),
			*args,
			**kwargs,
		)

	def _initialize_renderer(self):
		"""Initialize the renderer."""
		if self._model is None:
			raise ValueError('Call `reset` before rendering.')
		self._renderer = mujoco.Renderer(model=self._model, height=self._render_height, width=self._render_width)
		mujoco.mjv_defaultFreeCamera(self._model, self._camera)
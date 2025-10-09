"""
	Date: March 12, 2025
	Description: Implementation of SAC Agent from CleanRL.
	Author: Anant Garg (anant.garg@addverb.com)
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from agents.sac.networks import Actor, SoftQNetwork
from utils import utils

gamma = 0.99

class SAC:
	def __init__(self, obs_shape, action_shape, args):
		
		self.args = args
		self.device = self.args.device
		self.obs_dim = obs_shape
		self.action_dim = action_shape
		self.hidden_dim = self.args.hidden_dim
		self.q_lr = self.args.q_lr
		self.policy_lr = self.args.policy_lr

		self.stddev_schedule = self.args.stddev_schedule
		self.update_every_steps = self.args.update_every_steps
		self.learning_starts = self.args.learning_starts
	
		self.gamma = self.args.gamma
		self.tau = self.args.critic_target_tau    
		self.alpha = self.args.alpha
		self.autotune = self.args.autotune

		self.actor = Actor(obs_shape, action_shape, self.hidden_dim).to(self.device)
		self.qf1 = SoftQNetwork(obs_shape, action_shape, self.hidden_dim).to(self.device)
		self.qf2 = SoftQNetwork(obs_shape, action_shape, self.hidden_dim).to(self.device)
		self.qf1_target = SoftQNetwork(obs_shape, action_shape, self.hidden_dim).to(self.device)
		self.qf2_target = SoftQNetwork(obs_shape, action_shape, self.hidden_dim).to(self.device)
		self.qf1_target.load_state_dict(self.qf1.state_dict())
		self.qf2_target.load_state_dict(self.qf2.state_dict())
		self.q_optimizer = torch.optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.args.q_lr)
		self.actor_optimizer = torch.optim.Adam(list(self.actor.parameters()), lr=self.args.policy_lr)

		self.train()

		# Automatic entropy tuning
		if self.autotune:
			self.target_entropy = -torch.prod(torch.tensor(action_shape).to(self.device)).item()
			self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
			self.alpha = self.log_alpha.exp().item()
			self.a_optimizer = torch.optim.Adam([self.log_alpha], lr=self.q_lr)

	def train(self, training=True):
		self.training = training
		self.actor.train(training)
		self.qf1.train(training)
		self.qf2.train(training)
	
	def act(self, obs, step, eval_mode):
		obs = torch.as_tensor(obs, device=self.device)
		stddev = utils.schedule(self.stddev_schedule, step)
		action, _, _ = self.actor.get_action(obs.float().unsqueeze(0))
		if eval_mode:
			action = action.cpu().numpy()[0]
		else:
			action = action.cpu().numpy()[0] + np.random.normal(0, stddev, size=self.action_dim)
			if step < self.learning_starts:
				action = np.random.uniform(-1.0, 1.0, size=self.action_dim)
		return action.astype(np.float32)

	def observe(self, obs, action):
		obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
		action = torch.as_tensor(action, device=self.device).float().unsqueeze(0)

		q1, _ = self.qf1(obs, action)
		q2, _ = self.qf2(obs, action)

		return {
			'state': obs.cpu().numpy()[0],
			'value': torch.min(q1, q2).cpu().numpy()[0]
		}

	def select_action(self, state, eval_mode=False):
		state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
		action, _ = self.actor.sample(state) if not eval_mode else self.actor(state)
		return action.cpu().detach().numpy()[0]

	def update_critic(self, obs, action, reward, next_obs):
		metrics = dict()

		with torch.no_grad():
			next_state_actions, next_state_log_pi, _ = self.actor.get_action(next_obs)
			qf1_next_target = self.qf1_target(next_obs, next_state_actions)
			qf2_next_target = self.qf2_target(next_obs, next_state_actions)
			min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
			next_q_value = reward.flatten() + gamma * (min_qf_next_target).view(-1)

		qf1_a_values = self.qf1(obs, action).view(-1)
		qf2_a_values = self.qf2(obs, action).view(-1)
		qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
		qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
		qf_loss = qf1_loss + qf2_loss

		# optimize the model
		self.q_optimizer.zero_grad()
		qf_loss.backward()
		self.q_optimizer.step()

		metrics['critic_target_q'] = next_q_value.mean().item()
		metrics['critic_q1'] = qf1_a_values.mean().item()
		metrics['critic_q2'] = qf2_a_values.mean().item()
		metrics['critic_loss'] = qf_loss.item()

		return metrics

	def update_actor(self, obs):
		metrics = dict()
		pi, log_pi, _ = self.actor.get_action(obs)
		qf1_pi = self.qf1(obs, pi)
		qf2_pi = self.qf2(obs, pi)
		min_qf_pi = torch.min(qf1_pi, qf2_pi)
		actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()
		metrics['actor_loss'] = actor_loss.item()

		with torch.no_grad():
			_, log_pi, _ = self.actor.get_action(obs)
		alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

		self.a_optimizer.zero_grad()
		alpha_loss.backward()
		self.a_optimizer.step()
		alpha = self.log_alpha.exp().item()
		metrics['alpha'] = alpha
		return metrics

	def update(self, data, step):
		metrics = dict()
		obs, action, next_obs, _, reward = utils.to_torch(
			data, self.device)

		obs = obs.float()
		next_obs = next_obs.float()
		metrics['batch_reward'] = reward.mean().item()
		metrics.update(self.update_critic(obs, action, reward, next_obs))
		if step % self.update_every_steps == 0:
			metrics.update(self.update_actor(obs.detach()))
		utils.soft_update_params(self.qf1, self.qf1_target, self.args.critic_target_tau)
		utils.soft_update_params(self.qf2, self.qf2_target, self.args.critic_target_tau)
		return metrics

	def save(self, model_dir, step):
		model_save_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')
		model_save_dir.mkdir(exist_ok=True, parents=True)

		torch.save(self.actor.state_dict(), f'{model_save_dir}/actor.pt')
		torch.save(self.qf1.state_dict(), f'{model_save_dir}/qf1.pt')
		torch.save(self.qf2.state_dict(), f'{model_save_dir}/qf2.pt')

	def load(self, model_dir, step):
		print(f"Loading the model from {model_dir}, step: {step}")
		model_load_dir = Path(f'{model_dir}/step_{str(step).zfill(8)}')

		self.actor.load_state_dict(
			torch.load(f'{model_load_dir}/actor.pt', map_location=self.device)
		)
		self.qf1.load_state_dict(
			torch.load(f'{model_load_dir}/qf1.pt', map_location=self.device)
		)

		self.qf2.load_state_dict(
			torch.load(f'{model_load_dir}/qf2.pt', map_location=self.device)
		)
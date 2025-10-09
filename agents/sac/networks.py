import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class SoftQNetwork(nn.Module):
	def __init__(self, obs_dim, action_dim, hidden_dim):
		super().__init__()
		self.fc1 = nn.Linear(
			obs_dim + action_dim,
			hidden_dim,
		)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, 1)

	def forward(self, x, a):
		x = torch.cat([x, a], 1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class Actor(nn.Module):
	def __init__(self, obs_dim, action_dim, hidden_dim):
		super().__init__()
		# self.fc1 = nn.Linear(obs_dim, hidden_dim)
		# self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		# self.fc_mean = nn.Linear(hidden_dim, action_dim)

		self.fc1 = nn.Linear(obs_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, hidden_dim)
		self.fc4 = nn.Linear(hidden_dim, action_dim)
		# self.fc_logstd = nn.Linear(hidden_dim, action_dim)
		# action rescaling
		# self.register_buffer(
		# 	"action_scale",
		# 	torch.tensor(
		# 		(action_scale.high - action_scale.low) / 2.0,
		# 		dtype=torch.float32,
		# 	),
		# )
		# self.register_buffer(
		# 	"action_bias",
		# 	torch.tensor(
		# 		(action_bias.high + action_bias.low) / 2.0,
		# 		dtype=torch.float32,
		# 	),
		# )

	def forward(self, x):
		# if isinstance(x, np.ndarray):
		# 	x = torch.tensor(x, dtype=torch.float).to(self.fc1.device)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		mean = F.tanh(self.fc4(x))
		return mean, None
		log_std = F.tanh(self.fc_logstd(x))
		log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
		return mean, log_std

	def get_action(self, x):
		mean, log_std = self(x)
		std = log_std.exp()
		normal = torch.distributions.Normal(mean, std)
		x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
		y_t = torch.tanh(x_t)
		action = y_t #* self.action_scale + self.action_bias
		log_prob = normal.log_prob(x_t)
		# Enforcing Action Bound
		# log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
		log_prob -= torch.log((1 - y_t.pow(2)) + 1e-6)
		log_prob = log_prob.sum(1, keepdim=True)
		mean = torch.tanh(mean) #* self.action_scale + self.action_bias
		# return action
		return action, log_prob, mean
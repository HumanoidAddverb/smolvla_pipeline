import torch
import torch.nn as nn
import random
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from gymnasium import spaces
import wandb

# wandb.init(
#     entity='garg-anant',
#     project='sac_cobot',
#     name='test_sac_cobot',
# )


class Train(nn.Module):
    def __init__(self, args, env, agent):
        super(Train).__init__()

        self.args = args
        self.env = env
        self.device = self.args.device
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.args.obs_dim,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(self.args.action_dim,), dtype=np.float32)
        self.replay_buffer = ReplayBuffer(
            int(self.args.buffer_size),
            observation_space,
            action_space,
            self.args.device,
            handle_timeout_termination=False
        )
        # self.env.single_observation_space.dtype = np.float32
        # self.agent = agent(obs_shape=self.args.obs_dim, action_shape=self.args.action_dim, args=self.args)
        self.agent = agent
        self.current_timestep = 1

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        self.model_dir = self.args.model_dir

    def learn(self, total_timesteps):
        # obs = self.env.reset()['ob']
        obs, _ = self.env.reset()
        eps_reward = 0
        done = False
        while self.current_timestep < total_timesteps:
            if self.current_timestep < self.args.learning_starts:
                action = np.random.rand(self.args.action_dim)
            else:
                action, _, _ = self.agent.actor.get_action(torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(self.device))
                action = action.detach().cpu().numpy().reshape(-1)

            next_obs, reward, termination, truncation, infos = self.env.step(action)
            self.env.render() # Added the render function
            # next_obs = next_obs['ob']
            eps_reward += reward
            done = termination | truncation
            self.replay_buffer.add(obs, next_obs, action, reward, done, infos)

            if self.current_timestep > self.args.learning_starts:
                data = self.replay_buffer.sample(self.args.replay_buffer_batch_size)
                metrics = self.agent.update(data, self.current_timestep)
            else:
                metrics = {}

            if done:
                metrics['eps_reward'] = eps_reward
                wandb.log(metrics)
                eps_reward = 0
                obs = self.env.reset()
                
            obs = next_obs.copy()

            if self.current_timestep % self.args.log_ckpt_every == 0:
                self.agent.save(model_dir=self.model_dir, step=self.current_timestep)

            self.current_timestep += 1
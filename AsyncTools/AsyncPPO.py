import torch as t

import gymnasium as gym
import numpy as np

from tqdm import tqdm
from copy import deepcopy

import AsyncTools.utils as utils

class VecMemory:
	def __init__(self, num_envs: int):
		self.states = [[] for _ in range(num_envs)]
		self.actions = [[] for _ in range(num_envs)]
		self.rewards = [[] for _ in range(num_envs)]
		self.dones = [[] for _ in range(num_envs)]
		# self.state_values = [[] for _ in range(num_envs)]
		# self.log_probs = [[] for _ in range(num_envs)]
	
	def push(self, idx: int, state, action, reward, done):
		self.states[idx].append(state.astype(np.float32))
		self.actions[idx].append(action.astype(np.float32))
		self.rewards[idx].append(reward.astype(np.float32))
		self.dones[idx].append(done.astype(np.float32))

	def clear(self):
		for i in range(len(self.states)):
			del self.states[i][:]	
			del self.actions[i][:]	
			del self.rewards[i][:]	
			del self.dones[i][:]	
			# del self.state_values[i][:]
			# del self.log_probs[i][:]

class EnvVectorizer(gym.Env):
	def __init__(self, env: gym.Env, num_envs: int = 1):
		super().__init__()

		self.envs = [deepcopy(env) for _ in range(num_envs)]

		# Initialize envs_active as False for all environments
		self.envs_active = np.array([False] * num_envs)

		self.num_envs = num_envs
		self.action_space = env.action_space
		self.observation_space = env.observation_space

	def reset(self):
		# Reset all environments
		obs = []
		infos = []
		for i in range(self.num_envs):
			observation, info = self.envs[i].reset()
			obs.append(observation)
			infos.append(info)

		# Mark all environments as not terminalized upon reset
		self.envs_active = np.array([False] * self.num_envs)

		obs = np.stack(obs, axis=0)

		return obs, infos

	def step(self, actions: np.ndarray):
		step_results = []
		# active_envs_indices = [i for i in range(self.num_envs) if not self.envs_active[i]]
		active_envs_indices = np.arange(self.num_envs)[~self.envs_active]

		# Ensure actions are only for active environments
		# if actions.shape[0] != len(active_envs_indices):
		#     raise ValueError(f"Expected actions for {len(active_envs_indices)} active environments, but got {actions.shape[0]}")

		for i, env_idx in enumerate(active_envs_indices):
			# o - obs, r - reward, d - done, t - truncate, 
			# info - info (omg)
			o, r, d, t, info = self.envs[env_idx].step(actions[i])

			step_results.append([o, r, d, t, info])

		# Convert lists to numpy arrays
		obs = np.stack(
			[batch[0] for batch in step_results],
			axis=0
		)
		rewards = np.stack(
			[batch[1] for batch in step_results], 
			axis=0
		)
		dones = np.stack(
			[batch[2] for batch in step_results], 
			axis=0
		)
		truncates = np.stack(
			[batch[3] for batch in step_results], 
			axis=0
		)
		infos = np.stack(
			[batch[4] for batch in step_results], 
			axis=0
		)

		return obs, rewards, dones, truncates, infos

class AsyncPPO:
	def __init__(self, env: gym.Env, ppo: object, num_envs: int = 32, steps: int = 100000):
		self.env = EnvVectorizer(env, num_envs)
		
		self.num_envs = num_envs
		self.steps = steps
		self.ppo = ppo

		self.step_score = np.array(0, dtype=np.int32)
		self.reward_score = np.array(0.0, dtype=np.float32)

		self.buffer = VecMemory(num_envs)

	def worker(self):
		states = self.env.reset()[0]

		while True:
			actions = self.ppo.get_action(t.from_numpy(states))

			next_states, rewards, dones, truncates, _ =  self.env.step(actions)

			utils.buffer_append(
				self.buffer, 
					   			
				states, 
				actions, 
				rewards, 
				dones | truncates,
			
				self.env.envs_active,
				self.num_envs
			)

			self.reward_score += np.sum(rewards)
			self.step_score += np.sum(~self.env.envs_active)

			states = utils.inactive_states_dropout(next_states, dones | truncates)
			self.env.envs_active = utils.update_active_environments_list(self.env.envs_active, dones | truncates)

			if np.all(self.env.envs_active):
				utils.buffer_to_target_buffer_transfer(self.buffer, self.ppo.memory)

				break

	def run(self):
		pbar = tqdm(total=self.steps, unit='step')

		while pbar.n < self.steps:
			self.step_score = 0
			self.reward_score = 0
			
			self.worker()

			mean_reward = self.reward_score / self.num_envs

			pbar.update(min(pbar.total - pbar.n, self.step_score))
			pbar.set_description(f'Mean reward {mean_reward: .1f}')

			self.ppo.learn()
			# self.ppo.save_weights('code/Works/PPO_PRL')
		
		self.env.close()
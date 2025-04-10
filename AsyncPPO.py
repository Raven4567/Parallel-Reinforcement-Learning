import torch as t

import gymnasium as gym
import numpy as np

from tqdm import tqdm

from copy import deepcopy

import threading as thd

class AsyncPPO:
	def __init__(self, env: gym.Env, ppo: object, num_workers: int = 3, steps: int = 50000):
		self.envs = [deepcopy(env) for _ in range(num_workers)]
		
		self.num_workers = num_workers
		self.steps = steps
		self.ppo = ppo

	def worker(self, rank: int, env: gym.Env, ppo: object, sub_buffer: list, total_rewards: list, steps_score: list):
		state = env.reset()[0]

		while True:
			action, state_value, log_prob = ppo.get_action(
				t.from_numpy(state).unsqueeze(0)
			)

			next_state, reward, done, truncate, _ =  env.step(action)
		
			total_rewards[rank] += reward
			steps_score[rank] += 1

			sub_buffer[rank].append([state, action, reward, done, state_value, log_prob])

			state = next_state

			if done or truncate:
				break

	def run(self):

		pbar = tqdm(range(self.steps))

		while pbar.n < self.steps:
			steps_score = np.array([0 for _ in range(self.num_workers)])
			rewards_score = np.array([0 for _ in range(self.num_workers)])

			thread_list = []

			sub_buffer = [[] for _ in range(self.num_workers)]
			for i in range(self.num_workers):
				t = thd.Thread(target=self.worker, args=[i, self.envs[i], self.ppo, sub_buffer, rewards_score, steps_score])
				t.start()
				thread_list.append(t)
		
			for t in thread_list:
				t.join()
			
			median_reward = np.median(rewards_score, axis=0)
			
			for single_buffer in sub_buffer:
				for batch in single_buffer:
					self.ppo.memory.push(*batch)

			self.ppo.learn()
			#self.ppo.save_weights('code/Works/PPO_PRL')

			pbar.update(sum(steps_score))
			pbar.set_description(f'Median reward {median_reward: .1f}')
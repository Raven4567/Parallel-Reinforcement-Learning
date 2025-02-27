from PPO import PPO
from Graphic import Graphic

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from copy import deepcopy
#from itertools import count

import threading as thd

class PPO_PRL:
	def __init__(self, env: gym.Env, ppo: object, num_workers: int = 3, steps: int = 50000):
		self.envs = [deepcopy(env) for _ in range(num_workers)]
		
		self.num_workers = num_workers
		self.steps = steps
		self.ppo = ppo

	def worker(self, rank: int, env: gym.Env, ppo: object, total_rewards: list, steps_score: list):
		state = env.reset()[0]

		while True:
			action, state_value, log_prob = ppo.get_action(state)

			next_state, reward, done, truncate, _ =  env.step(action)
		
			total_rewards[rank] += reward
			steps_score[rank] += 1

			ppo.memorize([state, action, reward, done, state_value, log_prob], rank)

			state = next_state

			if done or truncate:
				break

	def run(self):

		pbar = tqdm(range(self.steps))
		while pbar.n < self.steps:
			steps_score = np.array([0 for _ in range(self.num_workers)])
			rewards_score = np.array([0 for _ in range(self.num_workers)])

			thread_list = []
			for i in range(self.num_workers):
				t = thd.Thread(target=self.worker, args=[i, self.envs[i], self.ppo, rewards_score, steps_score])
				t.start()
				thread_list.append(t)
		
			for t in thread_list:
				t.join()
			
			median_reward = np.median(rewards_score, axis=0)
			
			graphic.update(x=pbar.n, y=median_reward)
		
			self.ppo.education()

			self.ppo.save_weights('./data')

			pbar.update(
				np.clip(sum(steps_score), a_min=0, a_max=self.steps) if pbar.n + sum(steps_score) else sum(steps_score)
			)
			pbar.set_description(f'Median reward {median_reward: .1f}')
	
		graphic.show()

if __name__ == '__main__':
	NUM_WORKERS = 32

	env = gym.make('Pusher-v5', max_episode_steps=1000)

	ppo = PPO(
		has_continuous=True, Action_dim=env.action_space.shape[0], Observ_dim=env.observation_space.shape[0],
		action_scaling=None, Actor_lr=0.0010, Critic_lr=0.0025,
		policy_clip=0.2, k_epochs=23, GAE_lambda=0.95, 
		batch_size=2048, mini_batch_size=2048, gamma=0.995,
		use_RND=True, beta=0.04, num_workers=NUM_WORKERS
	)

	ppo.load_weights('./data')

	graphic = Graphic(
		x='Steps', y='Median rewards', title=f'Progress of learning Parallel-PPO-Agent in {env.spec.id}',
		hyperparameters={
			'Has_continuous': ppo.has_continuous,
			'Action_scaling': ppo.action_scaling if ppo.has_continuous else None,
			'Actor_lr': ppo.Actor_lr,
			'Critic_lr': ppo.Critic_lr,
			'Policy_clip': ppo.policy_clip,
			'K_epochs': ppo.k_epochs,
			'GAE_lambda': ppo.GAE_lambda,
			'Batch_size': ppo.batch_size,
			'Mini_batch_size': ppo.mini_batch_size,
			'Gamma': ppo.gamma,
			'use_RND': ppo.use_RND,
			'beta': ppo.beta,
			'Num_workers': NUM_WORKERS,
			'Action_dim': ppo.action_dim,
			'Observ_dim': ppo.observ_dim
		}
	)

	prl = PPO_PRL(
		env=env,
		ppo=ppo,
		num_workers=NUM_WORKERS,
		steps=20000000
	)

	prl.run()
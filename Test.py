from PPO import PPO

import torch as t

import gymnasium as gym

from tqdm import tqdm
from itertools import count

env = gym.make('CartPole-v1', max_episode_steps=500, render_mode='human')

ppo = PPO(
	is_continuous=False, observ_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
	# action_scaling=2.0
)

ppo.load_weights('C:/Users/Raven4567/code/Parallel-Reinforcement-Learning/PPO/data/')

for _ in (pbar := tqdm(count())):
	state, _ = env.reset()
	
	reward_per_episode = 0
	while True:
		action = ppo.get_action(t.from_numpy(state).unsqueeze(0))

		state, reward, done, truncate, _ = env.step(action.squeeze(0))

		env.render()

		reward_per_episode += reward

		pbar.set_description(f'reward: {reward_per_episode}, done: {done or truncate}')
			
		if done or truncate:
			break

env.close()
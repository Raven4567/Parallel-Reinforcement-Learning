from PPO import PPO

import gymnasium as gym

from tqdm import tqdm
from itertools import count

env = gym.make('CartPole-v1', max_episode_steps=500, render_mode='human')

ppo = PPO(
	has_continuous=False, action_dim=env.action_space.n, observ_dim=env.observation_space.shape[0],
)

#ppo.load_weights('code/Works/PPO_PRL')

for _ in (pbar := tqdm(count())):
	state, _ = env.reset()
	
	reward_per_episode = 0
	while True:
		action, _, _ = ppo.get_action(state)

		state, reward, done, truncate, _ = env.step(action)

		reward_per_episode += reward

		pbar.set_description(f'reward: {reward_per_episode}, done: {done or truncate}')
			
		if done or truncate:
			break
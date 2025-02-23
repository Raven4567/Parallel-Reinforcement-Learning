from PPO import PPO

import gymnasium as gym

from tqdm import tqdm
from itertools import count

env = gym.make('Pusher-v5', max_episode_steps=1000, render_mode='human')

ppo = PPO(
	has_continuous=True, Action_dim=env.action_space.shape[0], Observ_dim=env.observation_space.shape[0],
	action_scaling=2.0, num_workers=1
)

ppo.load_weights('./data')

for _ in (pbar := tqdm(count())):
	state = env.reset()[0]
	
	reward_per_episode = 0
	while True:
		action, value, log_prob = ppo.get_action(state)

		state, reward, done, truncate, _ = env.step(action)

		reward_per_episode += reward

		if done or truncate:
			pbar.set_description(f'Reward per episode: {reward_per_episode}')
			
			break
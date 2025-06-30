# Parallel-Reinforcement-Learning

**Languages:** [English](README.md) | [Русский](README.ru.md) | [Deutsch](README.de.md) | [Español](README.es.md) | [中文](README.zh-CN.md)

## Description
Small program that makes PPO learn using many enviroments at the same time, anyschronously, accelerating the learning and exploration.

## Installation 
run:
```
git clone https://github.com/Raven4567/Parallel-Reinforcement-Learning
```
then run:
```
pip install -r requirements.txt
```
being in the installed `.../Parallel-Reinforcement-Learning` folder.

## Quick start
```python
from PPO import PPO
from AsyncPPO import AsyncPPO

import gymnasium as gym

if __name__ == '__main__':
	env = gym.make('CartPole-v1', max_episode_steps=500)

	ppo = PPO(
		is_continuous=False, 
		observ_dim=env.observation_space.shape[0],
		action_dim=env.action_space.n, 
		lr=0.001, 
		# action_scaling=2.0
		policy_clip=0.2, 
		k_epochs=11, 
		GAE_lambda=0.95, 
		batch_size=1024, 
		mini_batch_size=512, 
		gamma=0.995,
		# use_RND=True, 
		# beta=0.001
	)

	async_ppo = AsyncPPO(
		env=env,
		ppo=ppo,
		num_envs=32,
		steps=100000
	)

	async_ppo.run()

	async_ppo.ppo.save_weights(path='(insert your path)/Parallel-Reinforcement-Learning/PPO/data')
```

## `PPO` parameters:

- `is_continuous` - set True if the environment demands continuous actions (False means discrete actions, and True means continuous ones)
- `observ_dim` - number of state features (e. g. `observ_dim=4` for CartPole-v1 or `observ_dim=348` for Humanoid-v5)
- `action_dim`  - number of possible actions (e. g. `action_dim=2` for CartPole-v1 or `action_dim=23` for Pusher-v5)
- `lr` - value of learning rate for optimizer.
- `action_scaling` - multiplier for actions, for example for Pusher-v5 we've to use `action_scaling=2.0` because the range of actions in Pusher-v5 is (-2, 2) and our network outputs only (-1, 1) actions if `is_continuous=True`, so it uses `action_scaling` for scaling of actions towards the right range.
- `policy_clip` - value of policy changes, e. g. `policy_clip=0.2` allows changes not more than 20%
- `k_epochs` - number of epochs for network learning on one set of data.
- `GAE_lambda` - smoothing factor for advantage calculation (0 = high variance, 1 = lower variance).
- `batch_size` - batch size
- `mini_batch_size` - mini-batch size
- `gamma` - affects the consideration of long-term rewards (usually 0.99-0.999)
- `use_RND` - whether we'll be using *Random Network Distillation*.
- `beta` - multiplier for `RND` rewards

More about RND - https://openai.com/index/reinforcement-learning-with-prediction-based-rewards/

## Custom loop

**WARNING**: the line `env.envs_active = utils.update_active_environments_list(env.envs_active, dones | truncates)` and `states = utils.inactive_states_dropout(next_states, dones | truncates)` are most important, lack of them will break the learning.

Also in this example I'll be using my own implementation, but feel free to copy this code and rewrite it for your implementation.

```python
# Import
from PPO import PPO
from AsyncPPO import EnvVectorizer, VecMemory

import utils

import torch as t
import numpy as np

from tqdm import tqdm

import gymnasium as gym

# Main loop
if __name__ == '__main__':
	# Create environment
	env = gym.make('CartPole-v1')

	# Initialise neural network (or your own implementation)
	ppo = PPO(
		is_continuous=False, 
		observ_dim=env.observation_space.shape[0],
		action_dim=env.action_space.n, 
		lr=0.001,
		# action_scaling=1.0,
		policy_clip=0.2, 
		k_epochs=11, 
		GAE_lambda=0.95, 
		batch_size=1024, 
		mini_batch_size=512, 
		gamma=0.995,
		use_RND=True, 
		beta=0.001
	)

	env = EnvVectorizer(env=env, num_envs=32) # Vectorised env
	buffer = VecMemory(num_envs=32) # Vectorized buffer with one buffer for every env

	# Data collecting loop with tqdm progress bar
	pbar = tqdm(
		total=100000,
		unit='step'
	)

	while pbar.n < pbar.total:
		states = env.reset()[0] # Get states

		rewards_score = np.array(0.) # Reset rewards score
		steps_score = np.array(0) # Reset steps score

		while True:
			# Get actions, state values, and log probabilities
			actions = ppo.get_action(t.from_numpy(states)) 

			# Execute steps
			next_states, rewards, dones, truncates, _ = env.step(actions) 

			# Append data in our AsyncPPO.VecMemory buffer
			utils.buffer_append(
				buffer,

				states, 
				actions, 
				rewards, 
				dones,

				is_env_terminal=env.envs_active,
				num_envs=32
			) 

			# Sifting states with done or truncate = True features, and also update envs activity list
			states = utils.inactive_states_dropout(next_states, dones | truncates) 
			env.envs_active = utils.update_active_environments_list(env.envs_active, dones | truncates)

			rewards_score += sum(rewards) # Update rewards score
			steps_score += sum(~env.envs_active) # Update steps score

			# If all environments are terminal we finish the episode
			if np.all(env.envs_active): 
				# Transfer data from our local buffer to ppo.memory buffer for ppo learning. You also can use your own function for transfer data in your own neural network buffer.
				utils.buffer_to_target_buffer_transfer(buffer, ppo.memory) 
				ppo.learn() # Launch the learning function

				pbar.set_description(f'Mean reward {rewards_score / 32: .1f}')
				pbar.update(min(pbar.total - pbar.n, steps_score))

				break # Get out from the episode and start new one
```

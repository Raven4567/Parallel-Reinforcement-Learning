from PPO import PPO
from AsyncTools.AsyncPPO import AsyncPPO

import gymnasium as gym

if __name__ == '__main__':
	print("Initializing of the environment...")
	env = gym.make('CartPole-v1')

	print("Initializing of PPO...")
	ppo = PPO(
		is_continuous=False,
		observ_dim=env.observation_space.shape[0],
		action_dim=env.action_space.n,
		# action_scaling=1.0,
		lr=0.0003,
		k_epochs=7,
		policy_clip=0.2,
		GAE_lambda=0.95,
		gamma=0.995,
		batch_size=512,
		mini_batch_size=256,
		# use_RND=True,
		# beta=0.001
	)
	
	print("Initializing of asynchronous PPO...")
	async_ppo = AsyncPPO(
		env=env,
		ppo=ppo,
		num_envs=32,
		steps=300_000
	)

	print("Start training...")
	async_ppo.run()
	print("Training is completed.")

	ppo.save_weights(path='PPO/data/')
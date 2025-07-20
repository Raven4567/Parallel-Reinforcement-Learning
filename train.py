from PPO import PPO
from AsyncTools.AsyncPPO import AsyncPPO

import gymnasium as gym

if __name__ == '__main__':
	print("Initializing of the environment...")
	env = gym.make('CartPole-v1')

	print("Initializing of PPO...")
	ppo = PPO(
		is_continuous=True, 
		observ_dim=env.observation_space.shape[0],
		action_dim=env.action_space.n, 
		lr=0.001,
		# action_scaling=2.0,
		policy_clip=0.2, 
		k_epochs=7, 
		GAE_lambda=0.95, 
		batch_size=1024, 
		mini_batch_size=32,
		gamma=0.995,
		# use_RND=True, 
		# beta=0.001
	)
	
	print("Initializing of asynchronous PPO...")
	async_ppo = AsyncPPO(
		env=env,
		ppo=ppo,
		num_envs=32,
		steps=100000
	)

	print("Start training...")
	async_ppo.run()
	print("Training is completed.")

	ppo.save_weights(path='PPO/data/')
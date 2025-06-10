from PPO import PPO
from AsyncPPO import AsyncPPO, EnvVectorizer, VecMemory

import gymnasium as gym

if __name__ == '__main__':
	env = gym.make('CartPole-v1', max_episode_steps=500)

	ppo = PPO(
		is_continuous=False, action_dim=env.action_space.n, observ_dim=env.observation_space.shape[0],
		actor_lr=0.0010, critic_lr=0.0025,# action_scaling=1.0,
		policy_clip=0.2, k_epochs=11, GAE_lambda=0.95, 
		batch_size=1024, mini_batch_size=1024, gamma=0.995,
		# use_RND=True, beta=0.01
	)
		
	async_ppo = AsyncPPO(
		env=env,
		ppo=ppo,
		num_envs=32,
		steps=100000
	)

	async_ppo.run()

	ppo.save_weights(path='C:/Users/Raven4567/code/PPO_PRL/PPO/data/')
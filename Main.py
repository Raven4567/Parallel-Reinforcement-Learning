from PPO import PPO
from AsyncPPO import AsyncPPO

import gymnasium as gym

if __name__ == '__main__':
	NUM_WORKERS = 32

	env = gym.make('CartPole-v1', max_episode_steps=500)

	ppo = PPO(
		has_continuous=False, action_dim=env.action_space.n, observ_dim=env.observation_space.shape[0],
		action_scaling=None, Actor_lr=0.0010, Critic_lr=0.0025,
		policy_clip=0.2, k_epochs=3, GAE_lambda=0.95, 
		batch_size=128, mini_batch_size=128, gamma=0.995,
		use_RND=True, beta=0.02
	)
	#ppo.load_weights('code/Works/PPO_PRL')

	prl = AsyncPPO(
		env=env,
		ppo=ppo,
		num_workers=NUM_WORKERS,
		steps=50000
	)

	prl.run()
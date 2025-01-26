from PPO import PPO
from Graphic import Graphic

import gymnasium as gym
import numpy as np
from tqdm import tqdm
#from itertools import count

import threading as thd

def worker(rank, env, ppo, total_rewards):
    env = envs[rank]

    state = env.reset()[0]

    while True:
        action, state_value, log_prob = ppo.get_action(state)

        next_state, reward, done, truncate, _ =  env.step(action)
        
        total_rewards[rank].append(reward)
        #ppo.sub_buffer[rank].push(state, action, reward, done, state_value, log_prob)
        ppo.memorize([state, action, reward, done, state_value, log_prob], rank)

        state = next_state

        if done or truncate:
            break

def Main():
    for episode in (pbar := tqdm(range(EPISODES))):
        total_rewards = [[] for _ in range(NUM_WORKERS)]

        thread_list = []
        for i in range(NUM_WORKERS):
            
            t = thd.Thread(target=worker, args=[i, envs, ppo, total_rewards])
            t.start()
            thread_list.append(t)
        
        for t in thread_list:
            t.join()
        
        median_reward = np.median([np.sum(i, axis=0) for i in total_rewards], axis=0)
        #print(f'Median reward by episode {episode+1}: {median_reward: .1f}')
        graphic.update(x=episode+1, y=median_reward)
        
        ppo.education()

        pbar.set_description(f'Median reward {median_reward: .1f} by episode: {episode+1}')
    
    graphic.show()

if __name__ == '__main__':
    NUM_WORKERS = 32
    EPISODES = 8000

    envs = [gym.make('Humanoid-v5') for _ in range(NUM_WORKERS)]

    ppo = PPO(
        has_continuous=True, Action_dim=envs[0].action_space.shape[0], Observ_dim=envs[0].observation_space.shape[0],
        Actor_lr=0.0010, Critic_lr=0.0025, action_scaling=0.4,
        policy_clip=0.2, k_epochs=23, GAE_lambda=0.95, 
        batch_size=2048, mini_batch_size=2048, gamma=0.995,
        use_RND=False, beta=None, num_workers=NUM_WORKERS
    )
                        
    graphic = Graphic(
        x='Episodes', y='Median rewards', title=f'Progress of learning Parallel-PPO-Agent in {envs[0].spec.id}',
        hyperparameters={
            'Has_continuous': ppo.has_continuous,
            'Actor_lr': ppo.Actor_lr,
            'Critic_lr': ppo.Critic_lr,
            'Policy_clip': ppo.policy_clip,
            'K_epochs': ppo.k_epochs,
            'GAE_lambda': ppo.GAE_lambda,
            'Gamma': ppo.gamma,
            'Batch_size': ppo.batch_size,
            'Mini_batch_size': ppo.mini_batch_size,
            'Action_scaling': ppo.action_scaling if ppo.has_continuous else None,
            'Num_workers': NUM_WORKERS,
            'Action_dim': ppo.action_dim,
            'Observ_dim': ppo.observ_dim
        }
    )

    Main()
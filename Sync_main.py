from PPO import PPO
from Graphic import Graphic

import gymnasium as gym
import numpy as np
from itertools import count

#from torch import multiprocessing as mp

def main():
    states, _ = envs.reset()
    
    for episode in count():
        
        total_rewards = [[] for _ in range(num_workers)]

        while True:
            actions, values, log_probs = ppo.get_action(states)

            next_states, rewards, dones, truncates, _ = envs.step(actions)

            [sub_buffer[i].append([states[i], actions[i], rewards[i], dones[i], values[i], log_probs[i]]) for i in range(num_workers)]
            [total_rewards[i].append(rewards[i]) for i in range(num_workers)]

            states = next_states

            if np.all(dones) or np.all(truncates):
                break
        
        median_reward = np.median(np.sum(total_rewards, axis=1), axis=0)

        print(f"Median reward of the episode {episode+1}: {median_reward: .1f}")
        graphic.update(x=episode+1, y=median_reward)

        for buffer in sub_buffer:
            for batch in buffer:
                ppo.memory.push(*batch)
            
            del buffer[:]
        
        ppo.education()
                
        ppo.save_weights('C:/Users/Эльдар/code/Works/PPO_PRL')

if __name__ == '__main__':
    num_workers = 32
    #episodes = 200

    envs = gym.make_vec('Pusher-v5', num_envs=num_workers, vectorization_mode='sync', max_episode_steps=2000)

    ppo = PPO(
        has_continuous=True, Action_dim=envs.action_space.shape[1], Observ_dim=envs.observation_space.shape[1],
        action_scaling=2.0, Actor_lr=0.0001, Critic_lr=0.0025,
        policy_clip=0.2, k_epochs=23, GAE_lambda=0.95,
        batch_size=2048, mini_batch_size=2048, use_RND=True
    )

    sub_buffer = [[] for _ in range(num_workers)]

    graphic = Graphic(
        x='Episodes', y='Median rewards', title=f'Progress of learning Parallel-PPO-Agent in {envs.spec.id}',
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
            'Action_dim': ppo.action_dim,
            'Observ_dim': ppo.observ_dim
        }
    )

    main()
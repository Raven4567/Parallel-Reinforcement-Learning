from PPO import PPO
#from Graphic import Graphic

#import numpy as np

import gymnasium as gym
from tqdm import tqdm

from itertools import count

env = gym.make('CartPole-v1', render_mode='human')

ppo = PPO(
        has_continuous=False, Action_dim=env.action_space.n, Observ_dim=env.observation_space.shape[0],
        Actor_lr=0.001, Critic_lr=0.0025,
        GAE_lambda=0.95, policy_clip=0.2, k_epochs=23,
        gamma=0.995, batch_size=64, mini_batch_size=64,
        use_RND=False, num_workers=1
    )

ppo.load_weights('C:/Users/Эльдар/code/Works/PPO_PRL')

def main():
    for i in (pbar := tqdm(count(start=1))):
        state, info = env.reset()

        total_reward = 0
        while True:
            action, value, log_prob = ppo.get_action(state)

            next_state, reward, done, truncate, _ = env.step(action)
            env.render()

            pbar.set_description(f"action: {action}, reward: {total_reward}, terminate: {done or truncate}")

            #ppo.memory.push(state, action, reward, done, value, log_prob)
            
            total_reward += reward
            state = next_state

            if done or truncate:
                break
        
        #ppo.education()

if __name__ == '__main__':
    main()
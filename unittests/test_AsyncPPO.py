import unittest

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PPO import PPO

import AsyncPPO
import gymnasium as gym

import numpy as np

class TestVecMemory(unittest.TestCase):
    def setUp(self):
        # Dummy data
        self.state = np.random.randn(1)
        self.action = np.random.randint(0, 2)
        self.reward = np.random.rand()
        self.done = np.random.choice([True, False])
        self.state_value = np.random.randn(4)
        self.log_prob = np.random.rand()
    
        self.memory = AsyncPPO.VecMemory(num_envs=4)

    def test_push(self):
        self.memory.push(
            idx=0,
            state=self.state,
            action=self.action,
            reward=self.reward,
            done=self.done,
            state_value=self.state_value,
            log_prob=self.log_prob
        )
    
    def test_clear(self):
        self.memory.clear()

class TestEnvVectorizer(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('CartPole-v1')
        self.env_vectorizer = AsyncPPO.EnvVectorizer(env=self.env, num_envs=4)
    
    def test_reset(self):
        self.env_vectorizer.reset()
    
    def test_step(self):
        self.env_vectorizer.reset()
        
        self.env_vectorizer.step(actions=np.random.randint(0, 2, size=4))

class TestAsyncPPO(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('CartPole-v1')
        self.model = PPO(
            is_continuous=False, action_dim=2, observ_dim=4
        )

        self.async_ppo = AsyncPPO.AsyncPPO(
            env=self.env,
            ppo=self.model,
            num_envs=4,
            steps=1000
        )
    
    def test_worker(self):
        self.async_ppo.worker()

    def test_run(self):
        self.async_ppo.run()

if __name__ == '__main__':
    unittest.main()
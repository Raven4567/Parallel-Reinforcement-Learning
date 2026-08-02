import pytest

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PPO import PPO

import AsyncTools
import gymnasium as gym

import numpy as np

class TestVecMemory:
    @pytest.fixture
    def vec_memory_setup(self):
        memory = AsyncTools.AsyncPPO.VecMemory(num_envs=4)
    
        # Dummy data
        state = np.random.randn(1)
        action = np.random.randint(0, 2, size=(1,))
        reward = np.random.rand(1)
        done = np.random.choice([True, False], size=(1,))
        # state_value = np.random.randn(4)
        # log_prob = np.random.rand()
        
        return {
            "memory": memory,
            "state": state,
            "action": action,
            "reward": reward,
            "done": done
            # "state_value": state_value,
            # "log_prob": log_prob
        }
    
    def test_push(self, vec_memory_setup):
        memory = vec_memory_setup["memory"]
        
        state = vec_memory_setup["state"]
        action = vec_memory_setup["action"]
        reward = vec_memory_setup["reward"]
        done = vec_memory_setup["done"]
        # state_value = vec_memory_setup["state_value"]
        # log_prob = vec_memory_setup["log_prob"]

        memory.push(
            idx=0,
            state=state,
            action=action,
            reward=reward,
            done=done,
            # state_value=self.state_value,
            # log_prob=self.log_prob
        )
    
    def test_clear(self, vec_memory_setup):
        memory = vec_memory_setup["memory"]
        memory.clear()

class TestEnvVectorizer:
    @pytest.fixture
    def setup_env_vectorizer(self):
        env = gym.make('CartPole-v1')
        env_vectorizer = AsyncTools.AsyncPPO.EnvVectorizer(env, num_envs=4)
        
        return env_vectorizer
    
    def test_reset(self, setup_env_vectorizer):
        env_vectorizer = setup_env_vectorizer
        env_vectorizer.reset()
    
    def test_step(self, setup_env_vectorizer):
        env_vectorizer = setup_env_vectorizer
        env_vectorizer.reset()
        env_vectorizer.step(actions=np.random.randint(0, 2, size=4))

class TestAsyncPPO:
    @pytest.fixture
    def setup_async_ppo(self):
        env = gym.make('CartPole-v1')
        model = PPO(
            is_continuous=False, action_dim=2, observ_dim=4
        )

        async_ppo = AsyncTools.AsyncPPO.AsyncPPO(
            env=env,
            ppo=model,
            num_envs=4,
            steps=1000
        )
        
        return async_ppo
    
    def test_worker(self, setup_async_ppo):
        async_ppo = setup_async_ppo
        async_ppo.worker()

    def test_run(self, setup_async_ppo):
        async_ppo = setup_async_ppo
        async_ppo.run()

if __name__ == "__main__":
    pytest.main()
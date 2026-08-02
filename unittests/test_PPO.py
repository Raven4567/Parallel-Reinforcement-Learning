import pytest

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PPO import PPO, ActorCritic, Memory

import torch as t
import numpy as np

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class TestPPO_Discrete:
    @pytest.fixture
    def setup_ppo(self):
        ppo = PPO(is_continuous=False, observ_dim=4, action_dim=2)
    
        # Dummy dataset
        for i in range(100):
            ppo.memory.push(
                state=np.random.randn(4),
                action=np.random.randint(0, 2, size=(1,)),
                reward=np.random.rand(1),
                done=np.random.choice([True, False]),
                # state_value=np.random.rand(),
                # log_prob=np.random.rand()
            )
        
        return ppo
    
    def test_get_action(self, setup_ppo):
        ppo = setup_ppo
        actions = ppo.get_action(
            t.from_numpy(np.random.randn(2, 4))
        )

        assert actions.shape == (2,)

class TestPPO_Continuous:
    @pytest.fixture
    def setup_ppo(self):
        ppo = PPO(is_continuous=True, observ_dim=4, action_dim=2, action_scaling=1.0)
    
        # Dummy dataset
        for i in range(100):
            ppo.memory.push(
                state=np.random.randn(4),
                action=np.random.randint(0, 2, size=(1,)),
                reward=np.random.rand(1),
                done=np.random.choice([True, False]),
                # state_value=np.random.rand(),
                # log_prob=np.random.rand()
            )
        
        return ppo

    def test_get_action(self, setup_ppo):
        ppo = setup_ppo
        actions = ppo.get_action(
            t.randn(1, 4)
        )

        assert actions.shape == (1, ppo.action_dim)
        assert actions.dtype == np.float32

class TestActorCriticDiscrete:
    @pytest.fixture
    def setup_actorcritic(self):
        actorcritic = ActorCritic(is_continuous=False, observ_dim=4, action_dim=2)
        return actorcritic

    def test_get_dist(self, setup_actorcritic):
        actorcritic = setup_actorcritic
        actorcritic.get_dist(
            state=t.randn(1, 4).to(device)
        )
    
    def test_get_state_value(self, setup_actorcritic):
        actorcritic = setup_actorcritic
        actorcritic.get_state_value(
            state=t.randn(1, 4).to(device)
        )
    
    def test_get_evaluate(self, setup_actorcritic):
        actorcritic = setup_actorcritic
        actorcritic.get_evaluate(
            states=t.randn(1, 4).to(device),
            actions=t.randint(0, 2, size=(1, 2)).to(device)
        )

class TestActorCriticContinuous:
    @pytest.fixture
    def setup_actorcritic(self):
        actorcritic = ActorCritic(is_continuous=True, observ_dim=4, action_dim=2)
        return actorcritic

    def test_get_dist(self, setup_actorcritic):
        actorcritic = setup_actorcritic
        actorcritic.get_dist(
            state=t.randn(1, 4).to(device)
        )
    
    def test_get_state_value(self, setup_actorcritic):
        actorcritic = setup_actorcritic
        actorcritic.get_state_value(
            state=t.randn(1, 4).to(device)
        )
    
    def test_get_evaluate(self, setup_actorcritic):
        actorcritic = setup_actorcritic
        actorcritic.get_evaluate(
            states=t.randn(1, 4).to(device),
            actions=t.randn(1, 2).to(device)
        )

# class TestRND(unittest.TestCase):
#     def setUp(self):
#         self.rnd = RND(4, 4, beta=0.001)
    
#     def test_compute_intristic_reward(self):
#         states = t.randn(32, 4).to(device)
#         values = t.utils.data.DataLoader(
#             states,
#             batch_size=16
#         )

#         intrinsic_reward = self.rnd.compute_intrinsic_reward(values)

#         self.assertEqual(intrinsic_reward.shape, (32,))
    
#     def test_update_pred(self):
#         states = t.randn(32, 4).to(device)
#         values = t.utils.data.DataLoader(
#             states,
#             batch_size=16
#         )

#         self.rnd.update_pred(values)

class TestMemory:
    @pytest.fixture
    def setup_memory(self):
        memory = Memory()
        return memory

    def test_push(self, setup_memory):
        memory = setup_memory
        memory.push(
            state=np.random.randn(4),
            action=np.random.randint(0, 2, size=(1,)),
            reward=np.random.rand(1),
            done=np.random.choice([True, False]),
            # state_value=np.random.rand(),
            # log_prob=np.random.rand()
        )
    
    def test_clear(self, setup_memory):
        memory = setup_memory
        memory.clear()

class TestUtils:
    def test_compute_gae(self):
        from PPO import utils
        
        rewards = np.random.rand(100)
        dones = np.random.choice([0, 1], size=(100,))
        state_values = np.random.rand(100)
        next_value = np.random.rand(1)

        returns = utils.compute_gae(
            rewards,
            dones,
            state_values,
            next_value,
            gamma=0.995,
            GAE_lambda=0.95
        )

        assert len(returns) == 100
    
    def test_batch_packer(self):
        from PPO import utils

        # Test for values that are divisible by batch_size
        values = [t.randn(128, 4), t.randn(128, 2)]
        batches = utils.batch_packer(values, batch_size=32)

        for batch_v1, batch_v2 in zip(*batches):
            assert batch_v1.shape == (32, 4)
            assert batch_v2.shape == (32, 2)
        
        # Test for values that are not divisible by batch_size

        values = [t.randn(136, 4), t.randn(136, 2)]
        batches = utils.batch_packer(values, batch_size=32)

        for i, (batch_v1, batch_v2) in enumerate(zip(*batches)):
            if i < 4:
                assert batch_v1.shape == (32, 4)
                assert batch_v2.shape == (32, 2)
            else:
                assert batch_v1.shape == (8, 4)
                assert batch_v2.shape == (8, 2)
        
        # Test for single value input
        values = t.randn(128, 4)

        batch = utils.batch_packer(values, batch_size=32)

        for batch_v1 in batch:
            assert batch_v1.shape == (32, 4)

if __name__ == "__main__":
    pytest.main()
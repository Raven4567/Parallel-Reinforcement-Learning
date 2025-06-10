import unittest

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PPO import PPO, ActorCritic, RND, Memory

import torch as t
import numpy as np

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class TestPPO_Discrete(unittest.TestCase):
    def setUp(self):
        self.ppo = PPO(is_continuous=False, action_dim=2, observ_dim=4, action_scaling=None)
    
        # Dummy dataset
        for i in range(100):
            self.ppo.memory.push(
                state=np.random.randn(4),
                action=np.random.randint(0, 2),
                reward=np.random.rand(),
                done=np.random.choice([True, False]),
                state_value=np.random.rand(),
                log_prob=np.random.rand()
            )
    
    def test_get_action(self):
        actions, state_values, log_probs = self.ppo.get_action(
            t.from_numpy(np.random.randn(1, 4))
        )

        self.assertEqual(actions.shape, (1,))
        self.assertEqual(state_values.shape, (1,))
        self.assertEqual(log_probs.shape, (1,))
    
    def test_batch_packer(self):
        dummy_values = t.randn(224, 4)
        
        batch = self.ppo.batch_packer(
            dummy_values,
            batch_size=32
        )

        dummy_values_list = [t.randn(224, 4), t.randint(0, 2, size=(224,)), t.rand(224), t.randint(0, 2, size=(224,))]
        batches = self.ppo.batch_packer(
            dummy_values_list,
            batch_size=32
        )

        self.assertEqual(len(batches), len(dummy_values_list))

class TestPPO_Continuous(unittest.TestCase):
    def setUp(self):
        self.ppo = PPO(is_continuous=True, action_dim=2, observ_dim=4, action_scaling=2.0)
    
        # Dummy dataset
        for i in range(100):
            self.ppo.memory.push(
                state=np.random.randn(4),
                action=np.random.randint(2),
                reward=np.random.rand(),
                done=np.random.choice([True, False]),
                state_value=np.random.rand(),
                log_prob=np.random.rand()
            )
    
    def test_get_action(self):
        actions, state_values, log_probs = self.ppo.get_action(
            t.randn(1, 4)
        )

        self.assertEqual(actions.shape, (1, self.ppo.action_dim,))
        self.assertEqual(state_values.shape, (1,))
        self.assertEqual(log_probs.shape, (1,))
    
    def test_batch_packer(self):
        dummy_values = t.randn(224, 4)
        
        batch = self.ppo.batch_packer(
            dummy_values,
            batch_size=32
        )

        dummy_values_list = [t.randn(224, 4), t.randint(0, 2, size=(224,)), t.rand(224), t.randint(0, 2, size=(224,))]
        batches = self.ppo.batch_packer(
            dummy_values_list,
            batch_size=32
        )

        self.assertEqual(len(batches), len(dummy_values_list))

class TestActorCriticDiscrete(unittest.TestCase):
    def setUp(self):
        self.actorcritic = ActorCritic(2, 4, is_continuous=False, action_scaling=None)
    
    def test_get_dist(self):
        self.actorcritic.get_dist(
            state=t.randn(1, 4).to(device)
        )
    
    def test_get_state_value(self):
        self.actorcritic.get_state_value(
            state=t.randn(1, 4).to(device)
        )
    
    def test_get_evaluate(self):
        self.actorcritic.get_evaluate(
            states=t.randn(1, 4).to(device),
            actions=t.randint(0, 2, size=(1, 2)).to(device)
        )

class TestActorCriticContinuous(unittest.TestCase):
    def setUp(self):
        self.actorcritic = ActorCritic(2, 4, is_continuous=True, action_scaling=1)
    
    def test_get_dist(self):
        self.actorcritic.get_dist(
            state=t.randn(1, 4).to(device)
        )
    
    def test_get_state_value(self):
        self.actorcritic.get_state_value(
            state=t.randn(1, 4).to(device)
        )
    
    def test_get_evaluate(self):
        self.actorcritic.get_evaluate(
            states=t.randn(1, 4).to(device),
            actions=t.randn(1, 2).to(device)
        )

class TestRND(unittest.TestCase):
    def setUp(self):
        self.rnd = RND(4, 4, beta=0.01, k_epochs=1)
    
    def test_compute_intristic_reward(self):
        states = t.randn(32, 4).to(device)
        values = t.utils.data.DataLoader(
            states,
            batch_size=16
        )

        intrinsic_reward = self.rnd.compute_intristic_reward(values)

        self.assertEqual(intrinsic_reward.shape, (32,))
    
    def test_update_pred(self):
        states = t.randn(32, 4).to(device)
        values = t.utils.data.DataLoader(
            states,
            batch_size=16
        )

        self.rnd.update_pred(values)

class TestMemory(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()

        # Dummy dataset
        for i in range(100):
            self.memory.push(
                state=np.random.randn(4),
                action=np.random.randint(0, 2),
                reward=np.random.randn(),
                done=np.random.choice([True, False]),
                state_value=np.random.rand(),
                log_prob=np.random.rand()
            )

    def test_push(self):
        self.memory.push(
            state=np.random.randn(4),
            action=np.random.randint(0, 2),
            reward=np.random.randn(),
            done=np.random.choice([True, False]),
            state_value=np.random.rand(),
            log_prob=np.random.rand()
        )
    
    def test_clear(self):
        self.memory.clear()

if __name__ == '__main__':
    unittest.main()
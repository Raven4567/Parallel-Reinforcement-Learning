import unittest

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from AsyncPPO import VecMemory
from PPO import Memory

import utils
import numpy as np

class TestUtils(unittest.TestCase):
    def test_indexes_of_active_environments(self):
        utils.indexes_of_active_environments(4, np.random.choice([False, True], 4))

    def test_number_of_active_environments(self):
        utils.number_of_active_environments(np.random.choice([False, True], 4))

    def test_range_of_active_environments(self):
        utils.range_of_active_environments(np.random.choice([False, True], 4))
    
    def test_inactive_states_dropout(self):
        utils.inactive_states_dropout(np.random.randn(4, 4), np.random.choice([False, True], 4))
    
    def test_buffer_append(self):
        utils.buffer_append(
            buffer=VecMemory(num_envs=4),

            states=np.random.randn(4, 4),
            actions=np.random.randint(0, 2, size=4),
            rewards=np.random.randn(4),
            dones=np.random.choice([False, True], size=4),
            state_values=np.random.randn(4),
            log_probs=np.random.randn(4),

            is_env_terminal=np.random.choice([False, True], size=4),
            num_envs=4
        )
    
    def test_update_active_environments_list(self):
        random_enviroments_activity = np.random.choice([False, True], size=4)
        utils.update_active_environments_list(random_enviroments_activity, np.random.choice([False, True], size=4-np.sum(random_enviroments_activity)))

    def test_buffer_to_target_buffer_transfer(self):
        buffer = VecMemory(num_envs=4)

        # Dummy dataset
        for i in range(4):
            random_length = np.random.randint(0, 100)

            buffer.states[i] = [np.random.randn(4) for _ in range(random_length)]
            buffer.actions[i] = [np.random.randint(0, 2) for _ in range(random_length)]
            buffer.rewards[i] = [np.random.randn() for _ in range(random_length)]
            buffer.dones[i] = [np.random.choice([False, True]) for _ in range(random_length)]
            buffer.state_values[i] = [np.random.randn() for _ in range(random_length)]
            buffer.log_probs[i] = [np.random.randn() for _ in range(random_length)]

        utils.buffer_to_target_buffer_transfer(
            buffer,
            target_buffer=Memory()
        )

if __name__ == '__main__':
    unittest.main()
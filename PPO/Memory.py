import torch as t

import numpy as np

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
    
    def push(
            self, 
            state: np.ndarray, 
            action: np.ndarray, 
            reward: np.ndarray, 
            done: np.ndarray
        ):
        self.states.append(state.astype(np.float32))
        self.actions.append(action.astype(np.float32))
        self.rewards.append(reward.astype(np.float32))
        self.dones.append(done.astype(np.float32))
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
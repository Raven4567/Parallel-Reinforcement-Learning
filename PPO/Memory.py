import torch as t

import numpy as np

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.state_values = []
        self.log_probs = []
    
    def push(self, state, action, reward, done, state_value, log_prob):
        self.states.append(np.array(state, dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.float32))
        self.rewards.append(np.array(reward, dtype=np.float32))
        self.dones.append(np.array(done, dtype=np.float32))
        self.state_values.append(np.array(state_value, dtype=np.float32))
        self.log_probs.append(np.array(log_prob, dtype=np.float32))
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.dones[:]
        del self.state_values[:]
        del self.log_probs[:]
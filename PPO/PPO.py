import torch as t
from torch import nn, optim

import numpy as np
from tqdm import tqdm

from .ActorCritic import ActorCritic
from .RND import RND
from .Memory import Memory

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class PPO:
    def __init__(
            self, 
            is_continuous: bool, 
            observ_dim: int, 
            action_dim: int, 
            action_scaling: float = None, 
            lr: float = 0.001, 
            k_epochs: int = 7, 
            policy_clip: float = 0.2, 
            GAE_lambda: float = 0.95,
            gamma: float = 0.995, 
            batch_size: int = 1024, 
            mini_batch_size: int = 64, 
            use_RND: bool = False, 
            beta: int = 0.001
        ):

        # Initializing the most important attributes of PPO.
        self.policy = ActorCritic(is_continuous, observ_dim, action_dim)
        self.policy_old = ActorCritic(is_continuous, observ_dim, action_dim)
        if use_RND:
            self.rnd = RND(in_features=observ_dim, out_features=observ_dim, beta=beta)

        # self.policy = t.compile(self.policy)
        # self.policy_old = t.compile(self.policy_old)
        # if use_RND:
        #     self.rnd = t.compile(self.rnd)

        self.memory = Memory()

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.policy.train()
        self.policy_old.eval()
        if use_RND:
            self.rnd.eval()

        self.loss_fn = nn.SmoothL1Loss() # loss function, SmoothL1Loss for tasks of regression
        # Optimizer AdamW for Actor&Critic
        self.optimizer = optim.AdamW(
            params = self.policy.parameters(),
            lr = lr
        )

        # Saving of collected hyperparameters, which u can get using PPO.your_hyperparameter,
        # it usefully, when you need get hyperparameters to Graphic class

        self.is_continuous = is_continuous

        self.action_scaling = action_scaling
        
        self.use_RND = use_RND
        self.beta = beta

        self.lr = lr
        
        self.policy_clip = policy_clip
        self.k_epochs = k_epochs
        self.GAE_lambda = GAE_lambda
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size

        self.observ_dim = observ_dim
        self.action_dim = action_dim

    @t.no_grad()
    def get_action(self, state: t.Tensor) -> np.ndarray:
        state = state.to(dtype=t.float32, device=device) # Transform numpy state to tensor state

        dist = self.policy_old.get_dist(state)

        action = dist.sample()
        # log_prob = dist.log_prob(action)

        if self.is_continuous:
            action = t.tanh(action).mul(self.action_scaling)
            # log_prob = log_prob.sum(-1)

        # state_value = self.policy_old.critic(features)

        return action.cpu().numpy()#, state_value).cpu().numpy(), log_prob.cpu().numpy()
    
    def batch_packer(self, values, batch_size: int):
        if isinstance(values, t.Tensor):
            batch = list(t.utils.data.DataLoader(values, batch_size))
        
        elif isinstance(values, list):
            batch = [list(t.utils.data.DataLoader(value, batch_size)) for value in values]

        return batch

    def compute_gae(self, rewards: np.ndarray, dones: np.ndarray, state_values: np.ndarray, next_value: np.ndarray):
        # Just computing of GAE.

        gae = 0
        returns = []
        for step in reversed(range(len(state_values))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - state_values[step]
            gae = delta + self.gamma * self.GAE_lambda * (1 - dones[step]) * gae
            
            returns.insert(0, gae + state_values[step])

            next_value = state_values[step]

        return returns

    def learn(self):
        if len(self.memory.states) < self.batch_size:
            return 

        # Copy data
        old_states = t.from_numpy(np.array(self.memory.states)).to(device, t.float32).detach()
        old_actions = t.from_numpy(np.array(self.memory.actions)).to(device, t.float32).detach()
        # old_state_values = t.from_numpy(np.array(self.memory.state_values)).to(device, t.float32).detach()
        # old_log_probs = t.from_numpy(np.array(self.memory.log_probs)).to(device, t.float32).detach()

        # Compute state values and log probabilities

        with t.no_grad():
            old_log_probs = []
            old_state_values = []
            
            for batch_old_states, batch_old_actions in zip(
                *self.batch_packer(
                    values = [
                        old_states,
                        old_actions
                    ],
                    batch_size = self.mini_batch_size
                )
            ):
                
                batch_log_probs, batch_state_values, _ = self.policy_old.get_evaluate(batch_old_states, batch_old_actions)

                old_log_probs.append(batch_log_probs)
                old_state_values.append(batch_state_values)
            
        old_log_probs = t.cat(old_log_probs, dim=0).detach()
        old_state_values = t.cat(old_state_values, dim=0).detach()
        
        # Compute rewards with or without intrinsic rewards

        if self.use_RND:
            rewards = (
                t.from_numpy(np.array(self.memory.rewards)).to(device, t.float32).add_(
                    self.rnd.compute_intrinsic_reward(
                        self.batch_packer(old_states, self.mini_batch_size)
                    )
                )
            ).detach().cpu().numpy()

            self.rnd.update_pred(
                self.batch_packer(old_states, self.mini_batch_size)
            )
        else:
            rewards = np.array(self.memory.rewards)
        dones = np.array(self.memory.dones)

        # Clear copied data
        self.memory.clear()

        # Computing GAE
        state_values = old_state_values.cpu().numpy()
        next_value = state_values[-1]

        returns = self.compute_gae(rewards, dones, state_values, next_value)
        returns = t.from_numpy(np.array(returns)).to(device, t.float32).detach()

        # Compute and normalise advantages
        advantages = returns - old_state_values

        # Break down data to batches
        batches = self.batch_packer(
            values = [
                old_states, 
                old_actions, 
                old_log_probs, 
                advantages, 
                returns
            ], 
            batch_size = self.mini_batch_size
        )

        # K_epochs cycle
        with tqdm(
            total = self.k_epochs * old_states.size(0), 
            leave=False
        ) as pbar:
            for _ in range(self.k_epochs):
                for batch_old_states, batch_old_actions, batch_old_log_probs, batch_advantages, batch_returns in zip(*batches):
                    # Collecting log probs, values of states, and dist entropy
                    batch_log_probs, batch_state_values, batch_entropy = self.policy.get_evaluate(batch_old_states, batch_old_actions)
                            
                    # calculating and clipping of log_probs, 'cause using of exp() function can will lead to inf or nan values
                    ratios = t.exp(t.clamp(batch_log_probs - batch_old_log_probs, min=-20, max=20))

                    surr1 = ratios * batch_advantages # calculating of surr1
                    surr2 = t.clamp(ratios, min=1 - self.policy_clip, max=1 + self.policy_clip) * batch_advantages  # clipping of ratios, where min is 1 - policy_clip, and max is 1 + policy_clip, next multiplying on advantages
                            
                    # gradient is loss of actor + 0.5 * loss of critic - 0.01 * dist_entropy. 0.01 is entropy bonus
                    loss = -t.min(surr1, surr2) + 0.5 * self.loss_fn(batch_state_values, batch_returns) - 0.01 * batch_entropy

                    self.optimizer.zero_grad()

                    loss.mean().backward() # using mean of loss to back propagation
                
                    # nn.utils.clip_grad_value_(self.policy.actor_parameters, 100) # cliping of actor parameters
                    # nn.utils.clip_grad_value_(self.policy.critic_parameters, 100) # cliping of critic parameters

                    nn.utils.clip_grad_norm_(self.policy.parameters(), 5)

                    self.optimizer.step()

                    pbar.update(batch_old_states.size(0))
                    pbar.set_description(f"Loss: {loss.mean().item(): .6f}")

        self.policy_old.load_state_dict(
            self.policy.state_dict()
        ) # load parameters of policy to policy_old

    def load_weights(self, path: str):
        try:
            self.policy.load_state_dict(t.load(path+'/Policy_weights.pth', weights_only=True))
            self.policy_old.load_state_dict(self.policy.state_dict())
            
            if self.use_RND:
                self.rnd.load_state_dict(t.load(path+'/RND_weights.pth', weights_only=True))
                
        except FileNotFoundError:
            pass
    
    def save_weights(self, path: str):
        t.save(self.policy.state_dict(), path+'/Policy_weights.pth')
        if self.use_RND:
            t.save(self.rnd.state_dict(), path+'/RND_weights.pth')
import torch as t
from torch import optim, nn

from ActorCritic import ActorCritic
from Memory import Memory

import utils

import numpy as np
from tqdm import tqdm

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class PPO:
    def __init__(
        self, 
        is_continuous: bool,
        observ_dim: int,
        action_dim: int,
        action_scaling: float = 1.0,
        lr: float = 0.0003,
        k_epochs: int = 7,
        policy_clip: float = 0.2,
        GAE_lambda: float = 0.95,
        gamma: float = 0.995,
        batch_size: int = 512,
        mini_batch_size: int = 64
    ):

        # Initializing the most important attributes of PPO.
        self.policy = ActorCritic(is_continuous, observ_dim, action_dim)
        self.policy_old = ActorCritic(is_continuous, observ_dim, action_dim)

        self.memory = Memory()

        self.policy_old.load_state_dict(
            self.policy.state_dict()
        )

        self.policy.train()
        self.policy_old.eval()

        self.loss_fn = nn.SmoothL1Loss() # loss function, SmoothL1Loss for tasks of regression
        self.optimizer = optim.AdamW(
            params = self.policy.parameters(), 
            lr = lr
        )

        # Saving of collected hyperparameters, which u can get using PPO.your_hyperparameter,
        # it usefully, when you need get hyperparameters to Graphic class

        self.is_continuous = is_continuous

        self.action_scaling = action_scaling

        self.lr = lr
        self.policy_clip = policy_clip
        self.k_epochs = k_epochs
        
        self.GAE_lambda = GAE_lambda
        self.gamma = gamma

        assert batch_size % mini_batch_size == 0
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size

        self.observ_dim = observ_dim
        self.action_dim = action_dim

    @t.no_grad()
    def get_action(self, state: t.Tensor) -> np.ndarray:
        state = state.to(device, t.float32) # Transfer state to device and data type

        dist = self.policy_old.get_dist(state)

        action = dist.sample()

        if self.is_continuous:
            action = t.tanh(action).mul(self.action_scaling)

        return action.cpu().numpy()

    def learn(self):
        if len(self.memory.states) < self.batch_size:
            return 

        # Copy data
        old_states = t.from_numpy(np.array(self.memory.states)).to(device, t.float32).detach()
        old_actions = t.from_numpy(np.array(self.memory.actions)).to(device, t.float32).detach()
        
        # Compute state values and log probabilities
        with t.no_grad():
            old_log_probs = []
            old_state_values = []
            
            for batch_old_states, batch_old_actions in zip(
                *utils.batch_packer(
                    values = [
                        old_states,
                        old_actions
                    ],
                    batch_size = self.mini_batch_size
                )
            ):

                batch_log_probs, batch_state_values, _ = self.policy_old.get_evaluate(
                    batch_old_states,
                    batch_old_actions
                )

                old_state_values.append(batch_state_values)
                old_log_probs.append(batch_log_probs)
            
        old_state_values = t.cat(old_state_values, dim=0).detach()
        old_log_probs = t.cat(old_log_probs, dim=0).detach()

        rewards = np.array(self.memory.rewards)
        dones = np.array(self.memory.dones)

        # Clear copied data
        self.memory.clear()

        # Computing GAE
        state_values = old_state_values.cpu().numpy()
        next_value = state_values[-1]

        returns = utils.compute_gae(
            rewards,
            dones,
            state_values,
            next_value,
            
            gamma=self.gamma,
            GAE_lambda=self.GAE_lambda
        )
        returns = t.from_numpy(
            np.array(returns)
        ).to(device, t.float32).detach()

        # Compute and normalize advantages
        advantages = returns - old_state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Normalizing advantages 

        # Break down data to batches
        batches = utils.batch_packer(
            values = [
                old_states, 
                old_actions,
                old_log_probs, 
                advantages,
                returns
            ],
            batch_size = self.mini_batch_size
        )

        # initialize progress bar
        pbar = tqdm(
            total = self.k_epochs * old_states.size(0),
            leave = False
        )

        n_accumulated_grad_batches = 0

        self.optimizer.zero_grad()

        # K_epochs cycle
        for _ in range(self.k_epochs):
            for batch_old_states, batch_old_actions, batch_old_log_probs, batch_advantages, batch_returns in zip(*batches):
                # Collect log probabilities, state values, and distribution entropy
                batch_log_probs, batch_state_values, batch_entropy = self.policy.get_evaluate(
                    batch_old_states,
                    batch_old_actions
                )
                        
                # calculating and clipping of log_probs, because using of exp() function might lead to inf or nan values
                ratios = t.exp(
                    t.clamp(
                        input = batch_log_probs - batch_old_log_probs,
                        min = -20,
                        max = 20
                    )
                )

                # calculating of surr1/surr2
                # clipping of ratios, where minimum is 1 - policy_clip, and maximum is 1 + policy_clip, 
                # next multiplying on advantages
                surr1 = t.mul(ratios, batch_advantages) # calculating of surr1
                surr2 = t.mul(
                    t.clamp(
                        input = ratios, 
                        min = 1 - self.policy_clip,
                        max = 1 + self.policy_clip
                    ),
                    batch_advantages
                )
                                                                    
                # gradient is loss of actor + 0.5 * loss of critic - 0.01 * dist_entropy.
                loss = -t.min(surr1, surr2).mean() + 0.5 * self.loss_fn(batch_state_values, batch_returns) - 0.01 * batch_entropy
                loss /= (self.batch_size // self.mini_batch_size) # using mean of loss for back propagation

                # self.optimizer.zero_grad()

                loss.backward() # using mean of loss for back propagation

                # self.optimizer.step()

                n_accumulated_grad_batches += 1

                if n_accumulated_grad_batches == (self.batch_size // self.mini_batch_size):
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5) # clipping of gradients, to avoid exploding gradients
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    n_accumulated_grad_batches = 0

                # Update progress bar
                pbar.update(batch_old_states.size(0))
                pbar.set_description(f"Loss: {loss.mean().item(): .6f}")
            
            if n_accumulated_grad_batches != 0:
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5) # clipping of gradients, to avoid exploding gradients
                    
                self.optimizer.step()
                self.optimizer.zero_grad()

                n_accumulated_grad_batches = 0

        # load parameters of policy to policy_old
        self.policy_old.load_state_dict(
            self.policy.state_dict()
        )

    def load_weights(self, path: str):
        try:
            self.policy.load_state_dict(
                t.load(path+'/Policy_weights.pth', weights_only=True)
            )
            self.policy_old.load_state_dict(
                self.policy.state_dict()
            )
            
            if self.use_RND:
                self.rnd.load_state_dict(
                    t.load(path+'/RND_weights.pth', weights_only=True)
                )
                
        except FileNotFoundError:
            pass
    
    def save_weights(self, path: str):
        t.save(self.policy.state_dict(), path+'/Policy_weights.pth')

        if self.use_RND:
            t.save(self.rnd.state_dict(), path+'/RND_weights.pth')
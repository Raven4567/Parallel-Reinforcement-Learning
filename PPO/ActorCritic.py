import torch as t
from torch import nn, distributions

import torch.nn.functional as F

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class ActorCritic(nn.Module):
    def __init__(self, is_continuous: bool, observ_dim: int, action_dim: int):
        super().__init__()

        self.is_continuous = is_continuous # discrete or continuous

        self.model = nn.Sequential(
            nn.Linear(observ_dim, 64, bias=False),
            nn.GroupNorm(64 // 8, 64),
            nn.SiLU(inplace=True),

        )

        if self.is_continuous:
            # mu_head for getting mean of actions
            self.mu_head = nn.Sequential(
                nn.Linear(64, 64, bias=False),
                nn.GroupNorm(64 // 8, 64),
                nn.SiLU(inplace=True),

                nn.Linear(64, action_dim)
            ) 
            # log_std for gettinf log of standard deviation which we predicting
            self.log_std_head = nn.Sequential(
                nn.Linear(64, 64, bias=False),
                nn.GroupNorm(64 // 8, 64),
                nn.SiLU(inplace=True),
                
                nn.Linear(64, action_dim)
            )

        else:
            self.actor = nn.Sequential(
                nn.Linear(64, 64, bias=False),
                nn.GroupNorm(64 // 8, 64),
                nn.SiLU(inplace=True),

                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            ) # Initialization of actor if you're using discrete PPO

        self.critic = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.GroupNorm(64 // 8, 64),
            nn.SiLU(inplace=True),

            nn.Linear(64, 1)
        ) # critic's initialization

        self.init_weights() # Initialization of weights of model
        
        self.to(device) # Send of model to GPU or CPU
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0, std=0.01)
                    
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state: t.Tensor):
        raise NotImplementedError

    def get_dist(self, state: t.Tensor):
        if self.is_continuous:
            features = self.model(state)

            mu = self.mu_head(features)
            std = F.softplus(
                t.clamp(
                    input = self.log_std_head(features), 
                    min = -2, 
                    max = 2
                )
            )

            # covariance matrix for multivariate normal distribution
            cov_matrix = t.diag_embed(
                t.square(std)
            )
            dist = distributions.MultivariateNormal(mu, cov_matrix)
        
        else:
            features = self.model(state)
            probs = self.actor(features)

            dist = distributions.Categorical(probs)

        return dist
    
    def get_state_value(self, state: t.Tensor):
        features = self.model(state)
        state_value = self.critic(features)

        return state_value.squeeze(-1)
    
    def get_evaluate(self, states: t.Tensor, actions: t.Tensor):
        features = self.model(states)

        if self.is_continuous:
            mu = self.mu_head(features)
            std = F.softplus(
                t.clamp(
                    input = self.log_std_head(features), 
                    min = -2, 
                    max = 2
                )
            )

            # covariance matrix for multivariate normal distribution
            cov_matrix = t.diag_embed(
                t.square(std)
            )
            dist = distributions.MultivariateNormal(mu, cov_matrix)
        
        else:
            probs = self.actor(features)
            dist = distributions.Categorical(probs)

        log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy().mean().detach()
        
        state_value = self.critic(features).squeeze(-1)

        return log_probs, state_value, dist_entropy
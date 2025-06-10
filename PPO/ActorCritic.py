import torch as t
from torch import nn, distributions

import torch.nn.functional as F

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class ActorCritic(nn.Module):
    def __init__(self, action_dim: int, observ_dim: int, is_continuous: bool, action_scaling: float):
        super().__init__()

        self.is_continuous = is_continuous # discrete or continuous

        if self.is_continuous:
            self.action_scaling = t.tensor(action_scaling, dtype=t.float32, device=device) # for scaling dist.sample() if you're using continuous PPO

            self.log_std = t.log(self.action_scaling / 4)
                
            self.Actor = nn.Sequential(
                nn.Linear(observ_dim, 64),
                nn.GroupNorm(64 // 8, 64),
                nn.SiLU(inplace=True),

                # nn.Linear(64, 64),
                # nn.GroupNorm(64 // 8, 64),
                # nn.SiLU(inplace=True),

            ) # Initialization of actor if you're using continuous PPO

            self.mu_head = nn.Linear(64, action_dim) # mu_head for getting mean of actions
            self.log_std_head = nn.Linear(64, action_dim) # log_std for gettinf log of standard deviation which we predicting

        else:
            self.Actor = nn.Sequential(
                nn.Linear(observ_dim, 64),
                nn.GroupNorm(64 // 8, 64),
                nn.SiLU(inplace=True),

                # nn.Linear(64, 64),
                # nn.GroupNorm(64 // 8, 64),
                # nn.SiLU(inplace=True),

                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            ) # Initialization of actor if you're using discrete PPO

        self.Critic = nn.Sequential(
            nn.Linear(observ_dim, 64),
            nn.GroupNorm(64 // 8, 64),
            nn.SiLU(inplace=True),

            # nn.Linear(64, 64),
            # nn.GroupNorm(64 // 8, 64),
            # nn.SiLU(inplace=True),

            nn.Linear(64, 1)
        ) # Critic's initialization

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
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
        
        # If out sequential model is split up, we unite our parameters of actor, mu_head, log_std else we just getting Actor.parameters()
        self.Actor_parameters = list(self.Actor.parameters()) + \
                                list(self.mu_head.parameters()) + \
                                list(self.log_std_head.parameters()) \
                                if is_continuous else list(self.Actor.parameters())
            
        self.Critic_parameters = list(self.Critic.parameters()) # Critic_parameters for discrete or continuous PPO
        
        self.to(device) # Send of model to GPU or CPU

    def forward(self, state: t.Tensor):
        raise NotImplementedError

    def get_dist(self, state: t.Tensor):
        if self.is_continuous:
            features = self.Actor(state)

            # mu = t.tanh(self.mu_head(features)) * self.action_scaling
            mu = self.mu_head(features)
            std = F.softplus(
                t.clamp(self.log_std_head(features), min=-self.log_std, max=self.log_std)
            )

            # std = t.exp(t.clamp(self.log_std_head(features), min=-self.log_std, max=self.log_std))
            # std = t.exp(self.log_std_head(features))

            dist = distributions.Normal(mu, std)
        
        else:
            probs = self.Actor(state)

            dist = distributions.Categorical(probs)

        return dist
    
    def get_state_value(self, state: t.Tensor):
        return self.Critic(state).squeeze(-1)
    
    def get_evaluate(self, states: t.Tensor, actions: t.Tensor):
        dist = self.get_dist(states)

        log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()

        if self.is_continuous:
            log_probs = log_probs.sum(-1)
            dist_entropy = dist_entropy.sum(-1)
        else:
            pass
        
        state_value = self.get_state_value(states)

        return log_probs, state_value, dist_entropy
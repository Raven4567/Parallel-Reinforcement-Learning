import torch as t
from torch import nn, optim

from copy import deepcopy

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class RND(nn.Module):
    """
    Module for an environment exploring by using intrinsic rewards, 
    and encouraging the agent to visit unknown states.
    """
    
    def __init__(self, in_features: int, out_features: int, beta: int = 0.001):
        """
        args:
            - `in_channels`: num of input channels, e. g. 1 for grayscale images, and 3 for RGB images.
            - `out_features`: num of output features for models.
            - `beta`: the weight for regularisation `RND`'s influence and don't let it dominate over external rewards
        """
        
        super().__init__()

        # Initialization of mobilenet_v4 mode
        self.model = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GroupNorm(64 // 8, 64),
            nn.SiLU(inplace=True),

            nn.Linear(64, out_features)
        )

        self.target_net = deepcopy(self.model)
        self.pred_net = deepcopy(self.model)

        del self.model

        self.init_weights()
        
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.beta = beta

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.AdamW(
            params = self.pred_net.parameters(), 
            lr=0.001
        )

        self.eval()
        
        self.to(device)
    
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
    
    @t.no_grad()
    def compute_intrinsic_reward(self, values: list[t.Tensor]) -> t.Tensor:
        """
        args:
            - `values`: list with tensor batches, can be got by using `PPO.batch_packer()`
        """

        target_batches = []
        pred_batches = []

        for i in values:
            with t.no_grad():
                targets = self.target_net(i)
                preds = self.pred_net(i)

            target_batches.append(targets)
            pred_batches.append(preds)
        
        target_batches = t.cat(target_batches, dim=0)
        pred_batches = t.cat(pred_batches, dim=0)

        rewards = t.norm(pred_batches - target_batches, dim=-1)

        return rewards * self.beta
    
    def update_pred(self, values: list[t.Tensor]) -> t.Tensor:
        """
        args:
            - `values`: list with tensor batches, can be got by using `PPO.batch_packer()`
        """

        self.pred_net.train()
        
        for i in values:
            with t.no_grad():
                targets = self.target_net(i)
            preds = self.pred_net(i)

            loss = self.loss_fn(preds, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.pred_net.eval()
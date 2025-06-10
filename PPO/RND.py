import torch as t
from torch import nn, optim

device = t.device('cuda' if t.cuda.is_available() else 'cpu')

class RND(nn.Module):
    def __init__(self, in_features: int, out_features: int, beta: int = 0.01, k_epochs: int = 3):
        super().__init__()

        self.target_net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GroupNorm(64 // 8, 64),
            nn.ReLU6(),

            nn.Linear(64, out_features)
        )

        self.pred_net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.GroupNorm(64 // 8, 64),
            nn.ReLU6(),

            nn.Linear(64, out_features)
        )
        
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.beta = beta
        self.k_epochs = k_epochs

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.pred_net.parameters(), lr=0.001)
        
        self.to(device)
    
    def compute_intristic_reward(self, values: list):
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

        intristic_rewards = t.norm(pred_batches - target_batches, dim=-1)

        self.update_pred(values)

        return intristic_rewards * self.beta
    
    def update_pred(self, values: list):
        self.pred_net.train()

        for _ in range(self.k_epochs):
            for i in values:
                with t.no_grad():
                    targets = self.target_net(i)
                preds = self.pred_net(i)

                loss = self.loss_fn(preds, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.pred_net.eval()
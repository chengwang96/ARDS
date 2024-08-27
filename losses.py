import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

if __name__ == "__main__":
    inputs = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.6, 0.4]], requires_grad=True)
    targets = torch.tensor([1, 0, 1])

    focal_loss = FocalLoss(alpha=1, gamma=2, reduction='mean')
    loss = focal_loss(inputs, targets)
    print(f"Focal Loss: {loss.item()}")

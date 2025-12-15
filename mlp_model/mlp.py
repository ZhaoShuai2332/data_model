import torch
import torch.nn as nn

"""
Definition of Multi-Layer Perceptron (MLP) for risk prediction

References:
    [1] Hinton, G. E. (1989). Connectionist learning procedures. Artificial intelligence, 40(1-3), 185-234.
    [2] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection.
        In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988).
"""
class RiskPredictionNet(nn.Module):
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(RiskPredictionNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            # if prev_dim == hidden_dim:
            #     residual = nn.Identity()
            #     layers.append(residual)
            
            # layers.append(nn.Linear(hidden_dim, hidden_dim))
            # layers.append(nn.BatchNorm1d(hidden_dim))
            # layers.append(nn.ReLU())
            # layers.append(nn.Dropout(dropout_rate * 0.5))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        # Use Sigmoid activation function to output probability
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance problems
    Reference: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

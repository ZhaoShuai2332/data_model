"""
ResNet1D: 1D Residual Network for Tabular Data Classification

This module implements a 1D variant of ResNet (Residual Network) for processing
tabular/sequential data for binary classification tasks.

References:
    [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). 
        "Deep Residual Learning for Image Recognition"
        CVPR 2016. https://arxiv.org/abs/1512.03385
    
    [2] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
        "Focal Loss for Dense Object Detection"
        ICCV 2017. https://arxiv.org/abs/1708.02002

Author: Data Modeling Project
"""

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with skip connection.
    
    Architecture: x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
    
    The skip connection allows gradients to flow directly through the network,
    enabling training of much deeper networks.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Convolution kernel size. Default: 3
        stride (int): Convolution stride. Default: 1
        downsample (nn.Module): Downsampling layer for identity mapping. Default: None
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        """
        Forward pass of the residual block.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, length)
            
        Returns:
            Tensor: Output tensor with residual connection applied
        """
        identity = x
        
        # First conv-bn-relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv-bn
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection with optional downsampling
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet1D(nn.Module):
    """
    1D ResNet for tabular data binary classification.
    
    This architecture adapts the ResNet design for 1D sequential/tabular data.
    Input features are treated as a 1D signal and processed through residual blocks.
    
    Architecture:
        Input → Conv1 → MaxPool → ResBlocks → AdaptiveAvgPool → Classifier → Output
    
    Args:
        input_dim (int): Number of input features
        num_classes (int): Number of output classes. Default: 1 (binary)
        layers (list): Number of residual blocks in each stage. Default: [2,2,2,2]
        channels (list): Number of channels in each stage. Default: [64,128,256,512]
        dropout_rate (float): Dropout probability. Default: 0.3
    
    Example:
        >>> model = ResNet1D(input_dim=40, num_classes=1)
        >>> x = torch.randn(32, 40)  # batch of 32 samples, 40 features
        >>> output = model(x)  # shape: (32, 1)
    """
    
    def __init__(self, input_dim, num_classes=1, 
                 layers=[2, 2, 2, 2], 
                 channels=[64, 128, 256, 512],
                 dropout_rate=0.3):
        super(ResNet1D, self).__init__()
        
        self.input_dim = input_dim
        
        # Initial convolution layer (stem)
        self.conv1 = nn.Conv1d(1, channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual stages
        self.layer1 = self._make_layer(channels[0], channels[0], layers[0])
        self.layer2 = self._make_layer(channels[0], channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], layers[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(channels[3], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """
        Construct a residual stage with multiple blocks.
        
        Args:
            in_channels (int): Input channels
            out_channels (int): Output channels
            blocks (int): Number of residual blocks
            stride (int): Stride for first block. Default: 1
            
        Returns:
            nn.Sequential: Sequential container of residual blocks
        """
        downsample = None
        
        # Downsample if dimensions don't match
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        # First block may need downsampling
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride, downsample=downsample))
        
        # Subsequent blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        Initialize network weights using He initialization.
        
        - Conv layers: Kaiming normal initialization
        - BatchNorm: weight=1, bias=0
        - Linear: Normal(0, 0.01)
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass of ResNet1D.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tensor: Output probabilities of shape (batch_size, num_classes)
        """
        # Reshape: (batch_size, input_dim) → (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and flatten
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in classification tasks.
    
    Focal Loss down-weights easy examples and focuses training on hard negatives.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Reference:
        Lin, T. Y., et al. (2017). "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002
    
    Args:
        alpha (float): Weighting factor for positive class. Default: 0.25
        gamma (float): Focusing parameter. Default: 2.0
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs (Tensor): Predicted probabilities
            targets (Tensor): Ground truth labels
            
        Returns:
            Tensor: Scalar focal loss value
        """
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()
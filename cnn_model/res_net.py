import torch.nn as nn
import torch


class ResidualBlock1D(nn.Module):
    """
    1D残差块
    
    结构: x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        
        # 第一层卷积
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # 第二层卷积
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        # 第一层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 第二层
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out
    
class ResNet1D(nn.Module):
    """
    1D ResNet用于表格数据分类
    
    架构:
    Input → Conv1 → ResBlock × N → ResBlock × N → ... → AvgPool → FC → Output
    """
    
    def __init__(self, input_dim, num_classes=1, 
                 layers=[2, 2, 2, 2], 
                 channels=[64, 128, 256, 512],
                 dropout_rate=0.3):
        super(ResNet1D, self).__init__()
        
        self.input_dim = input_dim
        
        # 初始卷积层
        self.conv1 = nn.Conv1d(1, channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(channels[0], channels[0], layers[0])
        self.layer2 = self._make_layer(channels[0], channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], layers[3], stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # 分类器
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
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """构建残差层"""
        downsample = None
        
        # 如果维度不匹配，需要downsample
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        # 第一个block可能需要downsample
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride, downsample=downsample))
        
        # 后续blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化权重"""
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
        # x: (batch_size, input_dim)
        # 转换为 (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 全局池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 分类
        x = self.classifier(x)
        
        return x


class FocalLoss(nn.Module):
    """Focal Loss"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()
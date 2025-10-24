import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # If input and output channels are different, use a 1x1 conv to match dimensions
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class ResidualCNN(nn.Module):
    def __init__(self, input_length):
        super(ResidualCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=201, padding=100)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        # Add a residual block with 16 channels
        self.resblock1 = ResidualBlock(16, 16, kernel_size=15, padding=7)
        # Add a residual block with 16->4 channels
        self.resblock2 = ResidualBlock(16, 4, kernel_size=15, padding=7)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        # x: (batch, 1, length)
        x = self.conv1(x)
        x = self.bn1(x)  # Apply batch normalization after the first convolution
        x = self.relu1(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.pool(x)  # (batch, 4, 1)
        x = x.view(x.size(0), -1)  # (batch, 4)
        x = self.fc(x)  # (batch, 1)
        return x.squeeze(1)  # (batch,)

class SimpleCNN(nn.Module):
    def __init__(self, input_length, input_channels=1, hidden_channels=[64, 128, 256], kernel_sizes=[3, 3, 3], dropout_rate=0.2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=201, padding=100)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 4, kernel_size=201, padding=101)
        self.bn2 = nn.BatchNorm1d(4)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(4, 1) 
        self.dropout = nn.Dropout(0.1)
        self.ln1 = nn.LayerNorm([16, input_length])
        self.ln2 = nn.LayerNorm([4, input_length+2])
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.ln2(x)
        x = self.pool(x)
        #x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)
    
class BaseCNN(nn.Module):
    def __init__(self, input_length):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=14, padding=7,bias=False)
        # Use BatchNorm1d for normalization after the first convolution
        self.bn1 = nn.BatchNorm1d(16)
        self.ln1 = nn.LayerNorm([16, input_length+1])
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 4, kernel_size=14, padding=7,bias=False)
        # Use BatchNorm1d for normalization after the second convolution
        self.bn2 = nn.BatchNorm1d(4)
        self.ln2 = nn.LayerNorm([4, input_length+1])
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        # x: (batch, 1, length)
        x = self.conv1(x)
        x = self.bn1(x)  # Apply batch normalization after the first convolution
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)  # Apply batch normalization after the second convolution
        x = self.relu2(x)
        x = self.pool(x)  # (batch, 8, 1)
        x = x.view(x.size(0), -1)  # (batch, 8)
        x = self.fc(x)  # (batch, 1)
        return x.squeeze(1)  # (batch,)

class ProposedCNN(nn.Module):
    """Lightweight 1D CNN for regression.

    Expected input shape: (batch, 1, length)
    Design goals:
      - Early downsampling for translation robustness
      - Small kernels (3) with one wider stem (7) to capture context
      - Global Average Pooling + single Linear head
    """
    def __init__(self, input_length: int, y_min: float = 0.0, y_max: float = 0.2):
        super(ProposedCNN, self).__init__()
        # Output scaling parameters
        self.y_min = y_min
        self.y_max = y_max
        # Stem: widen receptive field early with stride-2
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # Block 1: maintain then downsample
        self.block1 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            #nn.LayerNorm([32, 1250]),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
        )

        # Block 2: expand channels, modest dilation to grow receptive field
        self.block2 = nn.Sequential(
            nn.Conv1d(48, 64, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(64, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, 1, length)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        #x = nn.Dropout(0.1)(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.head(x)
        # Constrain to [0,1] then scale to [y_min, y_max]
        #x = self.act(x)
        #x = self.y_min + (self.y_max - self.y_min) * x
        return x.squeeze(1)

class VGG11_1D(nn.Module):
    """VGG-11 style network adapted to 1D signals for regression.

    Uses small kernels (3) and MaxPool for downsampling. Ends with GAP + Linear(1).
    Expected input: (batch, 1, length)
    """
    def __init__(self, input_length: int):
        super(VGG11_1D, self).__init__()
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_block(1, 32),
            nn.MaxPool1d(kernel_size=2, stride=2),

            conv_block(32, 64),
            nn.MaxPool1d(kernel_size=2, stride=2),

            conv_block(64, 128),
            conv_block(128, 128),
            nn.MaxPool1d(kernel_size=2, stride=2),

            conv_block(128, 256),
            conv_block(256, 256),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.head(x)
        return x.squeeze(1)

class AlexNet1D(nn.Module):
    """AlexNet style network adapted to 1D signals for regression.

    Uses a wider first conv and staged downsampling. Ends with GAP + Linear(1).
    Expected input: (batch, 1, length)
    """
    def __init__(self, input_length: int):
        super(AlexNet1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 48, kernel_size=11, stride=4, padding=5, bias=False),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Conv1d(48, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),

            nn.Conv1d(128, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Conv1d(192, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Conv1d(192, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.head(x)
        return x.squeeze(1)
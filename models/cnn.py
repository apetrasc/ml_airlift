import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_length):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=30, padding=15)
        # English comment: Use BatchNorm1d for normalization after the first convolution
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 4, kernel_size=30, padding=15)
        # English comment: Use BatchNorm1d for normalization after the second convolution
        self.bn2 = nn.BatchNorm1d(4)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        # x: (batch, 1, length)
        x = self.conv1(x)
        x = self.bn1(x)  # Apply batch normalization after the first convolution
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)  # Apply batch normalization after the second convolution
        x = self.relu2(x)
        x = self.pool(x)  # (batch, 8, 1)
        x = x.view(x.size(0), -1)  # (batch, 8)
        x = self.fc(x)  # (batch, 1)
        return x.squeeze(1)  # (batch,)
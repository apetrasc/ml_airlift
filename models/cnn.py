import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_length):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=201, padding=100)
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
        x = self.bn2(x)  # Apply batch normalization after the second convolution
        x = self.relu2(x)
        x = self.ln2(x)
        x = self.pool(x)
        #x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)

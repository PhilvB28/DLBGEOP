import torch.nn as nn

class CNN1DRegressor(nn.Module):
    def __init__(self, input_channels=6, sequence_length=60):
        super(CNN1DRegressor, self).__init__()
        # Convolutional layers with batch normalization
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Fully connected layers with dropout
        self.fc1 = nn.Sequential(
            nn.LazyLinear(256),
        )
        self.fc2 = nn.Linear(256, 1)  # Final layer for regression

    def forward(self, x):
        x = self.conv1(x)
        # x = self.pool(x)
        # x = self.conv2(x)
        # x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc1(x)
        x = self.fc2(x)
        return x.squeeze(1)  # Remove the extra dimension for regression

class EnhancedCNN1DRegressor(nn.Module):
    def __init__(self, input_channels=6, sequence_length=60):
        super(EnhancedCNN1DRegressor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(64, 256),
            nn.Dropout(0.5)  # No activation
        )
        self.fc2 = nn.Linear(256, 1)  # Final regression layer without activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x.squeeze(1)
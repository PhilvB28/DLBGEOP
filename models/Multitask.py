#TODO under construction
import torch
import torch.nn as nn

import torch.nn as nn


class CNN1DMultitask(nn.Module):
    def __init__(self, input_channels=6, sequence_length=60):
        super(CNN1DMultitask, self).__init__()

        # Convolutional layers with batch normalization
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        # Shared fully connected layer
        self.shared_fc = nn.Sequential(
            nn.LazyLinear(256),  # Shared representation layer
            nn.Dropout(0.5)  # Dropout for regularization
        )

        # Task-specific fully connected layers (Output Heads)
        self.fc_general = nn.Sequential(
            nn.Linear(256, 557),
            nn.Sigmoid()  # Output in [0, 1]
        )
        self.fc_deletion = nn.Sequential(
            nn.Linear(256, 536),
            nn.Sigmoid()  # Output in [0, 1]
        )
        self.fc_1bp_deletion = nn.Sequential(
            nn.Linear(256, 3),
            nn.Sigmoid()  # Output in [0, 1]
        )
        self.fc_1bp_insertion = nn.Sequential(
            nn.Linear(256, 4),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, x):
        # Pass through convolutional layers
        x = self.conv1(x)
        x = x.view(x.size(0), -1)  # Flatten the output

        # Shared fully connected representation
        x_shared = self.shared_fc(x)

        # Task-specific outputs
        out_general = self.fc_general(x_shared)
        out_deletion = self.fc_deletion(x_shared)
        out_1bp_deletion = self.fc_1bp_deletion(x_shared)
        out_1bp_insertion = self.fc_1bp_insertion(x_shared)

        return out_general, out_deletion, out_1bp_deletion, out_1bp_insertion
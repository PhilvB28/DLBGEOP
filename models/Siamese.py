#TODO under construction
import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self, sequence_length=60):
        super(SiameseNetwork, self).__init__()

        # Shared 1D convolutional layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Calculate the reduced sequence length after pooling
        reduced_length = sequence_length
        for _ in range(2):  # Two pooling layers
            reduced_length = reduced_length // 2

        # Shared fully connected layer
        self.shared_fc = nn.Linear(64 * reduced_length, 256)

        # Task-specific fully connected layers (Output Heads)
        self.fc_general = nn.Linear(256, 557)
        self.fc_deletion = nn.Linear(256, 536)
        self.fc_1bp_deletion = nn.Linear(256, 3)
        self.fc_1bp_insertion = nn.Linear(256, 4)

        self.relu = nn.ReLU()

    def forward_once(self, x):
        # Pass through shared CNN layers
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.shared_fc(x))
        return x

    def forward(self, input1, input2):
        # Process each input through the shared network
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # Calculate distance (Euclidean, cosine, or other)
        distance = torch.abs(output1 - output2)

        # Predict using the task-specific heads
        out_general = torch.sigmoid(self.fc_general(distance))
        out_deletion = torch.sigmoid(self.fc_deletion(distance))
        out_1bp_deletion = torch.sigmoid(self.fc_1bp_deletion(distance))
        out_1bp_insertion = torch.sigmoid(self.fc_1bp_insertion(distance))

        return {
            'general': out_general,
            'deletion': out_deletion,
            '1bp_deletion': out_1bp_deletion,
            '1bp_insertion': out_1bp_insertion
        }
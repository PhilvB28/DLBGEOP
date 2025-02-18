import torch
import torch.nn as nn
import torch.nn.functional as F


# Capsule Layer (unchanged)
class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, in_dim, out_dim, num_routing=3):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_routing = num_routing

        # Weight matrix for all output capsules
        self.W = nn.Parameter(torch.randn(1, num_capsules, in_dim, out_dim))

    def forward(self, x):
        batch_size = x.size(0)
        num_primary_caps = x.size(1)  # e.g., 32
        # x: [batch_size, num_primary_caps, in_dim] where in_dim = 8

        # Expand x for proper multiplication:
        # [B, num_primary_caps, 1, in_dim]
        x = x.unsqueeze(2)

        # Repeat the weight matrix for each sample in the batch:
        # self.W: [1, num_capsules, in_dim, out_dim] where num_capsules=6 and out_dim=16
        W = self.W.repeat(batch_size, 1, 1, 1)  # Now: [batch_size, num_capsules, in_dim, out_dim]
        # Unsqueeze to add a dimension for primary capsules:
        W = W.unsqueeze(1)  # [batch_size, 1, num_capsules, in_dim, out_dim]

        # Prepare x for multiplication:
        # Add an extra dimension so that x becomes a column vector:
        x = x.unsqueeze(-1)  # [batch_size, num_primary_caps, 1, in_dim, 1]
        # We need to swap the last two dims so that the 8-dim capsule becomes [1, 8]:
        x = x.transpose(-2, -1)  # [batch_size, num_primary_caps, 1, 1, in_dim]

        # Now multiply: [1, in_dim] @ [in_dim, out_dim] yields [1, out_dim]
        u_hat = torch.matmul(x, W)  # [batch_size, num_primary_caps, num_capsules, 1, out_dim]
        u_hat = u_hat.squeeze(-2)  # [batch_size, num_primary_caps, num_capsules, out_dim]

        # Initialize routing logits:
        b_ij = torch.zeros(batch_size, num_primary_caps, self.num_capsules, device=x.device)

        # Dynamic routing:
        for routing_iteration in range(self.num_routing):
            c_ij = F.softmax(b_ij, dim=2)  # coupling coefficients
            s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=1)
            v_j = self.squash(s_j)
            if routing_iteration < self.num_routing - 1:
                b_ij = b_ij + (u_hat * v_j.unsqueeze(1)).sum(dim=-1)

        return v_j  # [batch_size, num_capsules, out_dim]

    @staticmethod
    def squash(x):
        squared_norm = (x ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm) / torch.sqrt(squared_norm + 1e-8)
        return scale * x


# Length Layer (unchanged)
class Length(nn.Module):
    def forward(self, x):
        # Compute the Euclidean norm along the capsule dimension
        return torch.sqrt((x ** 2).sum(dim=-1))


# Capsule-based Classifier for 6 labels
class CapsNetRegressor_6(nn.Module):
    def __init__(self, input_channels=6, sequence_length=60):
        super(CapsNetRegressor_6, self).__init__()

        # Convolutional feature extractor
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # After two pooling layers:
        # - sequence_length goes from 60 -> 30 -> 15.
        # We'll collapse the 15-length dimension to 1 in the primary capsule layer.
        # Primary Capsule Layer: Create 32 capsules, each of dimension 8.
        # To do this, we set the out_channels to 32*8 = 256 and use a kernel that spans the entire 15-length.
        self.primary_caps = nn.Conv1d(64, 256, kernel_size=15, stride=15, padding=0)

        # Digit Capsule Layer: Route from the primary capsules to 6 output capsules.
        # Here, each capsule will have an output dimension (vector length) of 16.
        self.digit_caps = CapsuleLayer(num_capsules=6, in_dim=8, out_dim=16, num_routing=3)

        # The Length layer converts capsule vectors into scalar probabilities.
        self.length = Length()

    def forward(self, x):
        # x shape: [batch_size, 6, 60]
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # -> [batch_size, 32, 30]
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # -> [batch_size, 64, 15]

        # Primary Capsules:
        x = self.primary_caps(x)  # -> [batch_size, 256, 1]
        x = x.squeeze(-1)  # -> [batch_size, 256]
        # Reshape to get capsules: 256 channels rearranged into 32 capsules of 8 dimensions each.
        x = x.view(x.size(0), 32, 8)  # -> [batch_size, 32, 8]

        # Dynamic Routing to get 6 output capsules:
        x = self.digit_caps(x)  # -> [batch_size, 6, 16]

        # Get capsule lengths (used as probabilities)
        x = self.length(x)  # -> [batch_size, 6]
        x = torch.sigmoid(x)  # Sigmoid activation for multi-label probabilities

        return x
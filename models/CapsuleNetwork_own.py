import torch
import torch.nn as nn
import torch.nn.functional as F

# Capsule Layer
class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, in_dim, out_dim, num_routing=3):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_routing = num_routing

        # Capsule weight matrix
        self.W = nn.Parameter(torch.randn(1, num_capsules, in_dim, out_dim))

    def forward(self, x):
        batch_size = x.size(0)

        # Unsqueeze and tile input
        x = x.unsqueeze(2)  # Shape: [batch_size, num_features, 1, in_dim]
        W = self.W.repeat(batch_size, 1, 1, 1)  # Shape: [batch_size, num_capsules, in_dim, out_dim]

        # Compute "predicted outputs" (u_hat)
        u_hat = torch.matmul(x, W)  # Shape: [batch_size, num_features, num_capsules, out_dim]
        #print("Shape of u_hat:", u_hat.shape)

        b_ij = torch.zeros(batch_size, u_hat.size(1), self.num_capsules,device=x.device)  # Shape: [batch_size, num_features, num_capsules]

        # Routing mechanism
        for routing_iteration in range(self.num_routing):
            # Softmax over the capsule dimension (dim=2)
            c_ij = F.softmax(b_ij, dim=2)  # Shape: [batch_size, num_features, num_capsules]
            #print(f"Shape of c_ij (routing iteration {routing_iteration}):", c_ij.shape)

            # Weighted sum of u_hat (routing weights * predicted outputs)
            s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=1)  # Sum over features, shape: [batch_size, num_capsules, out_dim]
            #print(f"Shape of s_j (routing iteration {routing_iteration}):", s_j.shape)

            # Apply squashing
            v_j = self.squash(s_j)  # Shape: [batch_size, num_capsules, out_dim]
            #print(f"Shape of v_j (routing iteration {routing_iteration}):", v_j.shape)

            if routing_iteration < self.num_routing - 1:
                # Update b_ij (routing logits)
                b_ij = b_ij + (u_hat * v_j.unsqueeze(1)).sum(dim=-1)  # Shape: [batch_size, num_features, num_capsules]

        return v_j  # Final output shape: [batch_size, num_capsules, out_dim]

    @staticmethod
    def squash(x):
        "Squashing function for capsule outputs."
        squared_norm = (x ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm) / torch.sqrt(squared_norm + 1e-8)
        return scale * x

# Length Layer for scalar output
class Length(nn.Module):
    def forward(self, x):
        return torch.sqrt((x ** 2).sum(dim=-1))

# Capsule-based Regressor
class CapsNetRegressor(nn.Module):
    def __init__(self, input_channels=6, sequence_length=60):
        super(CapsNetRegressor, self).__init__()

        # Convolutional feature extractor
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Primary Capsule Layer
        self.primary_caps = nn.Conv1d(64, 557, kernel_size=3, stride=2, padding=1)

        # Capsule Layer
        self.digit_caps = CapsuleLayer(num_capsules=557, in_dim=8, out_dim=32, num_routing=3) #out_dim was 16 before, 32 showed better results with MSE

        # Length Layer for scalar regression output
        self.length = Length()

    def forward(self, x):
        #print("Input Size:", x.shape)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        #print("After Conv1:", x.shape)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        #print("After Conv2:", x.shape)

        x = self.primary_caps(x)  # Shape: [batch_size, num_features, capsule_dim]
        #print("After Primary Capsules:", x.shape)

        #x = self.primary_reshape(x)
        #print("Shape after primary_reshape:", x.shape)
        #print("--------------------")
        x = self.digit_caps(x)  # Shape: [batch_size, num_capsules, capsule_dim]
        #print("After Digit Capsules:", x.shape)
        #print("--------------------")
        x = self.length(x)  # Shape: [batch_size, num_capsules, capsule_dim]
        #print("After length function:", x.shape)
        x = torch.sigmoid(x)  # Multi-label probabilities for 557 outcomes
        #print("After Sigmoid function:", x.shape)

        return x

"""
ALEC - TGCN Model Definition
Temporal-Spatial Graph Convolutional Network with Learnable Adjacency Matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ALEC_TGCN(nn.Module):
    """
    Appliance-Level Energy Consumption Temporal Graph Convolutional Network.

    Combines:
      - Learnable adjacency matrix (captures spatial dependencies between appliances)
      - Graph convolution (propagates features across appliance nodes)
      - GRU (captures temporal patterns across the sequence)
      - Fully connected output (predicts next-step consumption for each appliance)

    Args:
        num_appliances: Number of appliance nodes (input feature size).
        hidden_dim: Hidden dimension for GRU and graph conv layers.
    """

    def __init__(self, num_appliances: int, hidden_dim: int):
        super(ALEC_TGCN, self).__init__()

        self.num_appliances = num_appliances
        self.hidden_dim = hidden_dim

        # Learnable adjacency matrix
        self.A = nn.Parameter(torch.randn(num_appliances, num_appliances))

        # Graph convolution weight
        self.gc_weight = nn.Linear(num_appliances, hidden_dim)

        # Temporal modeling
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.2)

        # Output layer
        self.fc = nn.Linear(hidden_dim, num_appliances)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, num_appliances)

        Returns:
            Predicted consumption tensor of shape (batch, num_appliances)
        """

        # Graph Convolution: apply normalized adjacency matrix
        A_norm = F.softmax(self.A, dim=1)
        x_gc = torch.matmul(x, A_norm)
        x_gc = self.gc_weight(x_gc)

        # GRU temporal modeling
        out, _ = self.gru(x_gc)

        # Take last timestep output
        out = out[:, -1, :]

        # Apply dropout
        out = self.dropout(out)

        # Prediction
        out = self.fc(out)

        return out
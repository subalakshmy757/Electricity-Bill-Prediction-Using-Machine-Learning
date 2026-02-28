import torch
import torch.nn as nn
import torch.nn.functional as F

class ALEC_TGCN(nn.Module):
    def __init__(self, num_appliances, hidden_dim):
        super(ALEC_TGCN, self).__init__()

        self.num_appliances = num_appliances

        #  Learnable adjacency matrix
        self.A = nn.Parameter(torch.randn(num_appliances, num_appliances))

        # Graph convolution weight
        self.gc_weight = nn.Linear(num_appliances, hidden_dim)

        # Temporal modeling
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Output layer
        self.fc = nn.Linear(hidden_dim, num_appliances)

    def forward(self, x):
        """
        x shape: (batch, seq_len, appliances)
        """

        # Graph Convolution
        A_norm = F.softmax(self.A, dim=1)
        x_gc = torch.matmul(x, A_norm)
        x_gc = self.gc_weight(x_gc)

        # GRU
        out, _ = self.gru(x_gc)

        # Last timestep
        out = out[:, -1, :]

        # Prediction
        out = self.fc(out)

        return out
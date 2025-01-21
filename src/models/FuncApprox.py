
import torch
import torch.nn as nn



class FuncApprox(nn.Module):
    "Neural Network to approximate reasonably well-behaved function."
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.2):
        super(FuncApprox, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Input layer: (x, y, t)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)  # Output layer: u(x, y, t)
        )

    def forward(self, x):
        return self.network(x)
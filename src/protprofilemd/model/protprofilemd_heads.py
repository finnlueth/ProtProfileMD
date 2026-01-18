import torch
from torch import nn


class ProfileHeadLinear(nn.Module):
    """Linear head for profile generation from T5 embeddings."""

    def __init__(
        self,
        dropout: float,
        hidden_size: int = 1024,
        num_classes: int = 20,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.classifier(x)
        x = torch.softmax(x, dim=2)
        return x


class ProfileHeadConv1D(nn.Module):
    """Conv1D head for profile generation from T5 embeddings.
    Based on https://github.com/hefeda/PGP/blob/master/prott5_batch_predictor.py#L144.
    """

    def __init__(
        self,
        dropout: float,
        hidden_size: int = 1024,
        num_classes: int = 20,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv1d(hidden_size, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, num_classes, kernel_size=7, padding=3),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.classifier(x)
        x = x.transpose(1, 2)
        x = torch.softmax(x, dim=2)
        return x

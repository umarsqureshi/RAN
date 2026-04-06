"""RAN model architectures: Generator and Critic."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RANGenerator(nn.Module):
    """Generator Architecture: 
    depth x [Dense -> LeakyReLU -> BatchNorm -> Dropout(0.2)]
                  -> Dense(1) -> 1 + log(softplus(x) + 1/e)
    Output activation: 1 + log(softplus(x) + 1/e).
    """
    def __init__(self, input_dim, width=50, depth=3, dropout=0.2):
        super().__init__()
        layers = []
        in_features = input_dim
        for i in range(depth):
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.BatchNorm1d(width, momentum=0.01, eps=0.001))
            layers.append(nn.Dropout(dropout))
            in_features = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_features, 1)

    def forward(self, x):
        h = self.backbone(x)
        raw = self.head(h)
        # 1 + log(softplus(x) + 1/e):
        #   min = 1+log(1/e) = 0  (at x->-inf, approached from above)
        #   x=0: ~1.06,  x=5: ~2.68,  x=100: ~5.62
        #   gradient = sigmoid(x)/(softplus(x)+1/e) > 0 always
        inv_e = 0.36787944117  # 1/e
        sp = F.softplus(raw)
        w = 1.0 + torch.log(sp + inv_e)
        return w


class RANCritic(nn.Module):
    """Critic Architecture: 
    depth x [SpectralNorm(Dense) -> LeakyReLU -> Dropout -> LayerNorm]
                  -> SpectralNorm(Dense(1))  (linear output)
    """
    def __init__(self, input_dim, width=50, depth=3, dropout=0.2):
        super().__init__()
        layers = []
        in_features = input_dim
        for i in range(depth):
            linear = nn.utils.spectral_norm(nn.Linear(in_features, width))
            layers.append(linear)
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.LayerNorm(width))
            in_features = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.utils.spectral_norm(nn.Linear(in_features, 1))

    def forward(self, x):
        # The outputs of the critic are clipped
        # to eliminate nonphysical values"
        return torch.clamp(self.head(self.backbone(x)), -10.0, 10.0)

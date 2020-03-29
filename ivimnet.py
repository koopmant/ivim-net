import torch
import torch.nn as nn


class IVIMNet(nn.Module):
    def __init__(self, b_values):
        super().__init__()

        self.b_values = b_values
        self.fc_layers = nn.ModuleList()
        for i in range(3):  # 3 fully connected hidden layers
            self.fc_layers.extend([nn.Linear(len(b_values),
                                             len(b_values)), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers,
                                     nn.Linear(len(b_values), 4))

    def forward(self, x):
        params = torch.abs(self.encoder(x))

        dp = params[:, 0].unsqueeze(1)
        dt = params[:, 1].unsqueeze(1)
        a = params[:, 2].unsqueeze(1)
        b = params[:, 3].unsqueeze(1)

        c = a + b
        fp = a / c  # (1 - fp) = b / c

        x_fit = c * (fp * torch.exp(-self.b_values * dp)
                     + (b / c)*torch.exp(-self.b_values * dt))

        return x_fit, dp, dt, fp, c

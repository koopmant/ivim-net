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
        params = torch.sigmoid(self.encoder(x))

        dt_min = 0
        dt_max = 0.005
        f_min = 0.0
        f_max = 0.7
        dp_min = 0.005
        dp_max = 0.5
        c_min = 0.8
        c_max = 1.2

        dp = dp_min + params[:, 0].unsqueeze(1) * (dp_max - dp_min)
        dt = dt_min + params[:, 1].unsqueeze(1) * (dt_max - dt_min)
        fp = f_min + params[:, 2].unsqueeze(1) * (f_max - f_min)
        c = c_min + params[:, 3].unsqueeze(1) * (c_max - c_min)

        x_fit = c * (fp * torch.exp(-self.b_values * dp)
                     + (torch.tensor(1) - fp)*torch.exp(-self.b_values * dt))

        return x_fit, dp, dt, fp, c

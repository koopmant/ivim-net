from datetime import datetime
import torch
import ivimnet
from pathlib import Path


class Config:
    def __init__(self):
        # Path to data
        self.path_data = Path("path/to/data/file.mat")

        # Path to save directory
        dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_dir = self.path_data.parent / f"training_{dt_string}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Neural net
        self.net = ivimnet.IVIMNetSigm

        # Optimizer
        self.optim = torch.optim.Adam
        self.learning_rate = 0.005

        #
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.batch_size = 128
        self.max_it = 1024
        self.patience = 4

        self.split = True
        self.split_ratio = 0.8

        self.batch_size_val = 8192
        self.max_it_val = 512

        self.save_estimates = True

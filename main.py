from hyperparameters import Config
import numpy as np
import scipy.io
import torch
import torch.backends.cudnn
from tqdm import tqdm
from torch.utils.data import DataLoader
import shutil
import matplotlib.pyplot as plt


class Trainer(Config):
    def __init__(self):
        super().__init__()

        self.b_values = None
        self.data_set = None
        self.data_s0 = None
        self.best_model = None

        # set CUDA if available
        self.device = self.set_cuda()

    @staticmethod
    def set_cuda():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        return device

    def run(self):
        # Copy configuration file to output directory
        shutil.copy("hyperparameters.py", str(self.save_dir))
        shutil.copy("ivimnet.py", str(self.save_dir))

        # Load data
        self.load_data()

        # Initialize network and optimizer
        self.net = self.net(self.b_values).to(self.device)
        self.optim = self.optim(self.net.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        # Move criterion to device
        self.criterion = self.criterion.to(self.device)

        # Train
        self.train()

        if self.save_estimates:
            # Save parameter estimates
            self.eval(self.best_model)

    def load_data(self):
        mat = scipy.io.loadmat(str(self.path_data))

        b = np.asarray(mat.get("bvec"), np.float32).squeeze()
        s = np.asarray(mat.get("data"), np.float32).squeeze()
        s = np.transpose(s)

        # Normalize on b == 0
        self.data_s0 = np.mean(s[:, b == 0], 1).reshape(-1,1)
        s = s / self.data_s0

        # Exclude b == 0
        s = s[:, b != 0]
        b = b[b != 0]

        self.data_set = torch.from_numpy(s)
        self.b_values = torch.from_numpy(b).to(self.device)

    def train(self):
        # Data loader
        data_loader = DataLoader(self.data_set,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 drop_last=True)

        # Initialize
        best_loss = 1e16
        num_bad_epochs = 0
        loss_train = []

        for epoch in range(2):
            print("-----------------------------------------------------------------")
            print(f"Epoch: {epoch}; Bad epochs: {num_bad_epochs}")

            # Run one epoch
            loss = self.iterate(data_loader, self.max_it, train=True)

            # show loss
            print(f"Loss: {loss}")

            # save loss history for plot
            loss_train.append(loss)
            # plot loss history
            self.plot(loss_train)

            # early stopping
            if loss < best_loss:
                print("############### Saving good model ########################")
                self.best_model = self.net.state_dict()
                best_loss = loss
                num_bad_epochs = 0
            else:
                num_bad_epochs = num_bad_epochs + 1
                if num_bad_epochs == self.patience:
                    break

        # Save best model
        torch.save(self.best_model, self.save_dir / "final_model.pt")
        print(f"Done, best loss: {best_loss}")

    def iterate(self, data_loader, max_it, train=True):
        if train:
            self.net.train()
        else:
            self.net.eval()

        total_it = np.min([max_it, np.floor(len(data_loader.dataset) // data_loader.batch_size)])
        total_loss = 0.

        for i, batch in enumerate(tqdm(data_loader, position=0, leave=True, total=total_it)):
            # Zero the parameter gradients
            self.optim.zero_grad()
            # Put batch on GPU if present
            batch = batch.to(self.device)
            # Forward
            fit = self.net(batch)[0]
            # Determine loss for batch
            loss = self.criterion(fit, batch)
            # Total loss
            total_loss += loss.item()

            if train:
                # Backward + optimize
                loss.backward()
                self.optim.step()

            if i >= total_it:
                break

        avg_loss = total_loss / total_it
        return avg_loss

    def plot(self, loss_train):
        plt.clf()
        plt.plot(loss_train)
        plt.yscale("log")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        # plt.show()
        plt.savefig(self.save_dir / "loss_train.png")

    def eval(self, model):
        # Load model
        self.net.load_state_dict(model)

        print("Evaluate network...")
        # Evaluate on data
        self.net.eval()
        with torch.no_grad():
            dp_pred, dt_pred, fp_pred, c_pred = self.net(self.data_set)[1:]
        print("Finished.")

        s0_pred = c_pred.numpy() * self.data_s0

        # save results
        fr = {"Dp": dp_pred.numpy(), "Dt": dt_pred.numpy(),
              "fp": fp_pred.numpy(), "s0": s0_pred}

        print("Save results")
        scipy.io.savemat(self.save_dir / "parameter_estimates.mat", fr, do_compression=True)
        print("Done.")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()

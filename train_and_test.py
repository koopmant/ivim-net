from pathlib import Path
from utils import read_yaml
import ivimnet
import numpy as np
import scipy.io
import torch.cuda
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.backends.cudnn
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
from shutil import copyfile


arg = read_yaml(Path("config.yaml"))


nets = {
    'ivim_net': ivimnet.IVIMNet,
    'ivim_net_abs': ivimnet.IVIMNetAbs,
    'ivim_net_sigm': ivimnet.IVIMNetSigm,
}
optims = {
    'adam': torch.optim.Adam
}

# load CUDA for PyTorch, if available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# load data
path_data = Path(arg['dir'], arg['datafile'])
mat = scipy.io.loadmat(path_data)

# b values
b_values = torch.from_numpy(np.array(mat.get("bvec")).astype(np.float32).squeeze())
b_values_no0 = b_values[1:].to(device)

# data
x = np.transpose(mat.get("data")).astype(np.float32)
x_norm = x[:, 1:]/x[:, 0:1]  # normalize on x at b=0

# %%

for ii in range(3, 40):

    dt_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_file_str = arg['net'] + f'{ii:02}' + "_" + dt_string

    copyfile("config.yaml", str(Path(arg['dir'], save_file_str + "_config.yaml")))

    # Network
    net = nets[arg['net']](b_values_no0).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = optims[arg['optimizer']](net.parameters(), lr=arg['learning_rate'], weight_decay=1e-4)

    # num_batches = len(x_norm) // arg['batch_size_train']
    data_set = TensorDataset(torch.from_numpy(x_norm.astype(np.float32)))
    data_loader = DataLoader(data_set,
                             batch_size=arg['batch_size_train'],
                             shuffle=True,
                             num_workers=0,
                             drop_last=True)

    # %% Train
    
    # Best loss
    best = 1e16
    num_bad_epochs = 0
    final_model = None
    loss_train = []
    # get_ipython().run_line_magic('matplotlib', 'inline')

    if not arg['split']:
        for epoch in range(1000):
            print("-----------------------------------------------------------------")
            print(f"Epoch: {epoch}; Bad epochs: {num_bad_epochs}")
            net.train()
            running_loss_train = 0.

            for i, x_batch in enumerate(tqdm(data_loader)):
                # zero the parameter gradients
                optimizer.zero_grad()
                # put batch on GPU if present
                x_batch = x_batch[0].to(device)

                # forward + backward + optimize
                x_pred = net(x_batch)[0]
                loss = criterion(x_pred, x_batch)
                loss.backward()
                optimizer.step()
                running_loss_train += loss.item()

            # save loss history for plot
            loss_train.append(running_loss_train)
            # show loss
            print(f"Loss: {running_loss_train}")
            if epoch > 0:
                # plot loss history
                plt.clf()
                plt.plot(loss_train)
                plt.yscale("log")
                plt.xlabel('epoch')
                plt.ylabel('loss')
                # plt.show()
                plt.savefig(Path(arg['dir'], save_file_str + "_plot.png"))
            # early stopping
            if running_loss_train < best:
                print("############### Saving good model ########################")
                final_model = net.state_dict()
                best = running_loss_train
                num_bad_epochs = 0
            else:
                num_bad_epochs = num_bad_epochs + 1
                if num_bad_epochs == arg['patience']:
                    torch.save(final_model,
                               Path(arg['dir'], save_file_str + "_model.pt"))
                    print(f"Done, best loss: {best}")
                    break

        # # plot loss history
        # plt.clf()
        # plt.plot(loss_train)
        # plt.yscale("log")
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # # plt.show()
        # plt.savefig(Path(arg['dir'], save_file_str + "_plot.png"))
    else:
        split = int(np.floor(len(x_norm) * arg['split_ratio']))
        train_set, val_set = torch.utils.data.random_split(data_set, [split, len(x_norm)-split])
        train_loader = DataLoader(train_set,
                                  batch_size=arg['batch_size_train'],
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=True)

        val_loader = DataLoader(val_set,
                                batch_size=arg['batch_size_val'],
                                shuffle=True,
                                num_workers=0,
                                drop_last=True)

        # defining the number of training and validation batches for normalisation later
        total_it_train = np.min([arg['max_it_train'], np.floor(split // arg['batch_size_train'])])
        total_it_val = np.min([arg['max_it_val'], np.floor(len(val_set) // arg['batch_size_val'])])
        batch_norm_train = total_it_train
        batch_norm_val = total_it_val

        # Initialising parameters
        loss_val = []
        # Train
        for epoch in range(1000):
            print("-----------------------------------------------------------------")
            print(f"Epoch: {epoch}; Bad epochs: {num_bad_epochs}")
            # initialising and resetting parameters
            net.train()
            running_loss_train = 0.
            running_loss_val = 0.

            for i, x_batch in enumerate(tqdm(train_loader, position=0, leave=True, total=total_it_train)):
                # zero the parameter gradients
                optimizer.zero_grad()
                # put batch on GPU if present
                x_batch = x_batch[0].to(device)
                # forward + backward + optimize
                x_pred = net(x_batch)[0]
                # determine loss for batch
                loss = criterion(x_pred, x_batch)
                # updating network
                loss.backward()
                optimizer.step()
                # total loss
                running_loss_train += loss.item()
                if i >= total_it_train:
                    break
            # after training, do validation in unseen data without updating gradients
            net.eval()

            for i, x_batch in enumerate(tqdm(val_loader, position=0, leave=True, total=total_it_val), 0):
                optimizer.zero_grad()
                x_batch = x_batch[0].to(device)
                x_pred = net(x_batch)[0]
                loss = criterion(x_pred, x_batch)
                running_loss_val += loss.item()
                if i >= total_it_val:
                    break
            # scale losses
            running_loss_train = running_loss_train / batch_norm_train
            running_loss_val = running_loss_val / batch_norm_val
            # save loss history for plot
            loss_train.append(running_loss_train)
            loss_val.append(running_loss_val)
            # show loss
            print(f"Loss: {running_loss_train}, validation_loss: {running_loss_val}")
            if epoch > 0:
                # plot loss history
                plt.clf()
                plt.plot(loss_train)
                plt.plot(loss_val)
                plt.yscale("log")
                plt.xlabel('epoch')
                plt.ylabel('loss')
                # plt.show()
                plt.savefig(Path(arg['dir'], save_file_str + "_plot.png"))
            # early stopping criteria
            if running_loss_val < best:
                print("############### Saving good model ###############################")
                final_model = net.state_dict()
                best = running_loss_val
                num_bad_epochs = 0
            else:
                # if loss not better, than add "bad epoch" and don't save network
                num_bad_epochs = num_bad_epochs + 1
                if num_bad_epochs == arg['patience']:
                    torch.save(final_model,
                               Path(arg['dir'], save_file_str + "_model.pt"))
                    print(f"Done, best loss: {best}")
                    break

        # # plot loss history
        # plt.clf()
        # plt.plot(loss_train)
        # plt.plot(loss_val)
        # plt.yscale("log")
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # # plt.show()
        # plt.savefig(Path(arg['dir'], save_file_str + "_plot.png"))

    print("Done")

    # apply net to data

    # Restore best model
    net.load_state_dict(final_model)

    # evaluate on data
    net.eval()
    with torch.no_grad():
        dp_pred, dt_pred, fp_pred, c_pred = net(
                torch.from_numpy(x_norm.astype(np.float32)))[1:]
    
    s0_pred = c_pred.numpy() * x[:, 0:1]
    
    # save results
    fr = {"Dp": dp_pred.numpy(), "Dt": dt_pred.numpy(),
          "fp": fp_pred.numpy(), "s0": s0_pred}
    
    scipy.io.savemat(Path(arg['dir'], save_file_str + "_fr.mat"),
                     fr, do_compression=True)

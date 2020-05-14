import numpy as np
import scipy.io
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
import datetime
import ivimnet
from utils import read_yaml


settings = read_yaml(Path("config.yaml"))

models = {
    'ivim_net': ivimnet.IVIMNet,
    'ivim_net_abs': ivimnet.IVIMNetAbs,
    'ivim_net_sigm': ivimnet.IVIMNetSigm,
}

# data
filepath_data = Path(
        settings['dir'], "concatenated_signals_for_deeplearning.mat")
mat = scipy.io.loadmat(filepath_data)

# b values
b_values = torch.from_numpy(np.array(mat.get("bvec")).astype(np.float32).squeeze())
b_values_no0 = b_values[1:]

# data
x = np.transpose(mat.get("data")).astype(np.float32)
x_norm = x[:, 1:]/x[:, 0:1]

# %%

for ii in range(1):

    dt_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Network
    net = models[settings['model']](b_values_no0)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    batch_size = 128
    num_batches = len(x_norm) // batch_size
    trainloader = utils.DataLoader(utils.TensorDataset(torch.from_numpy(x_norm.astype(np.float32))),
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=0,
                                   drop_last=True)
    
    # %% Train
    
    # Best loss
    best = 1e16
    num_bad_epochs = 0
    patience = 10
    final_model = None
    
    for epoch in range(1000):
        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        net.train()
        running_loss = 0.
    
        for i, x_batch in enumerate(tqdm(trainloader), 0):
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            x_pred = net(x_batch[0])[0]
            loss = criterion(x_pred, x_batch[0])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
        print("Loss: {}".format(running_loss))
        # early stopping
        if running_loss < best:
            print("############### Saving good model ########################")
            final_model = net.state_dict()
            best = running_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == patience:
                torch.save(final_model,
                           Path(settings['dir'],
                                "model_" + modelname + f'{ii:02}' + "_" + dt_string + ".pt"))
                print("Done, best loss: {}".format(best))
                break
    
    print("Done")

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
    
    scipy.io.savemat(Path(settings['dir'],
                          "fr_" + modelname + f'{ii:02}' + "_" + dt_string + ".mat"),
                     fr, do_compression=True)

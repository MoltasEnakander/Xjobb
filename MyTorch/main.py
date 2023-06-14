# this code works 'well' for training a network that should test the real cylinder

import os
import argparse
from time import time
from tqdm import tqdm
import utils
import numpy as np
import torch
from Networks import ISARNet
from custom_loss import CustomLoss
from CustomDataset import CustomDataset
from generate_rcs import generate_rcs_sparse, generate_multiple_rcs_sparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

def main_program():
    parser = argparse.ArgumentParser('Args')
    parser.add_argument('--epochs', '-e', type=int, default=6)
    parser.add_argument('--lambd', '-l', type=float, default=4700.0)
    parser.add_argument('--mu', '-mu', type=float, default=5.0)
    parser.add_argument('--batch_size', '-bs', type=int, default=1)
    parser.add_argument('--dataset', '-d', type=str, default="largedataset/")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define network
    network = ISARNet().to(device)
    
    optimiser = torch.optim.SGD(network.parameters(), lr=5e-9, momentum=0.1, nesterov=True)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=7, gamma=0.1)

    criterion = CustomLoss()

    # scene properties
    scene_width = 256
    scene_height = 256

    # rcs calculation properties
    cal_range = 7.5
    ff = 0
    f_start = 8
    f_stop = 12
    nf = 256
    nphi = 256
    nbins = 2048 # nr of positions in the RCS to calculate

    # isar image properties
    xmax = 1
    xmin = -xmax
    nx = scene_width
    ymax = 1
    ymin = -ymax
    ny = scene_height
    x_range = torch.linspace(xmin,xmax,nx).to(device)
    y_range = torch.linspace(ymin,ymax,ny).to(device)
    hanning_flag = 0
    elev_angle = 0

    # frequency and angular resolution
    f = torch.linspace(f_start,f_stop, nf).to(device)
    B = f_stop - f_start
    fc = (f_start + f_stop)/2
    # Resolution in y is c/(2B). Match x-resolution which is c/(2*fc*sin(phi_tot))
    phi_tot = np.arcsin(B/fc)*180/np.pi; # convert to degrees
    phi = torch.linspace(-phi_tot/2, phi_tot/2, nphi).to(device)

    # params
    epochs = args.epochs  # nr of training iterations
    mu = args.mu       # threshold in loss function
    lambd = args.lambd  # regularisation term

    # training
    network.train()

    from torch.utils.data import DataLoader, random_split

    try:
        for i in range(2):
            if i == 0: # first train with subset of dataset
                rcs_path = os.path.join(args.dataset, 'rcs1/')
                isar_path = os.path.join(args.dataset, 'isars1/')
                batch_size = args.batch_size

                dataset1 = CustomDataset(rcs_path, isar_path)

                test_abs = int(len(dataset1) * 0.85)
                train_subset, val_subset = random_split(
                    dataset1, [test_abs, len(dataset1) - test_abs])
                
            else: # train with entire dataset
                # load best model from first training
                #network = utils.load_model_state(network, "best_model.pth", AiQu=False)

                rcs_path = os.path.join(args.dataset, 'rcs2/')
                isar_path = os.path.join(args.dataset, 'isars2/')

                dataset2 = CustomDataset(rcs_path, isar_path)

                test_abs = int(len(dataset2) * 0.85)
                train_subset2, val_subset2 = random_split(
                    dataset2, [test_abs, len(dataset2) - test_abs])

                train_subset = torch.utils.data.ConcatDataset([train_subset, train_subset2])
                val_subset = torch.utils.data.ConcatDataset([val_subset, val_subset2])

            trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
            valloader = DataLoader(val_subset, batch_size=batch_size, shuffle=True, num_workers=0)

            best_val_loss = np.inf
            l0_losses = []
            l2_losses = []
            train_losses = []
            val_losses = []
            for epoch in tqdm(range(epochs)):
                l0_losss = 0
                l2_losss = 0
                epoch_loss = 0
                network.train()
                for batch_idx, (rcs, isar) in enumerate(tqdm(trainloader)):
                    torch.autograd.set_detect_anomaly(True)
                    torch.cuda.empty_cache()
                    
                    original_rcs = rcs.to(device).detach()   # rcs generated from the scene
                    low_res = isar.to(device)  # isar image generated from the rcs

                    scale_factor = torch.max(torch.abs(low_res))

                    low_res = low_res.view(batch_size, 1, low_res.shape[-2], low_res.shape[-1])         

                    high_res = network(low_res/scale_factor)
                    high_res = high_res.view(batch_size, high_res.shape[-2], high_res.shape[-1])
                    low_res = low_res.squeeze(dim=1)

                    bins = torch.randperm(nf * nphi)
                    bins = bins[0:nbins]
                    # convert to freqs and angles
                    freqs = bins // nphi
                    angles = bins % nphi

                    # create new rcs measurement of the high resolution image for certain frequencies and angles
                    new_rcs = generate_multiple_rcs_sparse(high_res, cal_range, ff, f[freqs], phi[angles], x_range, y_range, nbins, device)                        

                    # mask and normalise original rcs
                    original_rcs_masked = original_rcs[:, freqs, angles] / scale_factor        

                    norm = original_rcs.shape[-2] * original_rcs.shape[-1] / nbins

                    optimiser.zero_grad()
                    pseudo_l0_loss, l2_loss = criterion(high_res, new_rcs, original_rcs_masked, mu)

                    loss = l2_loss * norm + pseudo_l0_loss * lambd

                    l0_losss += (pseudo_l0_loss * lambd).detach().cpu().numpy()
                    l2_losss += (l2_loss * norm).detach().cpu().numpy()

                    loss.backward()
                    optimiser.step()
                    epoch_loss += loss.cpu().detach().numpy()
                
                train_losses.append(epoch_loss)
                l0_losses.append(l0_losss)
                l2_losses.append(l2_losss)
        #########################################################################################################################################################
        #########################################################################################################################################################
                network.eval()
                val_loss = 0
                for batch_idx, (rcs, isar) in enumerate(tqdm(valloader)):             
                    with torch.no_grad():
                        original_rcs = rcs.to(device).detach()   # rcs generated from the scene
                        low_res = isar.to(device)  # isar image generated from the rcs

                        low_res = low_res.view(batch_size, 1, low_res.shape[-2], low_res.shape[-1])         

                        high_res = network(low_res/scale_factor)
                        high_res = high_res.view(batch_size, high_res.shape[-2], high_res.shape[-1])
                        low_res = low_res.squeeze(dim=1)
                        
                        bins = torch.randperm(nf * nphi)
                        bins = bins[0:nbins]
                        # convert to freqs and angles
                        freqs = bins // nphi
                        angles = bins % nphi

                        new_rcs = generate_multiple_rcs_sparse(high_res, cal_range, ff, f[freqs], phi[angles], x_range, y_range, nbins, device)

                        # mask and normalise original rcs
                        original_rcs_masked = original_rcs[:, freqs, angles] / scale_factor

                        high_res = high_res.view(batch_size, 1, high_res.shape[-2], high_res.shape[-1])

                        norm = original_rcs.shape[-2] * original_rcs.shape[-1] / nbins
                        
                        pseudo_l0_loss, l2_loss = criterion(high_res, new_rcs, original_rcs_masked, mu)

                        loss = l2_loss * norm + pseudo_l0_loss * lambd
                        val_loss += loss.cpu().detach().numpy()

                val_losses.append(val_loss)                
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    utils.save_model(network, "best_model_2.pth", AiQu=False)
                fname = "aa_lambda4700_mu5_" + str(i) + "_" + str(epoch) + "_cl_2.pth"
                utils.save_model(network, fname, AiQu=False)

            fname2 = "losses22" + str(i) + ".txt"
            with open(os.path.join('trained_models/', fname2), 'w') as file:
                for item in l0_losses:
                    file.write(str(item) + " ")
                file.write('\n')
                file.write('\n')
                for item in l2_losses:
                    file.write(str(item) + " ")
                file.write('\n')
                file.write('\n')
                for item in train_losses:
                    file.write(str(item) + " ")
                file.write('\n')
                file.write('\n')
                for item in val_losses:
                    file.write(str(item) + " ")

            
        #########################################################################################################################################################
        #########################################################################################################################################################

    except Exception as e:
        fname = "error.txt"
        with open(os.path.join('trained_models/', fname), 'w') as file:
            file.write(str(e))


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == '__main__':
    # set_seed()
    main_program()
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
from generate_rcs import generate_rcs_sparse, generate_multiple_rcs_sparse, generate_rcs
import torch.nn.functional as F
import scipy

def main_program():
    parser = argparse.ArgumentParser('Args')
    parser.add_argument('--epochs', '-e', type=int, default=25)
    parser.add_argument('--lambd', '-l', type=float, default=8000.0)
    parser.add_argument('--mu', '-mu', type=float, default=5.0)
    parser.add_argument('--batch_size', '-bs', type=int, default=1)
    parser.add_argument('--l1', '-l1', action='store_true', default=False)
    parser.add_argument('--dataset', '-d', type=str, default="largedataset/")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define network
    network = ISARNet().to(device)

    #optimiser = torch.optim.Adam(network.parameters(), lr=1e-5)
    optimiser = torch.optim.SGD(network.parameters(), lr=5e-12, momentum=0.1, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=7, gamma=0.1)

    criterion = CustomLoss()       

    # mat = scipy.io.loadmat("meas_data/cylinder32mm0001_VV_gated.mat")
    # #mat = scipy.io.loadmat("meas_data/bil_bmw_hh_gated.mat")
    # # mat = scipy.io.loadmat("meas_data/bil_bmw_overlay.mat")

    # cal_range = mat['calrange'][0][0]
    # cal_range = cal_range.item()

    # rcs = mat['rcs_all']

    # theta_start = mat['thetastart'][0][0]
    # theta_stop = mat['thetastop'][0][0]
    # ntheta = mat['ntheta'][0][0]
    # fstart = mat['fstart'][0][0]
    # fstop = mat['fstop'][0][0]
    # nf = mat['nf'][0][0]

    # f = np.linspace(fstart, fstop, nf)
    # phi = np.linspace(theta_start, theta_stop, ntheta)
    # phi = phi[842:961]
    # f = f[26:82]
    # original_rcs = rcs[26:82, 842:961]

    # f_mat = matlab.double(f.tolist())
    # phi_mat = matlab.double(phi.tolist())
    # rcs_mat = matlab.double(original_rcs.tolist(), is_complex=True)

    cal_range = 7.61
    theta_start = -180.14
    theta_stop = 179.66000000000003
    ntheta = 1800
    fstart = 5.997
    fstop = 16.000500000000002
    nf = 135

    f = np.linspace(fstart, fstop, nf)
    phi = np.linspace(theta_start, theta_stop, ntheta)
    phi = phi[842:961] # 118
    f = f[26:82] # 55
    
    ff = 0
    nf = len(f)
    nphi = len(phi)
    nbins = 2048 # nr of positions in the RCS to calculate

    # isar image properties
    xmax = 1
    xmin = -xmax
    nx = 256
    ymax = 1
    ymin = -ymax
    ny = 256
    x_range = torch.linspace(xmin,xmax,nx).to(device)
    y_range = torch.linspace(ymin,ymax,ny).to(device)
    # x_range_mat = matlab.double(x_range.tolist()) # create matlab object of x, both will be needed later on
    # y_range_mat = matlab.double(y_range.tolist()) # same but for y
    hanning_flag = 0
    elev_angle = 0

    # low_res = eng.generateISAR(rcs_mat, f_mat, phi_mat, hanning_flag, elev_angle, cal_range, ff, x_range_mat, y_range_mat)
    # low_res = np.asarray(low_res) # matlab -> numpy
    # low_res = np.transpose(low_res)

    # load rcs and isar
    original_rcs = torch.load("meas_data/rcs/rcs_cylinder.pt")
    low_res = torch.load("meas_data/isars/isar_cylinder.pt")

    original_rcs = original_rcs.to(device)
    low_res = low_res.to(device)

    scale_factor = torch.max(torch.abs(low_res))

    # convert from numpy to torch
    # low_res = torch.from_numpy(low_res).to(device).type(torch.complex64)
    phi = torch.from_numpy(phi).to(device)
    f = torch.from_numpy(f).to(device)
    # original_rcs = torch.from_numpy(original_rcs).to(device)

    # params
    epochs = args.epochs  # nr of training iterations
    mu = args.mu       # threshold in loss function
    lambd = args.lambd  # regularisation term    

    # training
    network.train()
    
    try:        
        batch_size = 1
        # losses = []
        # l0_losses = []
        # l2_losses = []
        for epoch in tqdm(range(epochs)):
            network.train()
            for i in tqdm(range(2500)):
                torch.autograd.set_detect_anomaly(True)
                torch.cuda.empty_cache()

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

                # mask original rcs
                original_rcs_masked = original_rcs[freqs, angles] / scale_factor     

                norm = original_rcs.shape[-2] * original_rcs.shape[-1] / nbins * 10.11 # 32 /// 3,168

                optimiser.zero_grad()
                pseudo_l0_loss, l2_loss = criterion(high_res, new_rcs, original_rcs_masked, mu)
                loss = l2_loss * norm + pseudo_l0_loss * lambd

                #print("Loss: ", loss.detach().cpu().numpy())
                # print("Max: ", torch.max(torch.abs(low_res[0])).detach().cpu().numpy())
                # print("Min: ", torch.min(torch.abs(low_res[0])).detach().cpu().numpy())
                #print("Max: ", torch.max(torch.abs(high_res[0])).detach().cpu().numpy())
                # print("Min: ", torch.min(torch.abs(high_res[0])).detach().cpu().numpy())

                # l0_losses.append((pseudo_l0_loss * lambd).detach().cpu().numpy())
                # l2_losses.append((l2_loss*norm).detach().cpu().numpy())
                # losses.append(loss.cpu().detach().numpy())           

                loss.backward()
                optimiser.step()
            
    #########################################################################################################################################################
    #########################################################################################################################################################
            
            scheduler.step()    

            fname = "lambda2000_mu5_" + "_" + str(epoch) + "_real.pth"
            utils.save_model(network, fname, AiQu=False)
            scheduler.step()                           
            
        #########################################################################################################################################################
        #########################################################################################################################################################

    except Exception as e:
        print(str(e))
        # fname = "error.txt"
        # with open(os.path.join('real_model/', fname), 'x') as file:
        #     file.write(str(e))

if __name__ == '__main__':
    main_program()
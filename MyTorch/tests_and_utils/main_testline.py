import os
import argparse
from time import time
from tqdm import tqdm
import utils
import numpy as np
import torch
from Networks import ISARNet
from custom_loss import CustomLoss
from generate_rcs import generate_rcs_sparse, generate_multiple_rcs_sparse, generate_rcs
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matlab.engine
from pynput.keyboard import Key, Listener
from generate_scene import generate_line, generate_single_point_scene

show_plots = False

def press(key):
    global show_plots    
    if key == Key.ctrl_r:        
        show_plots = True
        return

def release(key):
    global show_plots
    if key == Key.ctrl_r:
        show_plots = False

def main_program():
    parser = argparse.ArgumentParser('Args')
    parser.add_argument('--epochs', '-e', type=int, default=25)
    parser.add_argument('--lambd', '-l', type=float, default=8000.0)
    parser.add_argument('--mu', '-mu', type=float, default=5)    
    parser.add_argument('--batch_size', '-bs', type=int, default=1)
    parser.add_argument('--l1', '-l1', action='store_true', default=False)
    args = parser.parse_args()
    eng = matlab.engine.start_matlab()
    eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner")
    eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner\Moltas")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define network
    network = ISARNet().to(device)
    
    optimiser = torch.optim.SGD(network.parameters(), lr=5e-11, momentum=0.1, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=7, gamma=0.1)

    criterion = CustomLoss()

    scene_width = 256
    scene_height = 256    

    # echo calculation properties
    cal_range = 7.5
    ff = 0
    f_start = 8
    f_stop = 12
    nf = 256
    nphi = 256

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

    # some type conversions (matlab does not like python objects)
    f_mat = matlab.double(f.tolist())
    phi_mat = matlab.double(phi.tolist())
    x_range_mat = matlab.double(x_range.tolist()) # create matlab object of x, both will be needed later on
    y_range_mat = matlab.double(y_range.tolist()) # same but for y

    scene = generate_line(scene_height, scene_width)
    scene = torch.from_numpy(scene).to(device).type(torch.complex64)

    original_rcs = generate_rcs(scene, cal_range, ff, f, phi, x_range, y_range, device, method=1)    
    rcs_mat = matlab.double(original_rcs.detach().cpu().numpy().tolist(), is_complex=True)

    low_res = eng.generateISAR(rcs_mat, f_mat, phi_mat, hanning_flag, elev_angle, cal_range, ff, x_range_mat, y_range_mat)
    low_res = np.asarray(low_res) # matlab -> numpy
    low_res = np.transpose(low_res)

    # convert from numpy to torch
    low_res = torch.from_numpy(low_res).to(device).type(torch.complex64)

    scale_factor = torch.max(torch.abs(low_res))

    #phi = torch.from_numpy(phi).to(device)
    #f = torch.from_numpy(f).to(device)
    #original_rcs = torch.from_numpy(original_rcs).to(device)

    # params
    epochs = args.epochs  # nr of training iterations
    mu = args.mu       # threshold in loss function
    lambd = args.lambd  # regularisation term    
    nbins = 2048
    #ka = 10

    # training
    network.train()

    with Listener(on_press=press, on_release=release) as listener: # listen for keyboard input from user
        try:        
            batch_size = 1
            losses = []
            l0_losses = []
            l2_losses = []            
            for epoch in tqdm(range(epochs)):
                network.train()
                for i in range(2500):
                    torch.autograd.set_detect_anomaly(True)
                    torch.cuda.empty_cache()

                    low_res = low_res.view(batch_size, 1, low_res.shape[-2], low_res.shape[-1])      

                    high_res = network(low_res/scale_factor)
                    high_res = high_res.view(batch_size, high_res.shape[-2], high_res.shape[-1])
                    high_res = high_res
                    low_res = low_res.squeeze(dim=1)

                    bins = torch.randperm(nf * nphi)
                    bins = bins[0:nbins]
                    # convert to freqs and angles
                    freqs = bins // nphi
                    angles = bins % nphi

                    new_rcs = generate_multiple_rcs_sparse(high_res, cal_range, ff, f[freqs], phi[angles], x_range, y_range, nbins, device)

                    # mask original rcs
                    original_rcs_masked = original_rcs[freqs, angles]                

                    norm = original_rcs.shape[-2] * original_rcs.shape[-1] / nbins

                    optimiser.zero_grad()
                    pseudo_l0_loss, l2_loss = criterion(high_res, new_rcs*scale_factor, original_rcs_masked, mu)
                    loss = l2_loss * norm / scale_factor + pseudo_l0_loss * lambd

                    print("Loss: ", loss.detach().cpu().numpy())
                    print("Max: ", torch.max(torch.abs(high_res[0])).detach().cpu().numpy())

                    l0_losses.append((pseudo_l0_loss * lambd).detach().cpu().numpy())
                    l2_losses.append((l2_loss*norm).detach().cpu().numpy())                    
                    losses.append(loss.cpu().detach().numpy())           

                    loss.backward()
                    optimiser.step()

                    if show_plots:
                        plt.figure("Low-res isar 2D")
                        fig = utils.plotcut_dB_in(low_res[0].cpu(), x_range.cpu(), y_range.cpu())

                        plt.figure("High-res isar 2D")
                        fig = utils.plotcut_dB_in(high_res[0].detach().cpu().numpy(), x_range.cpu(), y_range.cpu())

                        plt.figure("Losses")
                        plt.plot(l0_losses[-100:], label="Pseudo-L0")
                        plt.plot(l2_losses[-100:], label="L2")                        
                        plt.plot(losses[-100:], label="Losses")
                        plt.legend(loc="upper left")

                        plt.figure("Losses all")
                        plt.plot(l0_losses, label="Pseudo-L0")
                        plt.plot(l2_losses, label="L2")                        
                        plt.plot(losses, label="Losses")
                        plt.legend(loc="upper left")

                        plt.figure("Losses averages")
                        plt.plot(l0_losses, label="Pseudo-L0")
                        plt.plot(l2_losses, label="L2")                        
                        plt.plot(losses, label="Losses")
                        plt.legend(loc="upper left")

                        plt.show()
                
        #########################################################################################################################################################
        #########################################################################################################################################################
                
                scheduler.step()                
            
        #########################################################################################################################################################
        #########################################################################################################################################################

        except Exception as e:
            print(str(e))
            # fname = "error.txt"
            # with open(os.path.join('real_model/', fname), 'x') as file:
            #     file.write(str(e))
    listener.stop()

if __name__ == '__main__':
    main_program()
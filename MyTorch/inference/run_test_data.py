import sys
sys.path.insert(1, r"C:\Users\Lovisa Nilsson\Desktop\xjobb\MyTorch")

import os
from time import time
from tqdm import tqdm
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch
from Networks import ISARNet
from custom_loss import CustomLoss
from CustomDataset2 import CustomDataset
from pynput.keyboard import Key, Listener
from generate_rcs import generate_rcs, generate_rcs_sparse
import torch.nn.functional as F
from generate_scene import generate_point_scene_same_col, generate_single_point_scene, generate_line, generate_scene
import matlab.engine
from SSIM import ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

network = ISARNet().to(device)

network = utils.load_model_state(network, "best_model_2.pth", AiQu=False)

scene_width = 256
scene_height = 256
xmax = 1
xmin = -xmax
nx = scene_width
ymax = 1
ymin = -ymax
ny = scene_height
x_range = torch.linspace(xmin,xmax,nx).to(device)
y_range = torch.linspace(ymin,ymax,ny).to(device)

scene_width = 256
scene_height = 256
max_shapes = 4  # maximum nr of objects that are allowed to be in the random scene

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

#N = 10 # nr of scenes to create

from torch.utils.data import DataLoader
scene_path = os.path.join('test_data/', 'scenes/')
rcs_path = os.path.join('test_data/', 'rcs/')
isar_path = os.path.join('test_data/', 'isars/')
batch_size = 1

dataset = CustomDataset(scene_path, rcs_path, isar_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

for batch_idx, (scene, rcs, isar) in enumerate((dataloader)):
    torch.cuda.empty_cache()
    scene = scene.to(device).detach()
    original_rcs = rcs.to(device).detach()   # rcs generated from the scene
    low_res = isar.to(device)  # isar image generated from the rcs

    scale_factor = torch.max(torch.abs(low_res))   

    high_res = network(low_res/scale_factor)
    high_res = high_res.view(1, high_res.shape[-2], high_res.shape[-1])
    high_res = high_res*scale_factor

    # scene = scene.unsqueeze(dim=0)
    ssim_ = ssim(torch.abs(scene), torch.abs(low_res), 255)
    print("SSIM abs -- Scene - LR ISAR: ", ssim_.detach().cpu().numpy())
    ssim_ = ssim(scene.real, low_res.real, 255)
    print("SSIM real -- Scene - LR ISAR: ", ssim_.detach().cpu().numpy())
    ssim_ = ssim(scene.imag, low_res.imag, 255)
    print("SSIM imag -- Scene - LR ISAR: ", ssim_.detach().cpu().numpy())

    print("-------------------------------------------------------------")
    ssim_ = ssim(torch.abs(scene), torch.abs(high_res), 255)
    print("SSIM abs -- Scene - HR ISAR: ", ssim_.detach().cpu().numpy())
    ssim_ = ssim(scene.real, high_res.real, 255)
    print("SSIM real -- Scene - HR ISAR: ", ssim_.detach().cpu().numpy())
    ssim_ = ssim(scene.imag, high_res.imag, 255)
    print("SSIM imag -- Scene - HR ISAR: ", ssim_.detach().cpu().numpy())
    print("-------------------------------------------------------------")


    plt.figure("Original scene")
    fig = utils.plot_scene(torch.abs(scene[0].cpu()), x_range.cpu(), y_range.cpu())

    plt.figure("RCS")
    plt.imshow(torch.abs(original_rcs[0]).cpu(), cmap="inferno", extent=[-11.7891,11.7981,12,8], aspect='auto')
    plt.xlabel('Angle (Degrees)')
    plt.ylabel('Frequency (GHz)')
    plt.gca().invert_yaxis()
    #plt.colorbar()
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Amplitude (Linear)', rotation=270, labelpad=20)

    plt.figure("Low-res isar 2D")
    fig = utils.plotcut_dB_in(low_res[0].cpu(), x_range.cpu(), y_range.cpu())

    plt.figure("High-res isar 2D")
    fig = utils.plotcut_dB_in(high_res[0].detach().cpu().numpy(), x_range.cpu(), y_range.cpu())    

    #plt.figure("High-res x isar 1D horizontal")
    plt.figure("ISAR 1D horizontal")
    fig = utils.plot_horizontal2(low_res[0].cpu(), x_range.cpu(), high_res[0].detach().cpu(), x_range.cpu())

    plt.figure("ISAR 1D horizontal normalised")
    fig = utils.plot_horizontal2(low_res[0].cpu(), x_range.cpu(), (high_res[0] / torch.max(torch.abs(high_res[0])) * torch.max(torch.abs(low_res[0]))).detach().cpu(), x_range.cpu())

    plt.figure("ISAR 1D vertical")
    fig = utils.plot_vertical2(low_res[0].cpu(), y_range.cpu(), high_res[0].detach().cpu(), y_range.cpu())

    plt.figure("ISAR 1D vertical normalised")
    fig = utils.plot_vertical2(low_res[0].cpu(), y_range.cpu(), (high_res[0] / torch.max(torch.abs(high_res[0])) * torch.max(torch.abs(low_res[0]))).detach().cpu(), y_range.cpu())

    plt.show()
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
from CustomDataset import CustomDataset
from pynput.keyboard import Key, Listener
from generate_rcs import generate_rcs, generate_rcs_sparse
import torch.nn.functional as F
from generate_scene import generate_point_scene_same_col, generate_single_point_scene, generate_line
import matlab.engine
import scipy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define network
network = ISARNet().to(device)

network = utils.load_model_state(network, "best_model.pth", AiQu=False)

# load rcs, isar and overlay
original_rcs = torch.load("meas_data/rcs/rcs_cylinder.pt")
low_res = torch.load("meas_data/isars/isar_cylinder.pt")
overlay = scipy.io.loadmat("meas_data/rund_ov.mat")

# cylinder
cal_range = 7.61
theta_start = -180.14
theta_stop = 179.66000000000003
ntheta = 1800
fstart = 5.997
fstop = 16.000500000000002
nf = 135
xmax = 1
ymax = 1

# car
# cal_range = 10.8
# theta_start = -201
# theta_stop = 231
# ntheta = 21601
# fstart = 13.5
# fstop = 16.5
# nf = 122
# xmax = 2.6
# ymax = 2.6

# airplanes
# cal_range = 25.0
# theta_start = -10.0
# theta_stop = 190.0
# ntheta = 801
# fstart = 6
# fstop = 16
# nf = 201
# xmax = 1
# ymax = 1

f = np.linspace(fstart, fstop, nf)
phi = np.linspace(theta_start, theta_stop, ntheta)

# cylinder
phi = phi[842:961]
f = f[26:82]

# car
# phi = phi[9460:10640]

# airplanes
# phi = phi[0:81]
# f = f[40:121]

ff = 0
nf = len(f)
nphi = len(phi)
nbins = 2048 # nr of positions in the RCS to calculate

# isar image properties
# xmax = 1
xmin = -xmax
nx = 256
# ymax = 1
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

original_rcs = original_rcs.to(device)
low_res = low_res.to(device)

scale_factor = torch.max(torch.abs(low_res))

# convert from numpy to torch
# low_res = torch.from_numpy(low_res).to(device).type(torch.complex64)
phi = torch.from_numpy(phi).to(device)
f = torch.from_numpy(f).to(device)

low_res = low_res.view(1, 1, low_res.shape[-2], low_res.shape[-1])

scale_factor = torch.max(torch.abs(low_res))   
start = time()
high_res = network(low_res/scale_factor)
end = time()
print(end- start)
high_res = high_res.view(1, high_res.shape[-2], high_res.shape[-1])
high_res = high_res*scale_factor

plt.figure("Low-res isar 2D")
fig = utils.plotcut_dB_in(low_res[0,0].cpu(), x_range.cpu(), y_range.cpu())
#utils.plot_overlay(overlay, linewidth=0.5)

plt.figure("High-res isar 2D")
fig = utils.plotcut_dB_in(high_res[0].detach().cpu().numpy(), x_range.cpu(), y_range.cpu())

plt.figure("High-res isar 2D with overlay")
fig = utils.plotcut_dB_in(high_res[0].detach().cpu().numpy(), x_range.cpu(), y_range.cpu())
#utils.plot_overlay(overlay)

#plt.figure("High-res x isar 1D horizontal")
plt.figure("ISAR 1D horizontal")
fig = utils.plot_horizontal2(low_res[0,0].cpu(), x_range.cpu(), high_res[0].detach().cpu(), x_range.cpu())

plt.figure("ISAR 1D horizontal normalised")
fig = utils.plot_horizontal2(low_res[0,0].cpu(), x_range.cpu(), (high_res[0] / torch.max(torch.abs(high_res[0])) * torch.max(torch.abs(low_res[0,0]))).detach().cpu(), x_range.cpu())

plt.figure("ISAR 1D vertical")
fig, _, _ = utils.plot_vertical2(low_res[0,0].cpu(), y_range.cpu(), high_res[0].detach().cpu(), y_range.cpu())

plt.figure("ISAR 1D vertical normalised")
fig, col1, col2 = utils.plot_vertical2(low_res[0,0].cpu(), y_range.cpu(), (high_res[0] / torch.max(torch.abs(high_res[0])) * torch.max(torch.abs(low_res[0,0]))).detach().cpu(), y_range.cpu())

# max_lr = torch.max(torch.abs(low_res[0,0,:,col1]))
# max_hr = torch.max(torch.abs(high_res[0,:,col2])) / torch.max(torch.abs(high_res[0,:,col2])) * torch.max(torch.abs(low_res[0,0,:,col1]))

plt.show()
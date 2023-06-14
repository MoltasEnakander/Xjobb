import sys
sys.path.insert(1, r"C:\Users\Lovisa Nilsson\Desktop\xjobb\MyTorch")

import os
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch
from CustomDataset import CustomDataset
from generate_rcs import generate_rcs
import matlab.engine
from generate_scene import generate_scene, generate_single_point_scene, generate_point_scene_same_row

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load rcs, isar and overlay
original_rcs = torch.load("meas_data/rcs/rcs_curvedplane.pt")
low_res = torch.load("meas_data/isars/isar_curvedplane.pt")

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
# # phi = phi[842:961]
# # f = f[26:82]

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

rcs_norm = original_rcs / torch.max(torch.abs(low_res))
diffx = torch.max(torch.abs(torch.diff(rcs_norm, dim=1)))
diffy = torch.max(torch.abs(torch.diff(rcs_norm, dim=0)))
complexity = torch.sqrt(diffx**2 + diffy**2)
print(complexity)
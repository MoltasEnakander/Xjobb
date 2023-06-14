# setup shit
import os
import numpy as np
import torch
from tqdm import tqdm
import matlab.engine
from generate_scene import generate_scene, generate_point_scene, generate_point_scene_same_col, generate_point_scene_same_row
from generate_rcs import generate_rcs, generate_rcs_sparse
import matplotlib.pyplot as plt
import utils
from CustomDataset import CustomDataset
import scipy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read data
eng = matlab.engine.start_matlab()
eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner")
eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner\Moltas")

mat = scipy.io.loadmat("meas_data/cylinder32mm0001_VV_gated.mat")
#mat = scipy.io.loadmat("meas_data/bil_bmw_hh_gated.mat")
# mat = scipy.io.loadmat("meas_data/bil_bmw_overlay.mat")

cal_range = mat['calrange'][0][0]
cal_range = cal_range.item()

rcs = mat['rcs_all']

theta_start = mat['thetastart'][0][0]
theta_stop = mat['thetastop'][0][0]
ntheta = mat['ntheta'][0][0]
fstart = mat['fstart'][0][0]
fstop = mat['fstop'][0][0]
nf = mat['nf'][0][0]

f = np.linspace(fstart, fstop, nf)
phi = np.linspace(theta_start, theta_stop, ntheta)
phi = phi[842:961]
f = f[26:82]
original_rcs = rcs[26:82, 842:961]

f_mat = matlab.double(f.tolist())
phi_mat = matlab.double(phi.tolist())
rcs_mat = matlab.double(original_rcs.tolist(), is_complex=True)

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
x_range_mat = matlab.double(x_range.tolist()) # create matlab object of x, both will be needed later on
y_range_mat = matlab.double(y_range.tolist()) # same but for y
hanning_flag = 0
elev_angle = 0

low_res = eng.generateISAR(rcs_mat, f_mat, phi_mat, hanning_flag, elev_angle, cal_range, ff, x_range_mat, y_range_mat)
low_res = np.asarray(low_res) # matlab -> numpy
low_res = np.transpose(low_res)

# convert from numpy to torch
low_res = torch.from_numpy(low_res).to(device).type(torch.complex64)
phi = torch.from_numpy(phi).to(device)
f = torch.from_numpy(f).to(device)
original_rcs = torch.from_numpy(original_rcs).to(device)

for i in range(20):
    bins = torch.randperm(nf * nphi)
    bins = bins[0:nbins]
    # convert to freqs and angles
    freqs = bins // nphi
    angles = bins % nphi

    new_rcs = generate_rcs_sparse(low_res, cal_range, ff, f[freqs], phi[angles], x_range, y_range, nbins, device)
    alpha = torch.max(torch.abs(new_rcs)) / torch.max(torch.abs(original_rcs))

    # diff = new_rcs/22.2 - original_rcs[0, freqs, angles]
    # diff = torch.unsqueeze(diff, dim=0)
    # l2_loss = torch.abs(diff) @ torch.transpose(torch.abs(torch.conj(diff)), -1, -2)
    # print(l2_loss)

    # plt.figure("Original RCS")
    # plt.plot(torch.abs(original_rcs[0, freqs, angles]).detach().cpu().numpy())

    # plt.figure("New RCS")
    # plt.plot(torch.abs(new_rcs).detach().cpu().numpy())

    # plt.figure("New RCS normed")
    # plt.plot(torch.abs(new_rcs/22.2).detach().cpu().numpy())

    # plt.figure("RCS diff")
    # plt.plot(torch.abs(new_rcs/22.2 - original_rcs[0, freqs, angles]).detach().cpu().numpy())

    # plt.show()
            


    # # calculate average alpha value
    # alpha = (rcs_scale_points) / (len(os.listdir(scene_path)))

    print("Alpha: ", alpha.cpu().numpy())
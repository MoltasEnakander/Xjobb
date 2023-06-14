# setup shit
import os
import numpy as np
import torch
from tqdm import tqdm
import matlab.engine
from generate_scene import generate_scene, generate_point_scene, generate_point_scene_same_col, generate_point_scene_same_row, generate_line, generate_single_point_scene
from generate_rcs import generate_rcs, generate_rcs_sparse
import matplotlib.pyplot as plt
import utils
from CustomDataset import CustomDataset
import scipy

def pseudo_l0(y, mu):
    # pseudo l0-norm is defined as 
    #                            ( 1        if |y_i| >= mu
    # l0 = sum all a_i for a_i = ( |y_i|/mu if 0 < |y_i| < mu
    #                            ( 0        others
    return torch.sum((torch.abs(y) >= mu) + ((torch.logical_and(torch.abs(y)>0,torch.abs(y)<mu)) / mu) * torch.abs(y))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read data
eng = matlab.engine.start_matlab()
eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner")
eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner\Moltas")

scene_width = 256
scene_height = 256    

# echo calculation properties
cal_range = 7.5
ff = 0
f_start = 8
f_stop = 12
nf = 256
nphi = 256
nbins = 2048

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

bins = torch.randperm(nf * nphi)
bins = bins[0:nbins]
# convert to freqs and angles
freqs = bins // nphi
angles = bins % nphi

new_rcs = generate_rcs_sparse(low_res, cal_range, ff, f[freqs], phi[angles], x_range, y_range, nbins, device)
alpha = torch.max(torch.abs(new_rcs)) / torch.max(torch.abs(original_rcs))

diff = new_rcs/25.2 - original_rcs[freqs, angles]
diff = torch.unsqueeze(diff, dim=0)
l2_loss = torch.abs(diff) @ torch.transpose(torch.abs(torch.conj(diff)), -1, -2) / ((torch.max(torch.abs(low_res)))**2)
print(l2_loss)
print("Alpha: ", alpha.cpu().numpy())
mu = 5
pseudo_l0_loss = pseudo_l0(low_res/torch.max(torch.abs(scene)), mu)
print(pseudo_l0_loss)

# plt.figure("Original RCS")
# plt.plot(torch.abs(original_rcs[freqs, angles]).detach().cpu().numpy())

# plt.figure("New RCS")
# plt.plot(torch.abs(new_rcs).detach().cpu().numpy())

# plt.figure("New RCS normed")
# plt.plot(torch.abs(new_rcs/25.2).detach().cpu().numpy())

# plt.figure("RCS diff")
# plt.plot(torch.abs(new_rcs/25.2 - original_rcs[freqs, angles]).detach().cpu().numpy())

# plt.show()


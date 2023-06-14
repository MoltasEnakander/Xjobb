# setup shit
# import sys
# sys.path.insert(1, r"C:\Users\Lovisa Nilsson\Desktop\xjobb\MyTorch")

import os
import numpy as np
import torch
from tqdm import tqdm
#import matlab.engine
#from generate_scene import generate_scene, generate_point_scene, generate_point_scene_same_col, generate_point_scene_same_row
from generate_rcs import generate_rcs, generate_rcs_sparse
import matplotlib.pyplot as plt
import utils
from CustomDataset import CustomDataset

# eng = matlab.engine.start_matlab()
# eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner")
# eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner\Moltas")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

scene_width = 256
scene_height = 256
max_shapes = 5  # maximum nr of objects that are allowed to be in the random scene

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

rcs_scale_points = 0.0
nbins = 2048

from torch.utils.data import DataLoader
#scene_path = os.path.join('largedataset/', 'scenes1/')
rcs_path = os.path.join('largedataset/', 'rcs1/')
isar_path = os.path.join('largedataset/', 'isars1/')
batch_size = 1

dataset = CustomDataset(rcs_path, isar_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# create a bunch of scenes with random nr of points in random locations

for batch_idx, (rcs, isar) in enumerate(tqdm(dataloader)):
    with torch.no_grad():
        original_rcs = rcs.to(device).detach()   # rcs generated from the scene
        low_res = isar.to(device)  # isar image generated from the rcs

        bins = torch.randperm(nf * nphi)
        bins = bins[0:nbins]
        # convert to freqs and angles
        freqs = bins // nphi
        angles = bins % nphi

        new_rcs = generate_rcs_sparse(low_res[0], cal_range, ff, f[freqs], phi[angles], x_range, y_range, nbins, device)
        rcs_scale_points += torch.max(torch.abs(new_rcs)) / torch.max(torch.abs(original_rcs[0, freqs, angles]))

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
        


# calculate average alpha value
alpha = (rcs_scale_points) / (len(os.listdir(rcs_path)))

print("Alpha: ", alpha)
import sys
sys.path.insert(1, r"C:\Users\Lovisa Nilsson\Desktop\xjobb\MyTorch")

import os
from time import time
from tqdm import tqdm
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from CustomDataset import CustomDataset
from generate_rcs import generate_multiple_rcs_sparse, generate_rcs_sparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# rcs calculation properties
cal_range = 7.5
ff = 0
f_start = 8
f_stop = 12
nf = 256
nphi = 256
nbins = 2048 # nr of positions in the RCS to calculate

f = torch.linspace(f_start,f_stop, nf).to(device)
B = f_stop - f_start
fc = (f_start + f_stop)/2
# Resolution in y is c/(2B). Match x-resolution which is c/(2*fc*sin(phi_tot))
phi_tot = np.arcsin(B/fc)*180/np.pi; # convert to degrees
phi = torch.linspace(-phi_tot/2, phi_tot/2, nphi).to(device)

xmax = 1
xmin = -xmax
nx = 256
ymax = 1
ymin = -ymax
ny = 256
x_range = torch.linspace(xmin,xmax,nx).to(device)
y_range = torch.linspace(ymin,ymax,ny).to(device)

from torch.utils.data import DataLoader
scene_path = os.path.join('largedataset/', 'scenes1/')
rcs_path = os.path.join('largedataset/', 'rcs1/')
isar_path = os.path.join('largedataset/', 'isars1/')
batch_size = 1

dataset = CustomDataset(scene_path, rcs_path, isar_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

complexities = []

files = os.listdir(scene_path)

for batch_idx, (scene, rcs, isar) in enumerate(tqdm(dataloader)):    
    isar = isar.to(device).detach()   # scene

    bins = torch.randperm(nf * nphi)
    bins = bins[0:nbins]
    # convert to freqs and angles
    freqs = bins // nphi
    angles = bins % nphi

    rcs1 = generate_multiple_rcs_sparse(isar, cal_range, ff, f[freqs], phi[angles], x_range, y_range, nbins, device)

    rcs2 = generate_rcs_sparse(isar[0], cal_range, ff, f[freqs], phi[angles], x_range, y_range, nbins, device)

    diff = rcs2 - rcs1

    print(torch.sum(torch.abs(diff)))


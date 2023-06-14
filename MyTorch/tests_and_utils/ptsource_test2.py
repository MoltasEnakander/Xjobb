import sys
sys.path.insert(1, r"C:\Users\Lovisa Nilsson\Desktop\xjobb\MyTorch")

import torch
from ptsource import ptsource, ptsource_time, ptsource_sparse, ptsource_sparse2
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

scene_width = 256
scene_height = 256
max_shapes = 4  # maximum nr of objects that are allowed to be in the random scene

# echo calculation properties
cal_range = 7.5
ff = 0
f_start = 8
f_stop = 12
nf = 128
nphi = 128

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
# phi = torch.tensor([0]).to(device)

x = torch.tensor([0, 0.5, -0.4, 0.2, 0.8, -0.7]).to(device)
y = torch.tensor([0, 0.5, -0.1, 0.1, 0.05, -0.25]).to(device)
amp = torch.tensor([1, 0.2, 0.6, 0.4, 0.3, 0.7]).to(device)

nbins = 1 # nr of positions in the RCS to calculate
freqs = torch.randint(0, nf, size=(nbins,))
angles = torch.randint(0, nphi, size=(nbins, ))
freqs = torch.tensor([12]).to(device)
angles = torch.tensor([12]).to(device)
rcs_sparse1 = ptsource_sparse2(3,x,y,amp,f[freqs],phi[angles],cal_range,ff,nbins,device)
rcs = ptsource(x[0:2], y[0:2], amp[0:2], f, phi, cal_range, ff, device, method=1)
rcs2 = ptsource(x[2:4], y[2:4], amp[2:4], f, phi, cal_range, ff, device, method=1)
rcs3 = ptsource(x[4:6], y[4:6], amp[4:6], f, phi, cal_range, ff, device, method=1)
print(rcs_sparse1)
print(rcs[12,12])
print(rcs2[12,12])
print(rcs3[12,12])
print("hej")
#rcs_sparse2 = ptsource_sparse(x,y,amp,f[freqs],phi[angles],cal_range,ff,nbins,device,loop=2)

#print(torch.sum(torch.abs(rcs_sparse1 - rcs_sparse2)))
#print(torch.count_nonzero(~(rcs_sparse1 == rcs_sparse2)))

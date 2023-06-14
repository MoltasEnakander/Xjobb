import sys
sys.path.insert(1, r"C:\Users\Lovisa Nilsson\Desktop\xjobb\MyTorch")

import torch
from ptsource import ptsource, ptsource_time
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

x = torch.tensor([0, 0.5, -0.4]).to(device)
y = torch.tensor([0, 0.5, -0.1]).to(device)
amp = torch.tensor([1, 0.2, 0.6]).to(device)

rcs = ptsource(x,y,amp,f,phi,cal_range,ff,device,loop=2)

rcs_t = ptsource_time(x, y, amp, f, phi, cal_range, ff, device=device, loop=1)
rcs_t = torch.squeeze(rcs_t)

rcs_t2 = ptsource_time(x, y, amp, f, phi, cal_range, ff, device=device, loop=2)

diff = rcs_t - rcs

print(torch.sum(torch.abs(diff)))

rcs_f = torch.fft.fft(rcs_t, dim=-2)
rcs_f2 = torch.fft.fft(rcs_t2, dim=-2)

plt.figure(1)
plt.imshow(torch.abs(rcs_t).cpu().numpy(), cmap="inferno")
plt.colorbar()

plt.figure(2)
plt.imshow(torch.abs(rcs_t2).cpu().numpy(), cmap="inferno")
plt.colorbar()

plt.figure(3)
plt.imshow(torch.abs(rcs_f).cpu().numpy(), cmap="inferno")
plt.colorbar()

plt.figure(4)
plt.imshow(torch.abs(rcs_f2).cpu().numpy(), cmap="inferno")
plt.colorbar()

plt.figure(5)
plt.imshow(torch.abs(rcs).cpu().numpy(), cmap="inferno")
plt.colorbar()
plt.show()




import sys
sys.path.insert(1, r"C:\Users\Lovisa Nilsson\Desktop\xjobb\MyTorch")

import os
import numpy as np
import torch
import matlab.engine
from generate_scene import generate_scene
from generate_rcs import generate_rcs, generate_rcs_sparse
import matplotlib.pyplot as plt
import utils

eng = matlab.engine.start_matlab()
eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner")
eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner\Moltas")

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

# some type conversions (matlab does not like python objects)
f_mat = matlab.double(f.tolist())
phi_mat = matlab.double(phi.tolist())
x_range_mat = matlab.double(x_range.tolist()) # create matlab object of x, both will be needed later on
y_range_mat = matlab.double(y_range.tolist()) # same but for y



### create scene 256x256
scene = generate_scene(scene_height, scene_width, max_shapes)        
scene = torch.from_numpy(scene).to(device).type(torch.complex64)

### create rcs_lo 128x128
rcs_lo = generate_rcs(scene, cal_range, ff, f, phi, x_range, y_range, device, method=1)    
rcs_lo_mat = matlab.double(rcs_lo.detach().cpu().numpy().tolist(), is_complex=True)

### create isar_lo 256x256
isar_lo = eng.generateISAR(rcs_lo_mat, f_mat, phi_mat, hanning_flag, elev_angle, cal_range, ff, x_range_mat, y_range_mat)
isar_lo = torch.from_numpy(np.asarray(isar_lo)).to(device).type(torch.complex64) # matlab -> numpy -> torch tensor
isar_lo = torch.transpose(isar_lo, 0, 1)

### create rcs_hi 256x256
nf = 256
nphi = 256
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

rcs_hi = generate_rcs(scene, cal_range, ff, f, phi, x_range, y_range, device, method=1)    
rcs_hi_mat = matlab.double(rcs_hi.detach().cpu().numpy().tolist(), is_complex=True)

### create isar_hi 256x256
isar_hi = eng.generateISAR(rcs_hi_mat, f_mat, phi_mat, hanning_flag, elev_angle, cal_range, ff, x_range_mat, y_range_mat)
isar_hi = torch.from_numpy(np.asarray(isar_hi)).to(device).type(torch.complex64) # matlab -> numpy -> torch tensor
isar_hi = torch.transpose(isar_hi, 0, 1)

### pseudo_l0

### create rcs_lo2 128x128 av isar_lo
nf = 128
nphi = 128
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

plt.figure("Original scene")
fig = utils.plot_scene(torch.abs(scene.cpu()), x_range.cpu(), y_range.cpu())

plt.figure("Low-res isar")
fig = utils.plotcut_dB_in(isar_lo.cpu(), x_range.cpu(), y_range.cpu())

plt.figure("High-res isar")
fig = utils.plotcut_dB_in(isar_hi.cpu().numpy(), x_range.cpu(), y_range.cpu())
plt.show()

nbins = 1024
for i in range(10):
    bins = torch.randperm(nf * nphi)
    bins = bins[0:nbins]
    # convert to freqs and angles
    freqs = bins // nphi
    angles = bins % nphi

    rcs_lo2 = generate_rcs_sparse(isar_lo, cal_range, ff, f[freqs], phi[angles], x_range, y_range, nbins, device, method=1)

    ### create rcs_hi2 128x128 av isar_hi
    rcs_hi2 = generate_rcs_sparse(isar_hi, cal_range, ff, f[freqs], phi[angles], x_range, y_range, nbins, device, method=1)

    ### l2-norm
    print("L2-norm of RCSs for the two ISAR images: ", torch.square(torch.linalg.norm(rcs_hi2 - rcs_lo2)))
    print("L2-norm of RCSs for the true scene and ISAR_lo: ", torch.square(torch.linalg.norm(rcs_lo[freqs, angles] - rcs_lo2)))

print("------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------------")

nbins = 1024
for i in range(10):
    bins = torch.arange(nf * nphi)
    bins = bins[i*nbins:nbins*(i+1)]
    # convert to freqs and angles
    freqs = bins // nphi
    angles = bins % nphi

    rcs_lo2 = generate_rcs_sparse(isar_lo, cal_range, ff, f[freqs], phi[angles], x_range, y_range, nbins, device, method=1)

    ### create rcs_hi2 128x128 av isar_hi
    rcs_hi2 = generate_rcs_sparse(isar_hi, cal_range, ff, f[freqs], phi[angles], x_range, y_range, nbins, device, method=1)

    ### l2-norm
    print("L2-norm of RCSs for the two ISAR images: ", torch.square(torch.linalg.norm(rcs_hi2 - rcs_lo2)))
    print("L2-norm of RCSs for the true scene and ISAR_lo: ", torch.square(torch.linalg.norm(rcs_lo[freqs, angles] - rcs_lo2)))

    # plt.figure("Original RCS hi")
    # plt.imshow(torch.abs(rcs_hi).detach().cpu().numpy())

    # plt.figure("Original RCS lo")
    # plt.imshow(torch.abs(rcs_lo).detach().cpu().numpy())

    # plt.figure("New RCS hi")
    # plt.plot(torch.abs(rcs_hi2).detach().cpu().numpy())

    # plt.figure("New RCS lo")
    # plt.plot(torch.abs(rcs_lo2).detach().cpu().numpy())
    # plt.show()

### gör detta för olika typer av scener och testa hur slumpningen av bins påverkar



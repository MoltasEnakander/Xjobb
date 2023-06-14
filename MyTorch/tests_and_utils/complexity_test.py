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

# scene properties
scene_width = 256
scene_height = 256

# rcs calculation properties
calrange = 7.5
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
hanning_flag = 0
elev_angle = 0
cal_range = 7.5

# from torch.utils.data import DataLoader
# scene_path = os.path.join('data256_256/', 'scenes/')
# rcs_path = os.path.join('data256_256/', 'rcs/')
# isar_path = os.path.join('data256_256/', 'isars/')
# batch_size = 1

# dataset = CustomDataset(scene_path, rcs_path, isar_path)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

eng = matlab.engine.start_matlab()
eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner")
eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner\Moltas")

while True:
    scene = generate_scene(scene_height, scene_width, 4)
    scene = torch.from_numpy(scene).to(device).type(torch.complex64)
    print("scene")

    original_rcs = generate_rcs(scene, cal_range, ff, f, phi, x_range, y_range, device, method=1)
    orig_rcs_mat = matlab.double(original_rcs.detach().cpu().numpy().tolist(), is_complex=True)
    print("rcs")

    low_res = eng.generateISAR(orig_rcs_mat, f_mat, phi_mat, hanning_flag, elev_angle, cal_range, ff, x_range_mat, y_range_mat)
    low_res = torch.from_numpy(np.asarray(low_res)).to(device).type(torch.complex64) # matlab -> numpy -> torch tensor
    low_res = torch.transpose(low_res, 0, 1)
    print("isar")    

    rcs_norm = original_rcs / torch.max(torch.abs(low_res))
    diffx = torch.max(torch.abs(torch.diff(rcs_norm, dim=1)))
    diffy = torch.max(torch.abs(torch.diff(rcs_norm, dim=0)))
    complexity = torch.sqrt(diffx**2 + diffy**2)
    print(complexity)

    fig1 = plt.figure("Original scene")
    fig = utils.plot_scene(torch.abs(scene.cpu()), x_range.cpu(), y_range.cpu())

    fig2 = plt.figure("ISAR")
    fig = utils.plotcut_dB_in(low_res.cpu(), x_range.cpu(), y_range.cpu())

    fig3 = plt.figure("Original RCS")
    plt.imshow(torch.abs(original_rcs).cpu().numpy(), cmap="inferno")
    plt.xlabel('Angular bins')
    plt.ylabel('Frequency bins')
    #plt.colorbar()
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Amplitude (Linear)', rotation=270, labelpad=20)

    plt.show()
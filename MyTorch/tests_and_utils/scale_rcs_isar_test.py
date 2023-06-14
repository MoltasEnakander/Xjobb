import os
import numpy as np
import torch
from tqdm import tqdm
import matlab.engine
from generate_scene import generate_scene, generate_single_point_scene, generate_line
from generate_rcs import generate_rcs
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

# some type conversions (matlab does not like python objects)
f_mat = matlab.double(f.tolist())
phi_mat = matlab.double(phi.tolist())
x_range_mat = matlab.double(x_range.tolist()) # create matlab object of x, both will be needed later on
y_range_mat = matlab.double(y_range.tolist()) # same but for y

N = 1 # nr of scenes to create

for i in tqdm(range(N)):
    scene = generate_scene(scene_height, scene_width, 4)        
    scene = torch.from_numpy(scene).to(device).type(torch.complex64)

    rcs = generate_rcs(scene, cal_range, ff, f, phi, x_range, y_range, device, method=1)    
    rcs_mat = matlab.double(rcs.detach().cpu().numpy().tolist(), is_complex=True)

    low_res = eng.generateISAR(rcs_mat, f_mat, phi_mat, hanning_flag, elev_angle, cal_range, ff, x_range_mat, y_range_mat)
    low_res = torch.from_numpy(np.asarray(low_res)).to(device).type(torch.complex64) # matlab -> numpy -> torch tensor
    low_res = torch.transpose(low_res, 0, 1)   

    scale_factor = torch.max(torch.abs(low_res))    

    rcs_normed = rcs / scale_factor
    rcs_normed_mat = matlab.double(rcs_normed.detach().cpu().numpy().tolist(), is_complex=True)

    low_res2 = eng.generateISAR(rcs_normed_mat, f_mat, phi_mat, hanning_flag, elev_angle, cal_range, ff, x_range_mat, y_range_mat)
    low_res2 = torch.from_numpy(np.asarray(low_res2)).to(device).type(torch.complex64) # matlab -> numpy -> torch tensor
    low_res2 = torch.transpose(low_res2, 0, 1)

    low_res3 = low_res / scale_factor

    fig1 = plt.figure(1)
    plt.imshow(torch.abs(scene).cpu(), cmap="inferno")
    plt.gca().invert_yaxis()
    plt.colorbar()    

    # fig3 = plt.figure(3)
    # plt.imshow(torch.abs(rcs).cpu(), cmap="inferno")
    # plt.colorbar()

    fig2 = plt.figure(2)
    fig = utils.plotcut_dB_in(low_res.cpu(), x_range.cpu(), y_range.cpu())
    
    fig2 = plt.figure(3)
    fig = utils.plotcut_dB_in(low_res2.cpu(), x_range.cpu(), y_range.cpu())

    fig2 = plt.figure(4)
    fig = utils.plotcut_dB_in(low_res3.cpu(), x_range.cpu(), y_range.cpu())

    plt.show()

eng.quit()
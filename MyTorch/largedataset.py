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

eng = matlab.engine.start_matlab()
eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner")
eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner\Moltas")

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

# some type conversions (matlab does not like python objects)
f_mat = matlab.double(f.tolist())
phi_mat = matlab.double(phi.tolist())
x_range_mat = matlab.double(x_range.tolist()) # create matlab object of x, both will be needed later on
y_range_mat = matlab.double(y_range.tolist()) # same but for y

rcs_scale_points = 0.0
rcs_scale_shapes = 0.0
nbins = 2048

# create a bunch of scenes with random nr of points in random locations
N = 1500 # nr of scenes to create

for i in tqdm(range(N)):
    scene = generate_point_scene_same_row(scene_height, scene_width)        
    scene = torch.from_numpy(scene).to(device).type(torch.complex64)

    rcs = generate_rcs(scene, cal_range, ff, f, phi, x_range, y_range, device, method=1)    
    rcs_mat = matlab.double(rcs.detach().cpu().numpy().tolist(), is_complex=True)

    low_res = eng.generateISAR(rcs_mat, f_mat, phi_mat, hanning_flag, elev_angle, cal_range, ff, x_range_mat, y_range_mat)
    low_res = torch.from_numpy(np.asarray(low_res)).to(device).type(torch.complex64) # matlab -> numpy -> torch tensor
    low_res = torch.transpose(low_res, 0, 1)

    scene_path = os.path.join("largedataset/scenes/", "point_scene_" + str(3000 + i) + ".pt")
    rcs_path = os.path.join("largedataset/rcs/", "point_rcs_" + str(3000 + i) + ".pt")
    isar_path = os.path.join("largedataset/isars/", "point_isar_" + str(3000 + i) + ".pt")

    torch.save(scene, scene_path)
    torch.save(rcs, rcs_path)
    torch.save(low_res, isar_path)

    bins = torch.randperm(nf * nphi)
    bins = bins[0:nbins]
    # convert to freqs and angles
    freqs = bins // nphi
    angles = bins % nphi

    new_rcs = generate_rcs_sparse(low_res, cal_range, ff, f[freqs], phi[angles], x_range, y_range, nbins, device)
    rcs_scale_points += torch.max(torch.abs(new_rcs)) / torch.max(torch.abs(rcs[freqs, angles]))

    # plt.figure("Original scene")
    # fig = utils.plot_scene(torch.abs(scene.cpu()), x_range.cpu(), y_range.cpu())
    # plt.show()
# create a bunch of scenes with random objects (with and without points)
M = 1500

for i in tqdm(range(M)):
    scene = generate_point_scene_same_col(scene_height, scene_width)        
    scene = torch.from_numpy(scene).to(device).type(torch.complex64)

    rcs = generate_rcs(scene, cal_range, ff, f, phi, x_range, y_range, device, method=1)    
    rcs_mat = matlab.double(rcs.detach().cpu().numpy().tolist(), is_complex=True)

    low_res = eng.generateISAR(rcs_mat, f_mat, phi_mat, hanning_flag, elev_angle, cal_range, ff, x_range_mat, y_range_mat)
    low_res = torch.from_numpy(np.asarray(low_res)).to(device).type(torch.complex64) # matlab -> numpy -> torch tensor
    low_res = torch.transpose(low_res, 0, 1)

    scene_path = os.path.join("largedataset/scenes/", "point_scene_" + str(4500 + i) + ".pt")
    rcs_path = os.path.join("largedataset/rcs/", "point_rcs_" + str(4500 + i) + ".pt")
    isar_path = os.path.join("largedataset/isars/", "point_isar_" + str(4500 + i) + ".pt")

    torch.save(scene, scene_path)
    torch.save(rcs, rcs_path)
    torch.save(low_res, isar_path)

    bins = torch.randperm(nf * nphi)
    bins = bins[0:nbins]
    # convert to freqs and angles
    freqs = bins // nphi
    angles = bins % nphi

    new_rcs = generate_rcs_sparse(low_res, cal_range, ff, f[freqs], phi[angles], x_range, y_range, nbins, device)
    rcs_scale_shapes += torch.max(torch.abs(new_rcs)) / torch.max(torch.abs(rcs[freqs, angles]))

    # plt.figure("Original scene")
    # fig = utils.plot_scene(torch.abs(scene.cpu()), x_range.cpu(), y_range.cpu())
    # plt.show()


# calculate average alpha value
alpha = (rcs_scale_points + rcs_scale_shapes) / (N+M)
alpha_points = rcs_scale_points / N
alpha_shapes = rcs_scale_shapes / M

print("Alpha: ", alpha)
print("Alpha points: ", alpha_points)
print("Alpha shapes: ", alpha_shapes)


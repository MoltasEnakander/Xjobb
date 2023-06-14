# todo: testa hur skalningen blir av att skapa rcs från en lowres isar bild, troligtvis leder det till en enorm ökning av skala eftersom det är
# så otroligt många punkter som det räknas för.
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
from generate_scene import generate_scene, generate_single_point_scene, generate_point_scene

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# scene properties
scene_width = 256
scene_height = 256

# rcs calculation properties
calrange = 7.5
ff = 0
f_start = 8
f_stop = 12
nf = 100
nphi = 100

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
    #for batch_idx, (scene, rcs, isar) in enumerate((dataloader)):
    scene = generate_single_point_scene(scene_height, scene_width)        
    scene = torch.from_numpy(scene).to(device).type(torch.complex64)
    print("scene")

    original_rcs = generate_rcs(scene, cal_range, ff, f, phi, x_range, y_range, device, method=1)
    orig_rcs_mat = matlab.double(original_rcs.detach().cpu().numpy().tolist(), is_complex=True)
    print("rcs")

    low_res = eng.generateISAR(orig_rcs_mat, f_mat, phi_mat, hanning_flag, elev_angle, cal_range, ff, x_range_mat, y_range_mat)
    low_res = torch.from_numpy(np.asarray(low_res)).to(device).type(torch.complex64) # matlab -> numpy -> torch tensor
    low_res = torch.transpose(low_res, 0, 1)
    print("isar")

    # torch.cuda.empty_cache()
    # scene = scene.to(device)
    # original_rcs = rcs.to(device).detach()   # rcs generated from the scene
    # low_res = isar.to(device)  # isar image generated from the rcs

    rcs_temp = torch.ones_like(original_rcs)              
    new_rcs = generate_rcs(low_res, calrange, ff, f, phi, x_range, y_range, device, method=1)
    # print("rcs2")
    # rcs_mat = matlab.double(new_rcs.detach().cpu().numpy().tolist(), is_complex=True)
    # hanning_flag = 0
    # low_res2 = eng.generateISAR(rcs_mat, f_mat, phi_mat, hanning_flag, elev_angle, cal_range, ff, x_range_mat, y_range_mat)
    # low_res2 = torch.from_numpy(np.asarray(low_res2)).to(device).type(torch.complex64) # matlab -> numpy -> torch tensor
    # low_res2 = torch.transpose(low_res2, 0, 1)

    #print(torch.max(torch.abs(new_rcs)) / torch.max(torch.abs(original_rcs)))

    fig1 = plt.figure("Original scene")
    fig = utils.plot_scene(torch.abs(scene.cpu()), x_range.cpu(), y_range.cpu())

    fig2 = plt.figure("ISAR")
    fig = utils.plotcut_dB_in(low_res.cpu(), x_range.cpu(), y_range.cpu())

    fig3 = plt.figure("Original RCS")
    plt.imshow(torch.abs(rcs_temp).cpu().numpy(), cmap="inferno", extent=[-11.7891,11.7981,12,8], aspect='auto')
    plt.xlabel('Angle (Degrees)')
    plt.ylabel('Frequency (GHz)')
    plt.gca().invert_yaxis()
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Amplitude (Linear)', rotation=270, labelpad=20)

    fig4 = plt.figure("New RCS")
    plt.imshow(torch.abs(new_rcs).cpu().numpy(), cmap="inferno", extent=[-11.7891,11.7981,12,8], aspect='auto')
    plt.xlabel('Angle (Degrees)')
    plt.ylabel('Frequency (GHz)')
    plt.gca().invert_yaxis()
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Amplitude (Linear)', rotation=270, labelpad=20)

    # fig5 = plt.figure("ISAR new")
    # fig = utils.plotcut_dB_in(low_res2.cpu(), x_range.cpu(), y_range.cpu())

    plt.show()

    # fig1.savefig("scene.png")
    # fig2.savefig("isar.png")
    # fig3.savefig("original_rcs.png")
    # fig4.savefig("new_rcs.png")

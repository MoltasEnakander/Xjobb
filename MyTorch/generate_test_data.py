import os
import numpy as np
import torch
from tqdm import tqdm
import matlab.engine
from generate_scene import generate_scene, generate_single_point_scene, generate_line, generate_point_scene_same_row_closer
from generate_rcs import generate_rcs
import matplotlib.pyplot as plt
import utils

eng = matlab.engine.start_matlab()
eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner")
eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner\Moltas")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

scene_width = 256
scene_height = 256
max_shapes = 1  # maximum nr of objects that are allowed to be in the random scene

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

scenes = generate_point_scene_same_row_closer(scene_height, scene_width)

for i in tqdm(range(len(scenes))):    
    scene = torch.from_numpy(scenes[i]).to(device).type(torch.complex64)

    rcs = generate_rcs(scene, cal_range, ff, f, phi, x_range, y_range, device, method=1)    
    rcs_mat = matlab.double(rcs.detach().cpu().numpy().tolist(), is_complex=True)

    low_res = eng.generateISAR(rcs_mat, f_mat, phi_mat, hanning_flag, elev_angle, cal_range, ff, x_range_mat, y_range_mat)
    low_res = torch.from_numpy(np.asarray(low_res)).to(device).type(torch.complex64) # matlab -> numpy -> torch tensor
    low_res = torch.transpose(low_res, 0, 1)

    # pady = 0
    # padx = 0
    # if rcs.shape[0] < ny:
    #     pady = int((ny - rcs.shape[0]) / 2)

    # if rcs.shape[1] < nx:
    #     padx = int((nx - rcs.shape[1]) / 2)

    # zpad = torch.nn.ZeroPad2d((padx, padx, pady, pady))

    # echo_pad = zpad(rcs)

    # fft_isar = torch.fft.fftshift(torch.fft.fft2(echo_pad, norm="ortho"))

    fig1 = plt.figure(1)
    fig = utils.plot_scene(torch.abs(scene).cpu(), x_range.cpu(), y_range.cpu())
    #plt.imshow(torch.abs(scene).cpu(), cmap="inferno")
    #plt.gca().invert_yaxis()     

    # fig3 = plt.figure(3)
    # plt.imshow(torch.abs(rcs).cpu(), cmap="inferno")
    # plt.colorbar()

    fig2 = plt.figure(2)
    fig = utils.plotcut_dB_in(low_res.cpu(), x_range.cpu(), y_range.cpu())
    plt.show()

    # fig1.savefig("scene.png")
    # fig2.savefig("isar.png")
    # plt.figure(3)
    # plt.imshow(torch.abs(echo_pad).cpu())
    # plt.colorbar()

    # plt.figure(4)
    # fig = utils.plotcut_dB_in(low_res.cpu(), x_range.cpu(), y_range.cpu())

    # plt.figure(5)
    # fig = utils.plotcut_dB_in(fft_isar.cpu(), x_range.cpu(), y_range.cpu())

    plt.show()

    scene_path = os.path.join("test_data/scenes/", "scene_point_closer" + str(i) + ".pt")
    rcs_path = os.path.join("test_data/rcs/", "rcs_point_closer" + str(i) + ".pt")
    isar_path = os.path.join("test_data/isars/", "isar_point_closer" + str(i) + ".pt")

    torch.save(scene, scene_path)
    torch.save(rcs, rcs_path)
    torch.save(low_res, isar_path)

eng.quit()
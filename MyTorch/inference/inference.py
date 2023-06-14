import sys
sys.path.insert(1, r"C:\Users\Lovisa Nilsson\Desktop\xjobb\MyTorch")

import os
from time import time
from tqdm import tqdm
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch
from Networks import ISARNet
from custom_loss import CustomLoss
from CustomDataset import CustomDataset
from pynput.keyboard import Key, Listener
from generate_rcs import generate_rcs, generate_rcs_sparse
import torch.nn.functional as F
from generate_scene import generate_point_scene_same_col, generate_single_point_scene, generate_line, generate_scene
import matlab.engine
from SSIM import ssim
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import find_peaks

eng = matlab.engine.start_matlab()
eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner")
eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner\Moltas")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

network = ISARNet().to(device)

network = utils.load_model_state(network, "best_model.pth", AiQu=False)

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
x_range2 = torch.linspace(xmin, xmax, nx*1024)
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
    scene = generate_single_point_scene(scene_height, scene_width)
    scene = torch.from_numpy(scene).to(device).type(torch.complex64)

    rcs = generate_rcs(scene, cal_range, ff, f, phi, x_range, y_range, device, method=1)    
    rcs_mat = matlab.double(rcs.detach().cpu().numpy().tolist(), is_complex=True)

    low_res = eng.generateISAR(rcs_mat, f_mat, phi_mat, hanning_flag, elev_angle, cal_range, ff, x_range_mat, y_range_mat)
    low_res = torch.from_numpy(np.asarray(low_res)).to(device).type(torch.complex64) # matlab -> numpy -> torch tensor
    low_res = torch.transpose(low_res, 0, 1)

    low_res = low_res.view(1, 1, low_res.shape[-2], low_res.shape[-1])

    scale_factor = torch.max(torch.abs(low_res))   

    high_res = network(low_res/scale_factor)
    high_res = high_res.view(1, high_res.shape[-2], high_res.shape[-1])
    high_res = high_res*scale_factor

    scene = scene.unsqueeze(dim=0)
    ssim_ = ssim(torch.abs(scene), torch.abs(low_res), 255)
    print("SSIM abs -- Scene - LR ISAR: ", ssim_.detach().cpu().numpy())
    ssim_ = ssim(scene.real, low_res.real, 255)
    print("SSIM real -- Scene - LR ISAR: ", ssim_.detach().cpu().numpy())
    ssim_ = ssim(scene.imag, low_res.imag, 255)
    print("SSIM imag -- Scene - LR ISAR: ", ssim_.detach().cpu().numpy())

    print("-------------------------------------------------------------")
    ssim_ = ssim(torch.abs(scene), torch.abs(high_res), 255)
    print("SSIM abs -- Scene - HR ISAR: ", ssim_.detach().cpu().numpy())
    ssim_ = ssim(scene.real, high_res.real, 255)
    print("SSIM real -- Scene - HR ISAR: ", ssim_.detach().cpu().numpy())
    ssim_ = ssim(scene.imag, high_res.imag, 255)
    print("SSIM imag -- Scene - HR ISAR: ", ssim_.detach().cpu().numpy())


    plt.figure("Original scene")
    fig = utils.plot_scene(torch.abs(scene[0].cpu()), x_range.cpu(), y_range.cpu())

    plt.figure("Low-res isar 2D")
    fig = utils.plotcut_dB_in(low_res[0,0].cpu(), x_range.cpu(), y_range.cpu())

    plt.figure("High-res isar 2D")
    fig = utils.plotcut_dB_in(high_res[0].detach().cpu().numpy(), x_range.cpu(), y_range.cpu())    

    #plt.figure("High-res x isar 1D horizontal")
    # plt.figure("ISAR 1D horizontal")
    # fig = utils.plot_horizontal2(low_res[0,0].cpu(), x_range.cpu(), high_res[0].detach().cpu(), x_range.cpu())

    plt.figure("ISAR 1D horizontal normalised")
    fig, row, row2 = utils.plot_horizontal2(low_res[0,0].cpu(), x_range.cpu(), (high_res[0] / torch.max(torch.abs(high_res[0])) * torch.max(torch.abs(low_res[0,0]))).detach().cpu(), x_range.cpu())

    low_res_h = low_res[0,0,row,:].clone()
    high_res_h = high_res[0, row2,:].clone() / torch.max(torch.abs(high_res[0])) * torch.max(torch.abs(low_res[0,0]))

    cs_lr = CubicSpline(x_range.cpu().numpy(), torch.abs(low_res_h).cpu().numpy())
    cs_hr = CubicSpline(x_range.cpu().numpy(), torch.abs(high_res_h).detach().cpu().numpy())
    #interp = interp1d(x_range.cpu().numpy(), torch.abs(low_res_h).cpu().numpy())

    lr_upsample = cs_lr(x_range2.cpu())
    hr_upsample = cs_hr(x_range2.cpu())

    lr_max = np.max(np.abs(lr_upsample))
    hr_max = np.max(np.abs(hr_upsample))

    print(lr_max)
    print(hr_max)

    lr_upsample_half = lr_upsample - lr_max/2
    hr_upsample_half = hr_upsample - hr_max/2

    lr_halfs = np.argmin(lr_upsample_half)
    hr_halfs = np.argmin(hr_upsample_half)

    lr_upsample_half2 = lr_upsample_half.copy()
    lr_upsample_half2[np.abs(lr_upsample_half2)>0.05] = 1

    hr_upsample_half2 = hr_upsample_half.copy()
    hr_upsample_half2[np.abs(hr_upsample_half2)>0.05] = 1

    lr_upsample_half2 = np.abs(lr_upsample_half2 - 1)
    hr_upsample_half2 = np.abs(hr_upsample_half2 - 1)

    lr_peaks = find_peaks(lr_upsample_half2)
    hr_peaks = find_peaks(hr_upsample_half2)

    print(lr_peaks)
    print(hr_peaks)

    lr_peaks_x = x_range2.cpu().numpy()[lr_peaks[0]]
    lr_width = lr_peaks_x[1] - lr_peaks_x[0]
    print("Low-res width: ", lr_width)
    hr_peaks_x = x_range2.cpu().numpy()[hr_peaks[0]]
    hr_width = hr_peaks_x[1] - hr_peaks_x[0]    
    print("High-res width: ", hr_width)

    plt.figure("Test")
    fig = plt.plot(x_range.cpu(), torch.abs(low_res_h).cpu(), label="Conventional")
    plt.plot(x_range.cpu(), torch.abs(high_res_h).detach().cpu(), label="P-L0-CNN")
    plt.plot(x_range2.cpu(), cs_lr(x_range2.cpu()), label="Conventional (I-CS)")
    plt.plot(x_range2.cpu(), cs_hr(x_range2.cpu()), label="P-L0-CNN (I-CS)")
    plt.xlabel('Cross-range (m)')
    plt.ylabel('Amplitude (Linear)')
    plt.legend(loc="upper left")

    plt.figure("Test2")
    #fig = plt.plot(x_range.cpu(), torch.abs(low_res_h).cpu(), label="Conventional")
    #plt.plot(x_range.cpu(), torch.abs(high_res_h).detach().cpu(), label="P-L0-CNN")
    plt.plot(x_range2.cpu(), np.abs(lr_upsample_half2), label="Spline_lr")
    plt.plot(x_range2.cpu(), np.abs(hr_upsample_half2), label="Spline_hr")
    plt.xlabel('Cross-range (m)')
    plt.ylabel('Amplitude (Linear)')
    plt.legend(loc="upper left")

    # plt.figure("ISAR 1D vertical")
    # fig = utils.plot_vertical2(low_res[0,0].cpu(), y_range.cpu(), high_res[0].detach().cpu(), y_range.cpu())

    plt.figure("ISAR 1D vertical normalised")
    fig = utils.plot_vertical2(low_res[0,0].cpu(), y_range.cpu(), (high_res[0] / torch.max(torch.abs(high_res[0])) * torch.max(torch.abs(low_res[0,0]))).detach().cpu(), y_range.cpu())

    plt.show()

# from torch.utils.data import DataLoader
# scene_path = os.path.join('data256_256/', 'scenes/')
# rcs_path = os.path.join('data256_256/', 'rcs/')
# isar_path = os.path.join('data256_256/', 'isars/')
# batch_size = 1

# dataset = CustomDataset(scene_path, rcs_path, isar_path)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# for batch_idx, (scene, rcs, isar) in enumerate((dataloader)):
#     torch.cuda.empty_cache()
#     scene = scene.to(device).detach()
#     original_rcs = rcs.to(device).detach()   # rcs generated from the scene
#     low_res = isar.to(device)  # isar image generated from the rcs                        

#     # low_res = torch.abs(low_res)**(1/3) * torch.exp(1j * torch.angle(low_res))
#     low_res = low_res.view(batch_size, 1, low_res.shape[-2], low_res.shape[-1])         

#     high_res = network(low_res)
#     high_res = high_res.view(batch_size, high_res.shape[-2], high_res.shape[-1])

#     high_res = high_res/torch.max(torch.abs(high_res))

#     plt.figure("Original scene")
#     fig = utils.plot_scene(torch.abs(scene[0].cpu()), x_range.cpu(), y_range.cpu())

#     plt.figure("Low-res isar 2D")
#     fig = utils.plotcut_dB_in(low_res[0,0].cpu(), x_range.cpu(), y_range.cpu())

#     plt.figure("High-res isar 2D")
#     fig = utils.plotcut_dB_in(high_res[0].detach().cpu().numpy(), x_range.cpu(), y_range.cpu())    

#     #plt.figure("High-res x isar 1D horizontal")
#     plt.figure("ISAR 1D horizontal")
#     fig = utils.plot_horizontal2(low_res[0,0].cpu(), x_range.cpu(), high_res[0].detach().cpu(), x_range.cpu())

#     plt.figure("ISAR 1D horizontal normalised")
#     fig = utils.plot_horizontal2(low_res[0,0].cpu(), x_range.cpu(), (high_res[0] / torch.max(torch.abs(high_res[0])) * torch.max(torch.abs(low_res[0,0]))).detach().cpu(), x_range.cpu())

#     plt.show()
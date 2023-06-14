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
from CustomDataset2 import CustomDataset
from pynput.keyboard import Key, Listener
from generate_rcs import generate_rcs, generate_rcs_sparse, generate_rcs2
import torch.nn.functional as F
from generate_scene import generate_point_scene_same_row_closer2
import matlab.engine
from SSIM import ssim
from findpeaks import findpeaks
import scipy.ndimage.measurements as scnm

eng = matlab.engine.start_matlab()
eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner")
eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner\Moltas")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

network = ISARNet().to(device)

network = utils.load_model_state(network, "best_model.pth", AiQu=False)

network.eval()

# xmax = 1
# xmin = -xmax
# nx = scene_width
# ymax = 1
# ymin = -ymax
# ny = scene_height
# x_range = torch.linspace(xmin,xmax,nx).to(device)
# y_range = torch.linspace(ymin,ymax,ny).to(device)

scene_width = 4096
scene_height = 4096
max_shapes = 4  # maximum nr of objects that are allowed to be in the random scene
scene_x_range = torch.linspace(-1,1,scene_width).to(device)
scene_y_range = torch.linspace(-1,1,scene_height).to(device)

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
nx = 256
ymax = 1
ymin = -ymax
ny = 256
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

scenes = generate_point_scene_same_row_closer2(scene_height, scene_width, 120)

delta_x_scene = 1/2048
delta_x_isar = 1/128

distances_between = np.expand_dims(np.array([x*delta_x_scene for x in range(200, 0, -1)]), axis=0)
radians = np.expand_dims(np.array([x*np.pi/180 for x in range(0, 180, 200)]), axis=0)

x_positions = np.transpose(np.transpose(distances_between) * np.cos(radians))
y_positions = np.transpose(np.transpose(distances_between) * -np.sin(radians))

intensities = np.sqrt(0.5**2 + 0.5**2) * np.ones((2))
intensities = torch.from_numpy(intensities).to(device)

low_res_errors = []
high_res_errors = []
distances = []
ones = []

fp = findpeaks(method='mask')

lr_separabilities = np.empty((radians.size, distances_between.size))
hr_separabilities = np.empty((radians.size, distances_between.size))

for i in tqdm(range(radians.size)):
    lr_separability = np.empty(distances_between.size)
    hr_separability = np.empty(distances_between.size)
    check_lr = True
    check_hr = True

    for j in range(distances_between.size):
        x_pos = np.array([x_positions[i, j], 0])
        y_pos = np.array([y_positions[i, j], 0])

        x_pos = torch.from_numpy(x_pos).to(device)
        y_pos = torch.from_numpy(y_pos).to(device)

        rcs = generate_rcs2(x_pos, y_pos, intensities, cal_range, ff, f, phi, x_range, y_range, device, method=1)    
        rcs_mat = matlab.double(rcs.detach().cpu().numpy().tolist(), is_complex=True)

        low_res = eng.generateISAR(rcs_mat, f_mat, phi_mat, hanning_flag, elev_angle, cal_range, ff, x_range_mat, y_range_mat)
        low_res = torch.from_numpy(np.asarray(low_res)).to(device).type(torch.complex64) # matlab -> numpy -> torch tensor
        low_res = torch.transpose(low_res, 0, 1)
        low_res = low_res.view(1, low_res.shape[0], low_res.shape[1])

        scale_factor = torch.max(torch.abs(low_res))   

        high_res = network(low_res/scale_factor)
        high_res = high_res.view(1, high_res.shape[-2], high_res.shape[-1])
        high_res = high_res*scale_factor

        high_res_norm = high_res[0] / torch.max(torch.abs(high_res[0])) * torch.max(torch.abs(low_res[0]))

        high_res_thresh = torch.clone(torch.abs(high_res_norm))
        low_res_thresh = torch.clone(torch.abs(low_res[0]))

        high_res_thresh[torch.abs(high_res_thresh) < 0.3] = 0
        low_res_thresh[torch.abs(low_res_thresh) < 0.3] = 0
        
        # row2 = np.unravel_index(torch.argmax(torch.abs(low_res_thresh.cpu())), low_res_thresh.shape)[0]
        # row3 = np.unravel_index(torch.argmax(torch.abs(high_res_thresh.detach().cpu())), high_res_norm.shape)[0]

        # find peaks
        #scene_np = torch.abs(scene).cpu().numpy()
        # low_res_np = low_res_thresh.cpu().numpy()
        # high_res_np = high_res_thresh.detach().cpu().numpy()

        # lr_peaks = fp.fit(low_res_np)
        # hr_peaks = fp.fit(high_res_np)

        # lr_peaks = lr_peaks['Xdetect'].astype(int)
        # hr_peaks = hr_peaks['Xdetect'].astype(int)

        # grouped, ng = scnm.label(lr_peaks, np.ones((3,3)))
        # out = np.empty(ng+1, int)
        # out[grouped.reshape(-1)] = np.arange(lr_peaks.size)
        # lr_peaks2 = np.bincount(out[1:], None, lr_peaks.size).reshape(lr_peaks.shape)

        # grouped, ng = scnm.label(hr_peaks, np.ones((3,3)))
        # out = np.empty(ng+1, int)
        # out[grouped.reshape(-1)] = np.arange(hr_peaks.size)
        # hr_peaks2 = np.bincount(out[1:], None, hr_peaks.size).reshape(hr_peaks.shape)

        # print(np.count_nonzero(lr_peaks2))
        # print(np.count_nonzero(hr_peaks2))
        # print('___________________________')

        # if np.count_nonzero(lr_peaks2) >= 2 and check_lr:
        #     lr_separability[j] = 1
        #     # if j > 25:
        #     #     plt.figure(1)
        #     #     plt.imshow(lr_peaks2)
        #     #     plt.figure("Low-res isar 2D")
        #     #     fig = utils.plotcut_dB_in(low_res[0].cpu(), x_range.cpu(), y_range.cpu())
        #     #     plt.show()
        #     if np.count_nonzero(lr_peaks2) > 2:
        #         plt.figure(1)
        #         plt.imshow(lr_peaks2)
        #         plt.figure("Low-res isar 2D")
        #         fig = utils.plotcut_dB_in(low_res[0].cpu(), x_range.cpu(), y_range.cpu())
        #         plt.show()

        # else:
        #     check_lr = False
        #     lr_separability[j] = 0
        #     # plt.figure(1)
        #     # plt.imshow(lr_peaks)
        #     # plt.figure("Low-res isar 2D")
        #     # fig = utils.plotcut_dB_in(low_res[0].cpu(), x_range.cpu(), y_range.cpu())
        #     # plt.show()
        # if np.count_nonzero(hr_peaks2) >= 2 and check_hr:
        #     hr_separability[j] = 1
        #     if np.count_nonzero(hr_peaks2) > 2:
        #         plt.figure(1)
        #         plt.imshow(hr_peaks2)
        #         plt.figure("High-res isar 2D")
        #         fig = utils.plotcut_dB_in(high_res[0].detach().cpu(), x_range.cpu(), y_range.cpu())
        #         plt.figure("ISAR 1D horizontal")
        #         fig = utils.plot_horizontal2(low_res[0].cpu(), x_range.cpu(), high_res_norm.detach().cpu(), x_range.cpu())
        #         plt.show()

        # else:
        #     # plt.figure(3)
        #     # plt.imshow(hr_peaks)
        #     # plt.figure("High-res isar 2D")
        #     # fig = utils.plotcut_dB_in(high_res[0].detach().cpu().numpy(), x_range.cpu(), y_range.cpu())
        #     # plt.show()
        #     check_hr = False
        #     hr_separability[j] = 0

        # plt.figure("Low-res isar 2D")
        # fig = utils.plotcut_dB_in(low_res[0].cpu(), x_range.cpu(), y_range.cpu())

        # plt.figure("High-res isar 2D")
        # fig = utils.plotcut_dB_in(high_res[0].detach().cpu().numpy(), x_range.cpu(), y_range.cpu()) 

        #plt.figure("ISAR 1D horizontal normalised threshed")
        #fig = utils.plot_vertical2(low_res[0].cpu(), x_range.cpu(), high_res[0].detach().cpu(), x_range.cpu())

        str_number = '0' * 3
        str_number = str_number + str(j)
        str_number = str_number[len(str(j)):]
        lbl = 'c' + str_number + '_'

        fname = lbl + "closer_dist_" + str(distances_between[0,j])  + ".png"

        #plt.figure("ISAR 1D vertical normalised")
        utils.plot_horizontal2(low_res[0].cpu(), x_range.cpu(), (high_res[0] / torch.max(torch.abs(high_res[0])) * torch.max(torch.abs(low_res[0]))).detach().cpu(), x_range.cpu())

        plt.savefig("../Figurer/bildspel/" + fname)
        plt.clf()

    # lr_separabilities[i, :] = lr_separability
    # hr_separabilities[i, :] = hr_separability
        #scene_peaks = np.r_[True, scene_np[row,:][1:] > scene_np[row,:][:-1]] & np.r_[scene_np[row,:][:-1] > scene_np[row,:][1:], True]
        # low_res_peaks = np.r_[True, low_res_np[row2,:][1:] > low_res_np[row2,:][:-1]] & np.r_[low_res_np[row2,:][:-1] > low_res_np[row2,:][1:], True]
        # high_res_peaks = np.r_[True, high_res_np[row3,:][1:] > high_res_np[row3,:][:-1]] & np.r_[high_res_np[row3,:][:-1] > high_res_np[row3,:][1:], True]

        #scene_peaks_ind = np.where(scene_peaks)
        # low_res_peaks_ind = np.where(low_res_peaks)
        # high_res_peaks_ind = np.where(high_res_peaks)

        # convert peak indices to positions
        #scene_peaks_pos = scene_x_range[scene_peaks_ind]
        # low_res_peaks_pos = x_range[low_res_peaks_ind]
        # high_res_peaks_pos = x_range[high_res_peaks_ind]

        #distance_between_scene_peaks = torch.abs(scene_peaks_pos[0] - scene_peaks_pos[1])
        # distance_between_lr_peaks = torch.abs(low_res_peaks_pos[0] - low_res_peaks_pos[-1])
        # distance_between_hr_peaks = torch.abs(high_res_peaks_pos[0] - high_res_peaks_pos[-1])

        # print(distance_between_scene_peaks.cpu().numpy())
        # print(distance_between_lr_peaks.cpu().numpy())
        # print(distance_between_lr_peaks.cpu().numpy() / distance_between_scene_peaks.cpu().numpy())
        # print(distance_between_hr_peaks.cpu().numpy())    
        # print(distance_between_hr_peaks.cpu().numpy() / distance_between_scene_peaks.cpu().numpy())
        # print("---------------------")

        # low_res_errors.append(round(distance_between_lr_peaks.cpu().numpy() / distance_between_scene_peaks.cpu().numpy()))
        # high_res_errors.append(round(distance_between_hr_peaks.cpu().numpy() / distance_between_scene_peaks.cpu().numpy()))
        # distances.append(distance_between_scene_peaks.cpu().numpy())
        

    # low_res_error = torch.sum(torch.abs(scene_peaks_pos - low_res_peaks_pos))
    # high_res_error = torch.sum(torch.abs(scene_peaks_pos - high_res_peaks_pos))
    # distance = torch.abs(scene_peaks_pos[0] - scene_peaks_pos[1])

    # low_res_errors.append(low_res_error.cpu().numpy())
    # high_res_errors.append(high_res_error.cpu().numpy())
    # distances.append(distance.cpu().numpy())


    # plt.figure("Original scene")
    # fig = utils.plot_scene(torch.abs(scene.cpu()), scene_x_range.cpu(), scene_y_range.cpu())

    # plt.figure("Low-res isar 2D")
    # fig = utils.plotcut_dB_in(low_res[0].cpu(), x_range.cpu(), y_range.cpu())

    # plt.figure("High-res isar 2D")
    # fig = utils.plotcut_dB_in(high_res[0].detach().cpu().numpy(), x_range.cpu(), y_range.cpu())    

    #plt.figure("High-res x isar 1D horizontal")
    # plt.figure("ISAR 1D horizontal")
    # fig = utils.plot_horizontal2(low_res[0].cpu(), x_range.cpu(), high_res[0].detach().cpu(), x_range.cpu())

    # plt.figure("ISAR 1D horizontal normalised")
    # fig = utils.plot_horizontal3(low_res[0].cpu(), x_range.cpu(), (high_res[0] / torch.max(torch.abs(high_res[0])) * torch.max(torch.abs(low_res[0]))).detach().cpu(), x_range.cpu(), scene.cpu(), scene_x_range.cpu())

    # plt.figure("ISAR 1D horizontal normalised threshed")
    # fig = utils.plot_horizontal3(low_res_thresh.cpu(), x_range.cpu(), high_res_thresh.detach().cpu(), x_range.cpu(), scene.cpu(), scene_x_range.cpu())

    # plt.figure("ISAR 1D horizontal normalised peaks")
    # fig = utils.plot_horizontal3_2(torch.from_numpy(low_res_peaks.astype(int)), x_range.cpu(), torch.from_numpy(high_res_peaks.astype(int)), x_range.cpu(), torch.from_numpy(scene_peaks.astype(int)), scene_x_range.cpu())

    # plt.figure("Positional errors")
    # plt.plot(distances, high_res_errors, label="High-res")
    # plt.plot(distances, low_res_errors, label="Low-res")
    # plt.plot(distances, ones, linestyle='dashed')
    # plt.xlabel('Distance between points')
    # plt.ylabel('Positional error')
    # plt.legend(loc="upper right")

    # plt.figure("ISAR 1D vertical")
    # fig = utils.plot_vertical2(low_res[0].cpu(), y_range.cpu(), high_res[0].detach().cpu(), y_range.cpu())

    # plt.figure("ISAR 1D vertical normalised")
    # fig = utils.plot_vertical2(low_res[0].cpu(), y_range.cpu(), (high_res[0] / torch.max(torch.abs(high_res[0])) * torch.max(torch.abs(low_res[0]))).detach().cpu(), y_range.cpu())

    #plt.show()

# lr_separabilities_sum = np.sum(lr_separabilities, axis=0) / radians.size
# hr_separabilities_sum = np.sum(hr_separabilities, axis=0) / radians.size

# plt.figure("Separability")
# plt.plot(distances_between[0, :], lr_separabilities_sum, label="Conventional")
# plt.plot(distances_between[0, :], hr_separabilities_sum, label="P-L0_CNN")
# plt.yticks([1, 0])
# # plt.plot(distances, ones, linestyle='dashed')
# plt.xlabel('Distance between points (m)')
# plt.ylabel('Separability')
# plt.legend(loc="upper right")
# plt.ylim([0, 1.5])
# # plt.figure("Separability")
# # plt.plot(distances, low_res_errors, label="Low-res")
# # plt.plot(distances, high_res_errors, label="High-res")
# # # plt.plot(distances, ones, linestyle='dashed')
# # plt.xlabel('Distance between points (m)')
# # plt.ylabel('Separability')
# # plt.legend(loc="upper right")
# # plt.ylim([0, 1.5])
# plt.show()

eng.quit()



import sys
sys.path.insert(1, r"C:\Users\Lovisa Nilsson\Desktop\xjobb\MyTorch")

import os
from time import time
from tqdm import tqdm
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from CustomDataset import CustomDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xmax = 1
xmin = -xmax
nx = 256
ymax = 1
ymin = -ymax
ny = 256
x_range = torch.linspace(xmin,xmax,nx).to(device)
y_range = torch.linspace(ymin,ymax,ny).to(device)

from torch.utils.data import DataLoader
scene_path = os.path.join('largedataset/', 'scenes1/')
rcs_path = os.path.join('largedataset/', 'rcs1/')
isar_path = os.path.join('largedataset/', 'isars1/')
batch_size = 1

dataset = CustomDataset(scene_path, rcs_path, isar_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

complexities = []

files = os.listdir(scene_path)

# calculate complexities and add to list
for batch_idx, (scene, rcs, isar) in enumerate(tqdm(dataloader)):
    # if "shape" not in files[batch_idx]:
    #     continue
    # print(files[batch_idx])
    scene = scene.squeeze().to(device).detach()   # scene
    isar = isar.squeeze().to(device).detach()   # scene


    #original_rcs = rcs.squeeze().to(device).detach()   # rcs generated from the scene
    #low_res = isar.squeeze().to(device).detach()  # isar image generated from the rcs

    plt.figure("Original scene")
    fig = utils.plot_scene(torch.abs(scene.cpu()), x_range.cpu(), y_range.cpu())

    plt.figure("Low-res isar 2D")
    fig = utils.plotcut_dB_in(isar.cpu(), x_range.cpu(), y_range.cpu())

    # plt.figure("Original RCS")
    # plt.imshow(torch.abs(original_rcs).detach().cpu().numpy(), cmap="inferno")
    # plt.colorbar()

    plt.show()
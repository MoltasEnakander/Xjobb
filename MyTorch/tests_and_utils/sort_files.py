import os
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
from CustomDataset import CustomDataset
from generate_rcs import generate_rcs
import matlab.engine
from generate_scene import generate_scene

from torch.utils.data import DataLoader
scene_path = os.path.join('largedataset/', 'scenes/')
rcs_path = os.path.join('largedataset/', 'rcs/')
isar_path = os.path.join('largedataset/', 'isars/')
batch_size = 1

dataset = CustomDataset(scene_path, rcs_path, isar_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

complexities = []

# calculate complexities and add to list
for batch_idx, (_, rcs, isar) in enumerate(tqdm(dataloader)):  
    original_rcs = rcs.squeeze().to(device).detach()   # rcs generated from the scene
    low_res = isar.squeeze().to(device).detach()  # isar image generated from the rcs

    rcs_norm = original_rcs / torch.max(torch.abs(low_res))
    diffx = torch.max(torch.abs(torch.diff(rcs_norm, dim=1)))
    diffy = torch.max(torch.abs(torch.diff(rcs_norm, dim=0)))
    complexity = torch.sqrt(diffx**2 + diffy*2)

    complexities.append((complexity, batch_idx))

# sort the list
sorted_complexities = sorted(complexities, key=lambda comp:comp[0])

# order files
# sorted_complexities[:, 1] contains the order in the current file structure
# rename file i=sorted_complexities[i, 1] by adding a 'comp_j' infront where j is the iterator

indices = np.array([])
nr_digits = len(str(len(os.listdir(scene_path))))
for j in tqdm(range(len(os.listdir(scene_path)))):
    str_number = '0' * nr_digits
    str_number = str_number + str(j)
    str_number = str_number[len(str(j)):]
    lbl = 'c' + str_number + '_'
    idx = sorted_complexities[j][1] # find idx file to rename    
    relative_idx = np.sum(indices > idx)

    # rename scene
    curr_scene_name = os.listdir(scene_path)[idx + relative_idx] # file to rename
    curr_name = scene_path + curr_scene_name
    new_name = scene_path + lbl + curr_scene_name
    os.rename(curr_name, new_name)

    # rename rcs
    curr_rcs_name = os.listdir(rcs_path)[idx + relative_idx] # file to rename
    curr_name = rcs_path + curr_rcs_name
    new_name = rcs_path + lbl + curr_rcs_name
    os.rename(curr_name, new_name)

    # rename isar
    curr_isar_name = os.listdir(isar_path)[idx + relative_idx] # file to rename
    curr_name = isar_path + curr_isar_name
    new_name = isar_path + lbl + curr_isar_name
    os.rename(curr_name, new_name)    

    indices = np.append(indices, idx)

print("hej")

    

    


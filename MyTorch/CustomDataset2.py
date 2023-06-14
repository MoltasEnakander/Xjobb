import os

import torch
from torch.utils.data import Dataset
import pandas as pd

class CustomDataset(Dataset):
    """ Class for dataset containing scenes, rcs measurements for the scene and the low-res ISAR image created from the rcs measurements """

    def __init__(self, scene_dir, rcs_dir, isar_dir, transform=None):
        self.scene_frame = create_frame(scene_dir)
        self.scene_dir = scene_dir
        self.rcs_frame = create_frame(rcs_dir)
        self.rcs_dir = rcs_dir
        self.isar_frame = create_frame(isar_dir)
        self.isar_dir = isar_dir
        self.transform = transform

    def __len__(self):
        return len(self.isar_frame)

    def __getitem__(self, index):
        scene_path = os.path.join(self.scene_dir, self.scene_frame.iloc[index, 0])
        rcs_path = os.path.join(self.rcs_dir, self.rcs_frame.iloc[index, 0])
        isar_path = os.path.join(self.isar_dir, self.isar_frame.iloc[index, 0])

        scene = torch.load(scene_path) 
        rcs = torch.load(rcs_path)
        isar = torch.load(isar_path)

        return scene, rcs, isar


def create_frame(dir):
    frame = pd.DataFrame()
    file_list = os.listdir(dir)
    file_list.sort()
    for item in file_list:
        frame = pd.concat([frame, pd.DataFrame([[item, 0]])])
    return frame


        
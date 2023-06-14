import os
import argparse
from time import time
from tqdm import tqdm
import utils
import numpy as np
import torch
from Networks import ISARNet
from custom_loss import CustomLoss
from CustomDataset import CustomDataset
from generate_rcs import generate_rcs_sparse, generate_multiple_rcs_sparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

def main_program():
    parser = argparse.ArgumentParser('Args')
    parser.add_argument('--dataset', '-d', type=str, default="largedataset/")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from torch.utils.data import DataLoader, random_split

    for i in range(1):
        if i == 0: # first train with subset of dataset
            #scene_path = os.path.join('/data/', args.dataset, 'scenes1/')
            rcs_path = os.path.join(args.dataset, 'rcs1/')
            isar_path = os.path.join(args.dataset, 'isars1/')

            dataset1 = CustomDataset(rcs_path, isar_path)

            test_abs = int(len(dataset1) * 0.85)
            train_subset, val_subset = random_split(
                dataset1, [test_abs, len(dataset1) - test_abs])

        trainloader = DataLoader(train_subset, batch_size=1, shuffle=True, num_workers=0)

        incorrect_pairs = 0
        for batch_idx, (rcs, isar, rcs_path, isar_path) in enumerate(tqdm(trainloader)):
            if rcs_path[0].split('/')[-1][0:6] != isar_path[0].split('/')[-1][0:6]:
                incorrect_pairs += 1
                print("--------------------------------------------------------------------------------------")
                print(rcs_path)
                print(isar_path)
        
        print(incorrect_pairs)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == '__main__':
    set_seed()
    main_program()
import sys
sys.path.insert(1, r"C:\Users\Lovisa Nilsson\Desktop\xjobb\MyTorch")

from generate_scene import generate_point_scene_same_row_closer, generate_point_scene_same_col_closer
import utils
import matplotlib.pyplot as plt
import torch
import numpy as np

scenes = generate_point_scene_same_col_closer(256, 256)
x_range = torch.linspace(-1,1,256)
y_range = torch.linspace(-1,1,256)

for i in range(len(scenes)):
    plt.figure("Scene " + str(i))
    fig = utils.plot_scene(np.abs(scenes[i]), x_range.cpu(), y_range.cpu())
plt.show()
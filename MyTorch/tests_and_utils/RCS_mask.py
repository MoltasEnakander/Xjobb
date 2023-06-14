import torch
import matplotlib.pyplot as plt
import numpy as np


h = 256
w = 256
nbins = 2048

scene = np.zeros((h, w))

bins = torch.randperm(h * w)
bins = bins[0:nbins]
# convert to freqs and angles
freqs = bins // w
angles = bins % w

scene[freqs, angles] = 1

plt.figure(1)
plt.imshow(scene, cmap='binary', extent=[-11,11,8,12], aspect='auto')
plt.xlabel('Angle (Degrees)')
plt.ylabel('Frequency (GHz)')
plt.show()
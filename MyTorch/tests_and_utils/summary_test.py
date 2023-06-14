import sys
sys.path.insert(1, r"C:\Users\Lovisa Nilsson\Desktop\xjobb\MyTorch")

import torch
from Networks import ISARNet
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define networks
network = ISARNet().to(device)

input_size = (1, 64, 64)

summary(network, input_size, dtypes=[torch.complex64]*len(input_size))
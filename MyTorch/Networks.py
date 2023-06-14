import torch
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn.functional import relu, tanh, max_pool2d, avg_pool2d, dropout, dropout2d, interpolate, leaky_relu

def complex_tanh(input):
    return input.real.tanh().type(torch.complex64) + 1j * input.imag.tanh().type(torch.complex64)
    
def complex_relu(input):
    return relu(input.real).type(torch.complex64)+1j*relu(input.imag).type(torch.complex64)
    
def complex_lrelu(input):
    return leaky_relu(input.real, negative_slope=0.01).type(torch.complex64)+1j*leaky_relu(input.imag, negative_slope=0.01).type(torch.complex64)

class ISARNet(nn.Module):
    def __init__(self):
        super(ISARNet, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=48, kernel_size=(3,3),  stride=(1,1), padding='same', bias=False, dtype=torch.complex64)
        self.conv2 = Conv2d(in_channels=48, out_channels=96, kernel_size=(3,3), stride=(1,1), padding='same', bias=False, dtype=torch.complex64)
        self.conv3 = Conv2d(in_channels=96, out_channels=96, kernel_size=(3,3), stride=(1,1), padding='same', bias=False, dtype=torch.complex64)
        self.conv4 = Conv2d(in_channels=96, out_channels=192, kernel_size=(3,3), stride=(1,1), padding='same', bias=False, dtype=torch.complex64)
        self.conv5 = Conv2d(in_channels=192, out_channels=192, kernel_size=(3,3), stride=(1,1), padding='same', bias=False, dtype=torch.complex64)
        self.conv6 = Conv2d(in_channels=192, out_channels=192, kernel_size=(3,3), stride=(1,1), padding='same', bias=False, dtype=torch.complex64)
        self.conv7 = Conv2d(in_channels=192, out_channels=96, kernel_size=(3,3), stride=(1,1), padding='same', bias=False, dtype=torch.complex64)
        self.conv8 = Conv2d(in_channels=96, out_channels=48, kernel_size=(3,3), stride=(1,1), padding='same', bias=False, dtype=torch.complex64)
        self.conv9 = Conv2d(in_channels=48, out_channels=32, kernel_size=(3,3), stride=(1,1), padding='same', bias=False, dtype=torch.complex64)
        self.conv10 = Conv2d(in_channels=32, out_channels=16, kernel_size=(3,3), stride=(1,1), padding='same', bias=False, dtype=torch.complex64)
        self.conv11 = Conv2d(in_channels=16, out_channels=1, kernel_size=(3,3), stride=(1,1), padding='same', bias=False, dtype=torch.complex64)        

        # weight init
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.conv4.weight)
        torch.nn.init.xavier_uniform_(self.conv5.weight)
        torch.nn.init.xavier_uniform_(self.conv6.weight)
        torch.nn.init.xavier_uniform_(self.conv7.weight)
        torch.nn.init.xavier_uniform_(self.conv8.weight)
        torch.nn.init.xavier_uniform_(self.conv9.weight)
        torch.nn.init.xavier_uniform_(self.conv10.weight)
        torch.nn.init.xavier_uniform_(self.conv11.weight)

    # x represents our data
    def forward(self, x):
        x = self.conv1(x)
        x = complex_relu(x)

        x = self.conv2(x)
        x = complex_relu(x)

        x = self.conv3(x)
        x = complex_relu(x)

        x = self.conv4(x)
        x = complex_relu(x)

        x = self.conv5(x)
        x = complex_relu(x)

        x = self.conv6(x)
        x = complex_relu(x)

        x = self.conv7(x)
        x = complex_relu(x)

        x = self.conv8(x)
        x = complex_relu(x)

        x = self.conv9(x)
        x = complex_relu(x)

        x = self.conv10(x)
        x = complex_relu(x)

        x = self.conv11(x)

        return x
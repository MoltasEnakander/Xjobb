import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, y, y_echo, f, mu):
        # y is the generated image by the network
        # x is the image input to the network
        # y_echo is the result from generating echo from the high-res scene 
        # f is the original echo
        # mu is the threshold for the pseudo_l0 norm
        # lambda is the regularisation term
        # loss defined as: L = ||f - A*y||_2^2 + lambda*||y||_0^f
        # (||...||_2^2 is the squared l2-norm, ||...||_0^f is the pseudo_l0-norm)
        # calculate losses and divide by batch size        
        pseudo_l0_loss = self.pseudo_l0(y, mu) / y.shape[0]
        diff = (y_echo/22.8 - f)
        l2_loss = torch.abs(diff) @ torch.transpose(torch.abs(torch.conj(diff)), -1, -2)
        l2_loss = torch.trace(l2_loss) / y.shape[0]
        return pseudo_l0_loss, l2_loss
    
    """ Calculate the pseudo l0-norm of signal y"""
    def pseudo_l0(self, y, mu):
        # pseudo l0-norm is defined as 
        #                            ( 1        if |y_i| >= mu
        # l0 = sum all a_i for a_i = ( |y_i|/mu if 0 < |y_i| < mu
        #                            ( 0        others
        return torch.sum((torch.abs(y) >= mu) + ((torch.logical_and(torch.abs(y)>0,torch.abs(y)<mu)) / mu) * torch.abs(y))
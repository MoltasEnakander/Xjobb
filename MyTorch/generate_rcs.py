import torch
from ptsource import ptsource, ptsource_sparse, ptsource_sparse2

def generate_rcs_sparse(scene, calrange, ff, f, phi, x_range, y_range, nbins, device):
    # calrange is the distance to the center of the scene
    # ff indicates near-field (f=0) or far-field (f=1)
    # f contains frequencies (GHz)
    # phi contains angles (degrees)
    # x_range contain possible x-positions from xmin-xmax
    # y_range contain possible x-positions from ymin-ymax
    # method (= 0,1,2) decides how calculations in ptsource are performed

    # find indices of non-zero positions and convert to "real" positions in scene
    [y_ind, x_ind] = torch.where(torch.abs(scene) > 0)
    x_pos = x_range[x_ind]
    y_pos = y_range[y_ind]
    amp = scene[y_ind, x_ind] # amplitude of point source

    rcs = ptsource_sparse(x_pos, y_pos, amp, f, phi, calrange, ff, nbins, device) # generate echo from point sources
    return rcs

def generate_multiple_rcs_sparse(scenes, calrange, ff, f, phi, x_range, y_range, nbins, device):
    # calrange is the distance to the center of the scene
    # ff indicates near-field (f=0) or far-field (f=1)
    # f contains frequencies (GHz)
    # phi contains angles (degrees)
    # x_range contain possible x-positions from xmin-xmax
    # y_range contain possible x-positions from ymin-ymax
    # method (= 0,1,2) decides how calculations in ptsource are performed

    # find indices of non-zero positions and convert to "real" positions in scene
    [scene, y_ind, x_ind] = torch.where(torch.abs(scenes) > 0)    
    x_pos = x_range[x_ind]
    y_pos = y_range[y_ind]
    amp = scenes[scene, y_ind, x_ind] # amplitude of point source

    rcs = ptsource_sparse2(scenes.shape[0], x_pos, y_pos, amp, f, phi, calrange, ff, nbins, device) # generate echo from point sources
    return rcs

def generate_rcs(scene, calrange, ff, f, phi, x_range, y_range, device, method):
    # calrange is the distance to the center of the scene
    # ff indicates near-field (f=0) or far-field (f=1)
    # f contains frequencies (GHz)
    # phi contains angles (degrees)
    # x_range contain possible x-positions from xmin-xmax
    # y_range contain possible x-positions from ymin-ymax
    # method (= 0,1,2) decides how calculations in ptsource are performed

    # find indices of non-zero positions and convert to "real" positions in scene
    [y_ind, x_ind] = torch.where(torch.abs(scene) > 0)
    x_pos = x_range[x_ind]
    y_pos = y_range[y_ind]
    amp = scene[y_ind, x_ind] # amplitude of point source

    rcs = ptsource(x_pos, y_pos, amp, f, phi, calrange, ff, device, method) # generate echo from point sources
    return rcs


def generate_rcs2(x_pos, y_pos, amp, calrange, ff, f, phi, x_range, y_range, device, method):
    # calrange is the distance to the center of the scene
    # ff indicates near-field (f=0) or far-field (f=1)
    # f contains frequencies (GHz)
    # phi contains angles (degrees)
    # x_range contain possible x-positions from xmin-xmax
    # y_range contain possible x-positions from ymin-ymax
    # method (= 0,1,2) decides how calculations in ptsource are performed

    # find indices of non-zero positions and convert to "real" positions in scene
    #y_ind, x_ind] = torch.where(torch.abs(scene) > 0)
    #x_pos = x_ind * (1/2048) + (-1 + 1/4096)
    #y_pos = y_ind * (1/2048) + (-1 + 1/4096)
    #x_pos = x_range[x_ind]
    #y_pos = y_range[y_ind]
    #amp = scene[y_ind, x_ind] # amplitude of point source

    rcs = ptsource(x_pos, y_pos, amp, f, phi, calrange, ff, device, method) # generate echo from point sources
    return rcs
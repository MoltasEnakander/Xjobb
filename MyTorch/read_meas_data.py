import scipy.io
import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
import utils
import torch
import os

eng = matlab.engine.start_matlab()
eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner")
eng.addpath(r"C:\Users\Lovisa Nilsson\Desktop\xjobb\Rutiner\Moltas")

#mat = scipy.io.loadmat("meas_data/cylinder32mm0001_VV_gated.mat")
mat = scipy.io.loadmat("meas_data/rund_hh.mat")
overlay = scipy.io.loadmat("meas_data/rund_ov.mat")
# mat = scipy.io.loadmat("meas_data/bil_bmw_overlay.mat")

cal_range = mat['calrange'][0][0]
cal_range = cal_range.item()

rcs = mat['rcs_all']

theta_start = mat['thetastart'][0][0]
theta_stop = mat['thetastop'][0][0]
ntheta = mat['ntheta'][0][0]
fstart = mat['fstart'][0][0]
fstop = mat['fstop'][0][0]
nf = mat['nf'][0][0]

freqs = np.linspace(fstart, fstop, int(nf))
theta = np.linspace(theta_start, theta_stop, int(ntheta))
# cylinder
# theta = theta[842:961]
# freqs = freqs[26:82]
# rcs = rcs[26:82, 842:961]
# xmax = 1
# ymax = 1

# car
# theta = theta[9460:10640]
# rcs = rcs[:, 9460:10640]

# rak
# freqs = freqs[40:121]
# theta = theta[0: 81]
# rcs = rcs[40:121, 0:81]

# rund
freqs = freqs[40:121]
theta = theta[0: 81]
rcs = rcs[40:121, 0:81]

xmax = 1
ymax = 1
f_mat = matlab.double(freqs.tolist())
phi_mat = matlab.double(theta.tolist())
rcs_mat = matlab.double(rcs.tolist(), is_complex=True)

# xmax = 2.6
xmin = -xmax
nx = 256
# ymax = 2.6
ymin = -ymax
ny = 256
x_range = np.linspace(xmin,xmax,nx)
y_range = np.linspace(ymin,ymax,ny)
x_range_mat = matlab.double(x_range.tolist()) # create matlab object of x, both will be needed later on
y_range_mat = matlab.double(y_range.tolist()) # same but for y

hanning_flag = 0
elev_angle = 0
ff = 0

low_res = eng.generateISAR(rcs_mat, f_mat, phi_mat, hanning_flag, elev_angle, cal_range, ff, x_range_mat, y_range_mat)
low_res = np.asarray(low_res) # matlab -> numpy -> torch tensor
low_res = np.transpose(low_res)

scale_factor = np.max(np.abs(low_res))

rcs2 = rcs / scale_factor
rcs_mat2 = matlab.double(rcs2.tolist(), is_complex=True)

low_res2 = eng.generateISAR(rcs_mat2, f_mat, phi_mat, hanning_flag, elev_angle, cal_range, ff, x_range_mat, y_range_mat)
low_res2 = np.asarray(low_res2) # matlab -> numpy -> torch tensor
low_res2 = np.transpose(low_res2)

rcs_norm = rcs / torch.max(torch.abs(torch.from_numpy(low_res)))
diffx = torch.max(torch.abs(torch.diff(rcs_norm, dim=1)))
diffy = torch.max(torch.abs(torch.diff(rcs_norm, dim=0)))
complexity = torch.sqrt(diffx**2 + diffy**2)
print(complexity)

fig1 = plt.figure(1)
plt.imshow(np.abs(rcs), cmap="inferno")
plt.colorbar()
fig2 = plt.figure(2)
fig = utils.plotcut_dB_in(low_res, x_range, y_range)
utils.plot_overlay(overlay)
fig2 = plt.figure(3)
fig = utils.plotcut_dB_in(low_res2, x_range, y_range)
utils.plot_overlay(overlay)

plt.show()

eng.quit()



# rcs_path = os.path.join("meas_data/rcs/", "rcs_straightplane.pt")
# isar_path = os.path.join("meas_data/isars/", "isar_straightplane.pt")

# torch.save(torch.from_numpy(rcs).type(torch.complex64), rcs_path)
# torch.save(torch.from_numpy(low_res).type(torch.complex64), isar_path)

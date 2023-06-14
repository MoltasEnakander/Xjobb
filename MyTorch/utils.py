import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def save_model(model,name, AiQu=True):
    """ Save the state of the model """
    if AiQu:
        torch.save(model.state_dict(), os.path.join('/data/', "trained_models", name))
    else:
        torch.save(model.state_dict(), os.path.join("trained_models", name))


def load_model_state(model, name, AiQu=True):
    """ Loads the state of the model which is saved in *name* """
    model.cuda()
    if AiQu:
        state_dict = torch.load(os.path.join('/data/', "trained_models", name), map_location="cuda")
    else:
        state_dict = torch.load(os.path.join("trained_models", name), map_location="cuda")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def draw_line(scene, x, y, intensity):
    """ Draws a line by filling in the points in the scene corresponding to the 
        coordinates given by arrays x and y. If x or y exceeds bounds of the 
        scene then the line stops at the border """
    if not np.size(x) or not np.size(y):
        return scene

    if np.max(y) >= scene.shape[0]: # out of bounds
               if y[0] > y[-1]: # remove values at the front
                   ind_y = np.where(y >= scene.shape[0])
                   ind_y = np.max(ind_y)
                   y = y[ind_y + 1:]
                   x = x[ind_y + 1:]
               else: # remove values at the end
                   ind_y = np.where(y >= scene.shape[0])
                   ind_y = np.min(ind_y)
                   y = y[0:ind_y]
                   x = x[0:ind_y]

    if np.max(x) >= scene.shape[1]: # out of bounds
               if x[0] > x[-1]: # remove values at the front
                   ind_x = np.where(x >= scene.shape[1]) # ska kanske vara shape[0], men tror det borde vara såhär, (fel i gamla matlabkoden isåfall)
                   ind_x = np.max(ind_x)
                   x = x[ind_x + 1:]
                   y = y[ind_x + 1:]
               else: # remove values at the end
                   ind_x = np.where(x >= scene.shape[1]) # ska kanske vara shape[0], men tror det borde vara såhär, (fel i gamla matlabkoden isåfall)
                   ind_x = np.min(ind_x)
                   x = x[0:ind_x]
                   y = y[0:ind_x]
    
    scene[y, x] = intensity # draw line
    return scene


def plotcut_dB_in(isar, x, y, dynamic_range=50):
    """ Creates figure of isar in dB """
    # Code from Christer, slightly modified (plotcut_dB_in.m) 
    eps = 1e-40
    DY=(20*np.log10(np.abs(np.asarray(isar)) + eps))
    dx = 0
    dy = 0
    

    cmax = np.max(np.max(20*np.log10(np.abs(np.asarray(isar)) + eps)))
    cmin = cmax-dynamic_range

    x = np.asarray(x)
    y = np.asarray(y)

    fig = plt.pcolor(x-dx,y-dy,DY, cmap='inferno', vmin=cmin, vmax=cmax)
    plt.axis([np.min(x)-dx, np.max(x)-dx, np.min(y)-dy, np.max(y)-dy])
    plt.axis('square')

    plt.xlabel('Cross-range (m)')
    plt.ylabel('Down-range (m)')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Amplitude (Rel.A.U.)', rotation=270, labelpad=20)

    return fig


def plot_overlay(overlay, linewidth=0.2):
    drawing = overlay['ovpicture']
    xover = overlay['xover']
    yover = overlay['yover']
    fiover = overlay['fiover']

    x = drawing[:, 0] * np.cos(fiover * np.pi/180) + drawing[:, 1] * np.sin(fiover * np.pi/180)
    y = -drawing[:, 0] * np.sin(fiover * np.pi/180) + drawing[:, 1] * np.cos(fiover * np.pi/180)

    x = x + xover
    y = y + yover
    ha = plt.plot(x[0], y[0])
    plt.setp(ha, linewidth=linewidth, color='w')


def plot_scene(scene, x, y):
    dx = 0
    dy = 0
    x = np.asarray(x)
    y = np.asarray(y)
    scene[scene == 0] = float("NaN")
    fig = plt.pcolor(x-dx, y-dy, scene, cmap='jet')
    plt.axis([np.min(x)-dx, np.max(x)-dx, np.min(y)-dy, np.max(y)-dy])
    plt.axis('square')
    plt.xlabel('Cross-range (m)')
    plt.ylabel('Down-range (m)')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Amplitude (Linear)', rotation=270, labelpad=20)
    return fig


def plot_horizontal(scene, x):
    row = np.unravel_index(torch.argmax(torch.abs(scene)), scene.shape)[0]
    x = np.asarray(x)
    fig = plt.plot(x, torch.abs(scene[row, :]))
    plt.xlabel('Cross-range (m)')
    plt.ylabel('Amplitude')
    return fig


def plot_horizontal2(scene1, x1, scene2, x2):
    row = np.unravel_index(torch.argmax(torch.abs(scene1)), scene1.shape)[0]
    row2 = np.unravel_index(torch.argmax(torch.abs(scene2)), scene2.shape)[0]
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    fig = plt.plot(x1, torch.abs(scene1[row, :]), label="Conventional")
    plt.plot(x2, torch.abs(scene2[row2, :]), label="P-L0-CNN")
    plt.xlabel('Cross-range (m)')
    plt.ylabel('Amplitude (Linear)')
    plt.legend(loc="upper left")
    return fig, row, row2


def plot_horizontal3(scene1, x1, scene2, x2, scene3, x3):
    row = np.unravel_index(torch.argmax(torch.abs(scene1)), scene1.shape)[0]
    row2 = np.unravel_index(torch.argmax(torch.abs(scene2)), scene2.shape)[0]
    row3 = np.unravel_index(torch.argmax(torch.abs(scene3)), scene3.shape)[0]
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    fig = plt.plot(x1, torch.abs(scene1[row, :]), label="Conventional")
    plt.plot(x2, torch.abs(scene2[row2, :]), label="P-L0-CNN")
    plt.plot(x3, torch.abs(scene3[row3, :]), label="Scene")
    plt.xlabel('Cross-range (m)')
    plt.ylabel('Amplitude')
    plt.legend(loc="upper left")
    return fig


def plot_horizontal3_2(scene1, x1, scene2, x2, scene3, x3):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    fig = plt.plot(x1, torch.abs(scene1), label="Conventional")
    plt.plot(x2, torch.abs(scene2), label="P-L0-CNN")
    plt.plot(x3, torch.abs(scene3), label="Scene")
    plt.xlabel('Cross-range (m)')
    plt.ylabel('Amplitude')
    plt.legend(loc="upper left")
    return fig


def plot_vertical(scene, y):
    col = np.unravel_index(torch.argmax(torch.abs(scene)), scene.shape)[1]
    y = np.asarray(y)
    fig = plt.plot(y, torch.abs(scene[:, col]))
    plt.xlabel('Down-range (m)')
    plt.ylabel('Amplitude')
    return fig


def plot_vertical2(scene1, y1, scene2, y2):
    col1 = np.unravel_index(torch.argmax(torch.abs(scene1)), scene1.shape)[1]
    col2 = np.unravel_index(torch.argmax(torch.abs(scene2)), scene2.shape)[1]
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    fig = plt.plot(y1, torch.abs(scene1[:, col1]), label="Conventional")
    plt.plot(y2, torch.abs(scene2[:, col2]), label="P-L0-CNN")
    plt.xlabel('Down-range (m)')
    plt.ylabel('Amplitude (Linear)')
    plt.legend(loc="upper left")
    return fig, col1, col2

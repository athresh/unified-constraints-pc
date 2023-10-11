"Adapted from https://github.com/sahilsid/probabilistic-flow-circuits"
import numpy as np
import os
import torch
import errno
from PIL import Image
from scipy import stats
import matplotlib.pyplot as plt
from math import sqrt, ceil
import torchvision

def visualize_3d(model, dataset, save_dir, epoch=0):
    real_data = dataset.data.cpu().numpy()[np.random.choice(np.arange(0, len(dataset)), 1000)]
    gen_data = model.sample(len(real_data)).cpu().numpy()

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    (xmax, xmin), (ymax, ymin), (zmax, zmin) = (20.,-1.), (1.5,-1.5), (1.5,-1.5)
    plot3d(real_data, ax1, 0.25, ymin, ymax, xmin, xmax, zmin, zmax)
    ax1.set_title("Real Data", fontsize=12, fontweight='bold')

    plot3d(gen_data, ax2, 0.25, ymin, ymax, xmin, xmax, zmin, zmax)
    ax2.set_title(f"Generated Data \n Epoch: {epoch}", fontsize=12, fontweight='bold')
    plt.savefig(os.path.join(save_dir, f"{epoch}.png"), bbox_inches="tight")
    plt.close()
    # plt.show()


def plot3d(data, ax, alpha=0.25, ymin=-1, ymax=1, xmin=-1, xmax=1, zmin=-1, zmax=1, color=None):
    """
    Function to plot datapoints in 3D space. Takes as input a numpy array of size (N,3)
    and a matplotlib axis object on which to plot.
    """
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    xyz = np.vstack([x, y, z])
    z[z < zmin] = np.nan
    z[z > zmax] = np.nan
    y[y < ymin] = np.nan
    y[y > ymax] = np.nan
    x[x < xmin] = np.nan
    x[x > xmax] = np.nan

    if (color is None):
        density = stats.gaussian_kde(xyz)(xyz)
        idx = density.argsort()
        x, y, z, density = x[idx], y[idx], z[idx], density[idx]
        ax.scatter(x, y, z, c=density, alpha=alpha, s=25, cmap="rainbow")
    else:
        ax.scatter(x, y, z, c=color, alpha=alpha, s=25, cmap="rainbow")

    for ax_ in [ax.xaxis, ax.yaxis, ax.zaxis]:
        ax_.pane.set_edgecolor('r')
        ax_.pane.fill = False
        ax_.set_ticklabels([])
        for line in ax_.get_ticklines():
            line.set_visible(False)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.set_zlim(zmin, zmax)
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


def visualize_set_image(model, dataset, save_dir, epoch=0, h=28, w=28):
    real_data = dataset.data.cpu().numpy()[np.random.choice(np.arange(0, len(dataset)), 64)]
    gen_data = model.sample(len(real_data)).cpu().numpy()
    
    real_images = set_to_image(real_data, h=h, w=w) 
    real_grid = torchvision.utils.make_grid(real_images, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    torchvision.utils.save_image(real_grid, os.path.join(save_dir,"real_data.png"))
    
    gen_images = set_to_image(gen_data, h=h, w=w) 
    gen_grid = torchvision.utils.make_grid(gen_images, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    os.makedirs(os.path.join(save_dir, "generated"), exist_ok=True)
    torchvision.utils.save_image(gen_grid, os.path.join(save_dir, "generated", f"{epoch}.png"))
    
    samples = torch.cat([model.sample(64).detach() for _ in range(10)], dim=0)
    ll = torch.cat([model(samples[64*i:64*(i+1)]).detach() for i in range(10)], dim=0)
    top_ll, top_idx = torch.topk(ll.squeeze(), 64)
    top_samples = samples[top_idx].cpu().numpy()
    images = set_to_image(top_samples, h=28, w=28) 
    grid = torchvision.utils.make_grid(images, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    os.makedirs(os.path.join(save_dir, "best_samples"), exist_ok=True)
    torchvision.utils.save_image(grid, os.path.join(save_dir,"best_samples",f"{epoch}.png"))
    
    # plt.figure()
    # plt.imshow(grid.permute(1, 2, 0).detach().cpu().numpy())
    # plt.savefig(os.path.join(save_dir, "generated", f"{epoch}.png"), bbox_inches="tight")
    # plt.close()
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
    
def set_to_image(data, h=28, w=28, c=1):
    """
    Function to plot set of coordinates as an image.
    """
    images = []
    for i,coordinates in enumerate(data):
        image = np.zeros(c*h*w)
        
        # Set the pixels at the coordinates to 1
        for coord in coordinates:
            coord = coord.astype(int)
            
            if(coord >= h*w*c):
                continue
            
            if (coord >= 0).all():  # Ignore padding
                image[coord] = 1

        image = np.reshape(image,(c, h, w))
        images.append(torch.from_numpy(image))
    return images

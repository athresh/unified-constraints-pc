import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
import os
from pathlib import Path

SAVE_DIR = '../data/toy_3d'
PLOT_DIR = '../plots/toy_3d'

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

def make_helix(num_data=20000, num_helices=3, c=1, r=1, sd=0.1, test_size=0.5, save_data=False, visualize=False):
    x = np.linspace(0, 2 * num_helices * c * np.pi, num_data)
    y = r * np.sin(x)
    z = r * np.cos(x)
    helix = np.stack([x, y, z]).T
    noise = sd * np.random.normal(size=(num_data, 3))
    data = helix + noise
    data_val, data_tst = train_test_split(data, test_size=test_size, random_state=42)
    # data_tst, data_val = train_test_split(data_tst, test_size=0.5, random_state=42)
    if save_data:
        # Save the numpy files to the folder where they come from
        p = Path(SAVE_DIR)
        p.mkdir(parents=True, exist_ok=True)
        np.savetxt(SAVE_DIR + '/helix' + '.trn', data_trn)
        np.savetxt(SAVE_DIR + '/helix' + '.val', data_val)
        np.savetxt(SAVE_DIR + '/helix' + '.tst', data_tst)
    if visualize:
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')
        (xmax, xmin), (ymax, ymin), (zmax, zmin) = (20., -1.), (1.5, -1.5), (1.5, -1.5)
        plot3d(data_trn, ax1, 0.25, ymin, ymax, xmin, xmax, zmin, zmax)
        ax1.set_title("Train", fontsize=12, fontweight='bold')

        plot3d(data_val, ax2, 0.25, ymin, ymax, xmin, xmax, zmin, zmax)
        ax2.set_title("Val", fontsize=12, fontweight='bold')

        plot3d(data_tst, ax3, 0.25, ymin, ymax, xmin, xmax, zmin, zmax)
        ax3.set_title("Test", fontsize=12, fontweight='bold')
        plt.show()
    return data_trn, data_tst

def make_circles(num_data=20000, num_circles=2, c=10, r=1, sd=0.1, test_size=0.5, save_data=False):
    x = [i * c for i in range(num_circles)] * int(num_data/2)
    angle = np.linspace(0, 2 * np.pi, num_data)
    y = r * np.sin(angle)
    z = r * np.cos(angle)
    circle = np.stack([x,y,z]).T
    noise = np.append(np.zeros(shape=(num_data, 1)), sd * np.random.normal(size=(num_data, 2)), 1)
    data = circle + noise
    data_trn, data_tst = train_test_split(data, test_size=test_size, random_state=42)
    data_tst, data_val = train_test_split(data_tst, test_size=0.5, random_state=42)
    if save_data:
        # Save the numpy files to the folder where they come from
        p = Path(SAVE_DIR)
        p.mkdir(parents=True, exist_ok=True)
        np.savetxt(SAVE_DIR + '/circle' + '.trn', data_trn)
        np.savetxt(SAVE_DIR + '/circle' + '.val', data_val)
        np.savetxt(SAVE_DIR + '/circle' + '.tst', data_tst)
    return data_trn, data_tst

def make_helix_short(num_data=10000, num_helices_train=1, num_helices_test=2, c=1, r=1, sd=0.1, test_size=0.5, save_data=False, visualize=True):
    x = np.linspace(0, 2 * num_helices_train * c * np.pi, num_data)
    y = r * np.sin(x)
    z = r * np.cos(x)
    helix = np.stack([x, y, z]).T
    noise = sd * np.random.normal(size=(num_data, 3))
    data_trn = helix + noise

    x = np.linspace(0, 2 * num_helices_test * c * np.pi, num_data)
    y = r * np.sin(x)
    z = r * np.cos(x)
    helix = np.stack([x, y, z]).T
    noise = sd * np.random.normal(size=(num_data, 3))
    data_tst = helix + noise
    data_tst, data_val = train_test_split(data_tst, test_size=0.5, random_state=42)
    if save_data:
        p = Path(SAVE_DIR)
        p.mkdir(parents=True, exist_ok=True)
        # Save the numpy files to the folder where they come from
        np.savetxt(SAVE_DIR + '/helix_short' + '.trn', data_trn)
        np.savetxt(SAVE_DIR + '/helix_short' + '.val', data_val)
        np.savetxt(SAVE_DIR + '/helix_short' + '.tst', data_tst)
    if visualize:
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')
        (xmax, xmin), (ymax, ymin), (zmax, zmin) = (20., -1.), (1.5, -1.5), (1.5, -1.5)
        plot3d(data_trn, ax1, 0.25, ymin, ymax, xmin, xmax, zmin, zmax)
        ax1.set_title("Train", fontsize=12, fontweight='bold')

        plot3d(data_val, ax2, 0.25, ymin, ymax, xmin, xmax, zmin, zmax)
        ax2.set_title("Val", fontsize=12, fontweight='bold')

        plot3d(data_tst, ax3, 0.25, ymin, ymax, xmin, xmax, zmin, zmax)
        ax3.set_title("Test", fontsize=12, fontweight='bold')
        plt.show()
    return data_trn, data_tst

def make_helix_uneven(num_data=10000, num_helices_train=1, num_helices_test=2, c=1, r=1, sd=0.1, test_size=0.5, save_data=False, visualize=True):
    x = np.linspace(0, 2 * num_helices_train * c * np.pi, num_data)
    y = r * np.sin(x)
    z = r * np.cos(x)
    helix = np.stack([x, y, z]).T
    noise = sd * np.random.normal(size=(num_data, 3))
    data_trn = helix + noise

    # x = np.linspace(2 * num_helices_test/2 * c * np.pi, 2 * num_helices_test * c * np.pi, num_data)
    # y = r * np.sin(x)/2 + 1
    # z = r * np.cos(x)/2 - 1
    # helix = np.stack([x, y, z]).T
    # noise = sd * np.random.normal(size=(num_data, 3))
    # data_tst = helix + noise
    # data_tst, data_val = train_test_split(data_tst, test_size=0.5, random_state=42)
    x = [20 + i * c for i in range(2)] * int(num_data / 2)
    angle = np.linspace(0, 2 * np.pi, num_data)
    y = r * np.sin(angle)/2
    z = r * np.cos(angle)/2
    circle = np.stack([x, y, z]).T
    noise = np.append(np.zeros(shape=(num_data, 1)), sd * np.random.normal(size=(num_data, 2)), 1)
    data = circle + noise
    data_tst, data_val = train_test_split(data, test_size=0.5, random_state=42)
    if save_data:
        p = Path(SAVE_DIR)
        p.mkdir(parents=True, exist_ok=True)
        # Save the numpy files to the folder where they come from
        np.savetxt(SAVE_DIR + '/helix_uneven' + '.trn', data_trn)
        np.savetxt(SAVE_DIR + '/helix_uneven' + '.val', data_val)
        np.savetxt(SAVE_DIR + '/helix_uneven' + '.tst', data_tst)
    if visualize:
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')
        (xmax, xmin), (ymax, ymin), (zmax, zmin) = (20., -1.), (1.5, -1.5), (1.5, -1.5)
        plot3d(data_trn, ax1, 0.25, ymin, ymax, xmin, xmax, zmin, zmax)
        ax1.set_title("Train", fontsize=12, fontweight='bold')

        plot3d(data_val, ax2, 0.25, ymin, ymax, xmin, xmax, zmin, zmax)
        ax2.set_title("Val", fontsize=12, fontweight='bold')

        plot3d(data_tst, ax3, 0.25, ymin, ymax, xmin, xmax, zmin, zmax)
        ax3.set_title("Test", fontsize=12, fontweight='bold')
        plt.show()
    return data_trn, data_tst

def make_helix_short_appended(num_data=10000, num_helices_train=1, num_helices_test=2, c=1, r=1, sd=0.1, test_size=0.5, save_data=False, visualize=False):
    x = np.linspace(0, 2 * num_helices_train * c * np.pi, num_data)
    y = r * np.sin(x)
    z = r * np.cos(x)
    helix = np.stack([x, y, z]).T
    noise = sd * np.random.normal(size=(num_data, 3))
    data_trn = helix + noise

    # Add a small amount of data similar to the test/val datasets to the train dataset
    x = np.linspace(2 * num_helices_train * np.pi, 2 * num_helices_test * c * np.pi, 100)
    y = r * np.sin(x)
    z = r * np.cos(x)
    data_append = np.stack([x, y, z]).T
    data_trn = np.vstack([data_trn, data_append])

    x = np.linspace(0, 2 * num_helices_test * c * np.pi, num_data)
    y = r * np.sin(x)
    z = r * np.cos(x)
    helix = np.stack([x, y, z]).T
    noise = sd * np.random.normal(size=(num_data, 3))
    data_tst = helix + noise
    data_tst, data_val = train_test_split(data_tst, test_size=0.5, random_state=42)
    if save_data:
        p = Path(SAVE_DIR)
        p.mkdir(parents=True, exist_ok=True)
        # Save the numpy files to the folder where they come from
        np.savetxt(SAVE_DIR + '/helix_short_appended' + '.trn', data_trn)
        np.savetxt(SAVE_DIR + '/helix_short_appended' + '.val', data_val)
        np.savetxt(SAVE_DIR + '/helix_short_appended' + '.tst', data_tst)
    if visualize:
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')
        (xmax, xmin), (ymax, ymin), (zmax, zmin) = (20., -1.), (1.5, -1.5), (1.5, -1.5)
        plot3d(data_trn, ax1, 0.25, ymin, ymax, xmin, xmax, zmin, zmax)
        ax1.set_title("Train", fontsize=12, fontweight='bold')

        plot3d(data_val, ax2, 0.25, ymin, ymax, xmin, xmax, zmin, zmax)
        ax2.set_title("Val", fontsize=12, fontweight='bold')

        plot3d(data_tst, ax3, 0.25, ymin, ymax, xmin, xmax, zmin, zmax)
        ax3.set_title("Test", fontsize=12, fontweight='bold')
        plt.show()
    return data_trn, data_tst


make_helix_short_appended(save_data=False, visualize=True)

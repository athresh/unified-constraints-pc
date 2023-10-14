import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
import os
from pathlib import Path

SAVE_DIR = '../data/toy_3d'
PLOT_DIR = '../plots/toy_3d'

def make_helix(num_data=20000, num_helices=3, c=1, r=1, sd=0.1, test_size=0.5, save_data=False):
    x = np.linspace(0, 2 * num_helices * c * np.pi, num_data)
    y = r * np.sin(x)
    z = r * np.cos(x)
    helix = np.stack([x, y, z]).T
    noise = sd * np.random.normal(size=(num_data, 3))
    data = helix + noise
    data_trn, data_tst = train_test_split(data, test_size=test_size, random_state=42)
    data_tst, data_val = train_test_split(data_tst, test_size=0.5, random_state=42)
    if save_data:
        # Save the numpy files to the folder where they come from
        p = Path(SAVE_DIR)
        p.mkdir(parents=True, exist_ok=True)
        np.savetxt(SAVE_DIR + '/helix' + '.trn', data_trn)
        np.savetxt(SAVE_DIR + '/helix' + '.val', data_val)
        np.savetxt(SAVE_DIR + '/helix' + '.tst', data_tst)
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

def make_helix_short(num_data=10000, num_helices_train=1, num_helices_test=2, c=1, r=1, sd=0.1, test_size=0.5, save_data=False):
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
    return data_trn, data_tst

def make_helix_short_appended(num_data=10000, num_helices_train=1, num_helices_test=2, c=1, r=1, sd=0.1, test_size=0.5, save_data=False):
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
    return data_trn, data_tst


data_trn, data_tst = make_helix_short_appended(save_data=True)
data_trn, data_tst = make_helix_short(save_data=True)
data_trn, data_tst = make_helix(save_data=True)
data_trn, data_tst = make_circles(save_data=True)


import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Load the binarized MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

def convert_images_to_sets(data_loader, num_elements=50):
    image_sets = []

    for image, _ in data_loader:
        # Find the 2D coordinates of non-zero pixels
        non_zero_pixels = torch.nonzero(image.squeeze(), as_tuple=False)

        # If there are less than num_elements non-zero pixels, we pad the set with [-1, -1]
        if non_zero_pixels.shape[0] < num_elements:
            padding = torch.full((num_elements - non_zero_pixels.shape[0], 2), -1)
            non_zero_pixels = torch.cat((non_zero_pixels, padding), dim=0)

        # Randomly sample num_elements pixels from the non-zero pixels
        indices = torch.randperm(non_zero_pixels.shape[0])[:num_elements]
        sampled_pixels = non_zero_pixels[indices]

        # Add the set of sampled pixels to the list of image sets
        image_sets.append(sampled_pixels)

    return image_sets

# Convert images to sets
train_sets = convert_images_to_sets(train_loader)
test_sets = convert_images_to_sets(test_loader)

# Save the sets to a file inside ../data/MNIST
np.save('../data/MNIST/train_sets.npy', train_sets)
np.save('../data/MNIST/test_sets.npy', test_sets)

import matplotlib.pyplot as plt

def plot_image_from_coordinates(coordinates):
    # Create an empty image
    image = np.zeros((28, 28))

    # Set the pixels at the coordinates to 1
    for coord in coordinates:
        if (coord >= 0).all():  # Ignore padding
            image[coord[0], coord[1]] = 1

    # Plot the image
    plt.imsave("../data/MNIST/binarized_set_to_image.png",image, cmap='gray')

# Plot the first image in the training set
plot_image_from_coordinates(train_sets[0])
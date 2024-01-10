import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os 
import torchvision

num_elements = 100

digits = [0,1,2,3,4,5,6,7,8,9]
digits = [1,3,5,7,9]
digits = [2,4,6,8]

# Load the binarized MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)

def convert_images_to_sets(data_loader, num_elements=50):
    image_sets = []

    for image, target in data_loader:
        if target.item() not in digits:
            continue
        # Find the coordinates of non-zero pixels
        image = image.squeeze().view(-1).round()
        non_zero_pixels = torch.nonzero(image.squeeze(), as_tuple=False)

        # If there are less than num_elements non-zero pixels, skip the image
        if non_zero_pixels.shape[0] < num_elements:
            continue
            # padding = torch.full((num_elements - non_zero_pixels.shape[0], 2), -1)
            # non_zero_pixels = torch.cat((non_zero_pixels, padding), dim=0)

        # Randomly sample num_elements pixels from the non-zero pixels
        indices = torch.randperm(non_zero_pixels.shape[0])[:num_elements]
        sampled_pixels = non_zero_pixels[indices]
        # Add the set of sampled pixels to the list of image sets
        image_sets.append(sampled_pixels)

    return image_sets

# Convert images to sets
train_sets = convert_images_to_sets(train_loader, num_elements=num_elements)
test_sets = convert_images_to_sets(test_loader, num_elements=num_elements)

# Save the sets to a file inside ../data/MNIST
os.makedirs(f"../data/MNIST/num_elements={num_elements}", exist_ok=True)
np.save(f'../data/MNIST/num_elements={num_elements}/train_sets.npy', train_sets)
np.save(f'../data/MNIST/num_elements={num_elements}/test_sets.npy', test_sets)

import matplotlib.pyplot as plt

def plot_image_from_coordinates(coordinate_set):
    
    images = []
    for coordinates in coordinate_set:
        # Create an empty image
        image = np.zeros(784)

        # Set the pixels at the coordinates to 1
        for coord in coordinates:
            if (coord >= 0).all():  # Ignore padding
                image[coord] = 1

        image = torch.from_numpy(np.reshape(image,(1, 28, 28)))
        images.append(image)
    
    grid = torchvision.utils.make_grid(images, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    torchvision.utils.save_image(grid, f"../data/MNIST/num_elements={num_elements}/binarized_set_to_image.png")
    # plt.figure()
    # plt.imshow(grid.permute(1, 2, 0).detach().cpu().numpy())
    # # Plot the image
    # plt.savefig(f"../data/MNIST/num_elements={num_elements}/binarized_set_to_image.png", bbox_inches="tight")
    # plt.imsave(f"../data/MNIST/num_elements={num_elements}/binarized_set_to_image.png",image, cmap='gray')

# Plot the first image in the training set
plot_image_from_coordinates(train_sets[:64])
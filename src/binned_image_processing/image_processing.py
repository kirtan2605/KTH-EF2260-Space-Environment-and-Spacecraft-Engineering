import os
import glob
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter



def compute_image_differences(images):
    """
    Calculate differences between consecutive images in an array.

    Parameters:
    -----------
    images : numpy.ndarray
        Input array of shape (n, a, b), where:
        - n is the number of images
        - a is the height of each image
        - b is the width of each image

    Returns:
    --------
    numpy.ndarray
        Array of image differences of shape (n-1, a, b)
        Where each element is images[i+1] - images[i]
    """
    # Validate input
    if not isinstance(images, np.ndarray):
        raise TypeError("Input must be a NumPy array")

    if images.ndim != 3:
        raise ValueError("Input must be a 3D array (n, a, b)")

    # Convert to signed integer type to preserve negative values
    # Use int16 or int32 depending on your image data range
    images_signed = images.astype(np.int16)

    # Calculate differences with proper signed integer subtraction
    image_differences = images_signed[1:] - images_signed[:-1]

    return image_differences



def apply_median_filter(images, filter_size = 3):
    """
    Apply median filtering to each image in a dataset.

    Args:
        images (numpy.ndarray): A 3D array of shape (n, a, b), where n is the number of images,
                                and each image has dimensions (a, b).
        filter_size (int): The size of the median filter. Default is 3 (3x3 filter).

    Returns:
        numpy.ndarray: A 3D array of the same shape as the input, containing the filtered images.
    """
    # Ensure the input is a 3D array
    if len(images.shape) != 3:
        raise ValueError("Input images must be a 3D array of shape (n, a, b)")

    # Apply median filtering to each image
    filtered_images = np.empty_like(images)
    for i in range(images.shape[0]):
        filtered_images[i] = median_filter(images[i], size=filter_size)

    return filtered_images

# Define the folder path containing the .bin files
folder_path = '/home/kirtan/github/KTH-EF2260-Space-Environment-and-Spacecraft-Engineering/image_data/python_parsed_data_files/'  # Replace with the correct path
datetime_array_filepath = folder_path+'date_time.npy'
image_data_filepath = folder_path+'image_data.npy'

dates_array = np.load(datetime_array_filepath, allow_pickle=True)
images_array = np.load(image_data_filepath, allow_pickle=True)

# very important to get a sensible value of differences
images_array = images_array.astype(np.int16)

image_differences = compute_image_differences(images_array)

median_filtered_image_differences = apply_median_filter(image_differences, filter_size=3)


# Optionally, display the first image (now in chronological order)
plt.figure(figsize=(10, 6))
plt.imshow(median_filtered_image_differences[1500], cmap='viridis', aspect='auto')
plt.colorbar()
plt.title(f"First Difference Image (Chronologically): {dates_array[100]}")
plt.show()

# Optionally, display the first image (now in chronological order)
plt.figure(figsize=(10, 6))
plt.imshow(median_filtered_image_differences[1501], cmap='viridis', aspect='auto')
plt.colorbar()
plt.title(f"First Difference Image (Chronologically): {dates_array[100]}")
plt.show()

# Optionally, display the first image (now in chronological order)
plt.figure(figsize=(10, 6))
plt.imshow(median_filtered_image_differences[1502], cmap='viridis', aspect='auto')
plt.colorbar()
plt.title(f"First Difference Image (Chronologically): {dates_array[100]}")
plt.show()


# Optionally, display the first image (now in chronological order)
plt.figure(figsize=(10, 6))
plt.imshow(median_filtered_image_differences[1503], cmap='viridis', aspect='auto')
plt.colorbar()
plt.title(f"First Difference Image (Chronologically): {dates_array[100]}")
plt.show()


# Optionally, display the first image (now in chronological order)
plt.figure(figsize=(10, 6))
plt.imshow(median_filtered_image_differences[1504], cmap='viridis', aspect='auto')
plt.colorbar()
plt.title(f"First Difference Image (Chronologically): {dates_array[100]}")
plt.show()

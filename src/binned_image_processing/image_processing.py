import numpy as np
from PIL import Image
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import matplotlib.colors as colors
import gc

def seu_image_processing(images, window_size=3, filter_size=3, new_min=-1, new_max=1):
    """
    Comprehensive image processing function with multiple operations.
    Args:
        images (numpy.ndarray): Input 3D array of images
        window_size (int): Size of sliding window for average calculation
        filter_size (int): Size of median filter
        new_min (float): Minimum value for scaling
        new_max (float): Maximum value for scaling
    Returns:
        numpy.ndarray: Processed and scaled image array
    """
    # Validate input
    if images.ndim != 3:
        raise ValueError("Input must be a 3D numpy array with shape (n, a, b).")
    if images.shape[0] < window_size:
        raise ValueError(f"Input must have at least {window_size} images.")

    # Compute sliding window average
    windowed_avg = np.array([
        np.mean(images[i:i+window_size], axis=0)
        for i in range(images.shape[0] - window_size + 1)
    ])

    print("window average computed")

    # Slice original images
    images = images[1:-1]

    # Compute element-wise difference
    images = images - windowed_avg

    print("image difference calculated")

    # freeing space to avoid the process from getting killed
    del windowed_avg
    # Run garbage collection manually
    gc.collect()

    # Apply median filtering
    median_filtered = np.array([
        median_filter(img, size=filter_size)
        for img in images
    ])

    print("difference median filtered")

    # Extract noise by taking difference between original and filtered
    images = images - median_filtered

    # freeing space to avoid the process from getting killed
    del median_filtered
    # Run garbage collection manually
    gc.collect()

    print("noise extracted")

    # ONLY SCALING
    # Scale each image independently
    scaled_noise = np.zeros_like(images, dtype=float)
    for i in range(images.shape[0]):
        img = images[i]

        # Skip scaling if image is constant
        if np.min(img) == np.max(img):
            scaled_noise[i] = np.zeros_like(img)
        else:
            # Scale to specified range for each individual image
            scaled_noise[i] = ((img - np.min(img)) /
                                (np.max(img) - np.min(img))) * (new_max - new_min) + new_min

    # freeing space to avoid the process from getting killed
    del images
    # Run garbage collection manually
    gc.collect()

    print("scaled noise calculated")

    # THRESHOLDING
    # Set the threshold for SEU detection
    threshold = 0.5

    # Apply the threshold to create a binary matrix
    binary_images = (abs(scaled_noise) >= threshold).astype(np.int8)

    return binary_images

def save_binary_images_with_names(matrix, datetimes, output_dir, format="png"):
    """
    Saves each (1, a, b) slice of a binary (n, a, b) matrix as a black-and-white image,
    using corresponding names from a string array as filenames.

    Parameters:
        matrix (numpy.ndarray): Input 3D binary matrix of shape (n, a, b), values 0 or 1.
        filenames (list of str): Array of n filenames (without extensions).
        output_dir (str): Directory to save the images.
        format (str): Image format, e.g., "png".
    """
    # Ensure the matrix is a NumPy array
    matrix = np.asarray(matrix)
    
    # Check if the input is 3D
    if matrix.ndim != 3:
        raise ValueError("Input matrix must be a 3D array of shape (n, a, b).")
    
    # Check if the matrix is binary
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError("Input matrix must only contain binary values (0 and 1).")
    
    # Check if filenames match the number of slices
    if len(datetimes) != matrix.shape[0]:
        raise ValueError("Length of filenames array must match the number of slices in the matrix.")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over each (a, b) slice in the matrix and corresponding filename
    for i in range(matrix.shape[0]):
        # Convert the binary slice to a Pillow image in mode '1' (1-bit pixels)
        image = Image.fromarray(matrix[i].astype(np.uint8) * 255)  # Scale 0/1 to 0/255
        image = image.convert('1')  # Convert to 1-bit pixels (black-and-white)

        # Convert datetime to YYYYMMDDHHMMSS format
        filename = datetimes[i].strftime("%Y%m%d%H%M%S") + f".{format}"
        
        # Save the image with the corresponding filename
        image.save(os.path.join(output_dir, filename))
    
    print(f"Saved {matrix.shape[0]} binary images to {output_dir}")

def calculate_sums(matrix):
    """
    Calculate the sum of elements for each (1, a, b) slice in an (n, a, b) binary matrix.

    Parameters:
        matrix (numpy.ndarray): Input 3D binary matrix of shape (n, a, b), values 0 or 1.

    Returns:
        numpy.ndarray: A 2D array of shape (n, 1), containing the sum of elements for each slice.
    """
    # Ensure the matrix is a NumPy array
    matrix = np.asarray(matrix)
    
    # Check if the input is 3D
    if matrix.ndim != 3:
        raise ValueError("Input matrix must be a 3D array of shape (n, a, b).")
    
    # Check if the matrix is binary
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError("Input matrix must only contain binary values (0 and 1).")
    
    # Calculate the sum of each (a, b) slice
    slice_sums = np.sum(matrix, axis=(1, 2), keepdims=True)
    
    return slice_sums

# Define the folder path containing the .bin files
folder_path = '/home/kirtan/github/KTH-EF2260-Space-Environment-and-Spacecraft-Engineering/image_data/python_parsed_data_files/'  # Replace with the correct path
datetime_array_filepath = folder_path+'date_time.npy'
image_data_filepath = folder_path+'image_data.npy'

dates_array = np.load(datetime_array_filepath, allow_pickle=True)
images_array = np.load(image_data_filepath, allow_pickle=True)

# very important to get a sensible value of differences
images_array = images_array.astype(np.int16)

print("image array imported")

seu_identifiable_images = (seu_image_processing(images_array)).astype(np.int8)
seu_identifiable_dates = dates_array[1:-1]

# freeing space to avoid the process from getting killed
del dates_array
del images_array
# Run garbage collection manually
gc.collect()

'''
# TO SAVE IMAGEs
output_directory = "/home/kirtan/github/KTH-EF2260-Space-Environment-and-Spacecraft-Engineering/image_data/seu_identifiable_images/"

# Save the images
save_binary_images_with_names(seu_identifiable_images, seu_identifiable_dates, output_directory, format="png")
'''

# Calculate the sums for each (1, a, b) slice
seu_sums = calculate_sums(seu_identifiable_images)

# Flatten sums to match x_values
seu_sums = seu_sums.flatten()



# BINNING THE DATA OVER TIME
bin_size_seconds = 900

# Initialize bins
start_time = seu_identifiable_dates[0]
end_time = seu_identifiable_dates[-1]
bin_size = timedelta(seconds=bin_size_seconds)
current_bin_start = start_time
binned_dates = []
binned_sums = []

while current_bin_start <= end_time:
    # Determine the end of the current bin
    current_bin_end = current_bin_start + bin_size
    
    # Find indices of sums within the current bin
    in_bin = (seu_identifiable_dates >= current_bin_start) & (seu_identifiable_dates < current_bin_end)
        
    # Sum SEUs in the current bin
    bin_sum = np.sum(seu_sums[in_bin])
    binned_dates.append(current_bin_start)
    binned_sums.append(bin_sum)
        
    # Move to the next bin
    current_bin_start = current_bin_end

    
# Plot
plt.figure(figsize=(8, 6))
plt.plot(seu_identifiable_dates, seu_sums, linestyle='-', color='b', label="Sum of SEUs")
plt.xlabel("DateTime")
plt.ylabel("Number of SEUs")
plt.title("SEU variation over time")
plt.grid(True)
plt.legend()
plt.show()


# Plot the binned data as a bar chart
plt.figure(figsize=(10, 6))
plt.bar(binned_dates, binned_sums, width=0.01, color='b', label="Binned SEUs")  # Bar chart with width adjusted for readability
plt.xlabel("DateTime")
plt.ylabel("Number of SEUs")
plt.title("SEU Variation (Binned Every 15 Minutes)")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


'''
# Plot
plt.figure(figsize=(8, 6))
plt.plot(binned_dates, binned_sums, linestyle='-', color='b', label="Binned Sum of SEUs")
plt.xlabel("DateTime")
plt.ylabel("Number of SEUs")
plt.title("SEU variation over time")
plt.grid(True)
plt.legend()
plt.show()
'''


'''
using averaging over 3 images, we can not use the first and the second image.
Thus, for initial number of images = N, the SEU identifiable images are N-2
thus, the date-time information can be matched with the SEU identifiable images
by removing the first and last elements
'''



'''
# FOR ONLY SCALED IMAGES

plt.figure(figsize=(10, 6))
im = plt.imshow(seu_identifiable[10], cmap='viridis', aspect='auto', norm=colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
cbar = plt.colorbar(im)
cbar.set_ticks([-1, 0, 1])
cbar.set_ticklabels(['-1', '0', '1'])
plt.title(f"SEU Processed Image: {10}")
plt.show()
'''

'''
plt.figure(figsize=(10, 6))
im = plt.imshow(seu_identifiable[0], cmap='viridis', aspect='auto')
plt.colorbar(im)
plt.title(f"SEU Processed Image: {0}")
plt.show()

plt.figure(figsize=(10, 6))
im = plt.imshow(seu_identifiable[1], cmap='viridis', aspect='auto')
plt.colorbar(im)
plt.title(f"SEU Processed Image: {1}")
plt.show()

plt.figure(figsize=(10, 6))
im = plt.imshow(seu_identifiable[2], cmap='viridis', aspect='auto')
plt.colorbar(im)
plt.title(f"SEU Processed Image: {2}")
plt.show()


plt.figure(figsize=(10, 6))
im = plt.imshow(seu_identifiable[10], cmap='viridis', aspect='auto')
plt.colorbar(im)
plt.title(f"SEU Processed Image: {10}")
plt.show()

plt.figure(figsize=(10, 6))
im = plt.imshow(seu_identifiable[100], cmap='viridis', aspect='auto')
plt.colorbar(im)
plt.title(f"SEU Processed Image: {100}")
plt.show()

plt.figure(figsize=(10, 6))
im = plt.imshow(seu_identifiable[500], cmap='viridis', aspect='auto')
plt.colorbar(im)
plt.title(f"SEU Processed Image: {500}")
plt.show()

plt.figure(figsize=(10, 6))
im = plt.imshow(seu_identifiable[1000], cmap='viridis', aspect='auto')
plt.title(f"SEU Processed Image: {1000}")
plt.show()

plt.figure(figsize=(10, 6))
im = plt.imshow(seu_identifiable[2500], cmap='viridis', aspect='auto')
plt.colorbar(im)
plt.title(f"SEU Processed Image: {2500}")
plt.show()

plt.figure(figsize=(10, 6))
im = plt.imshow(seu_identifiable[5000], cmap='viridis', aspect='auto')
plt.colorbar(im)
plt.title(f"SEU Processed Image: {5000}")
plt.show()

plt.figure(figsize=(10, 6))
im = plt.imshow(seu_identifiable[7500], cmap='viridis', aspect='auto')
plt.colorbar(im)
plt.title(f"SEU Processed Image: {7500}")
plt.show()

plt.figure(figsize=(10, 6))
im = plt.imshow(seu_identifiable[10000], cmap='viridis', aspect='auto')
plt.colorbar(im)
plt.title(f"SEU Processed Image: {10000}")
plt.show()

plt.figure(figsize=(10, 6))
im = plt.imshow(seu_identifiable[12500], cmap='viridis', aspect='auto')
plt.colorbar(im)
plt.title(f"SEU Processed Image: {12500}")
plt.show()
'''

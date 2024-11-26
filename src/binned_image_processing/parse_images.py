import os
import glob
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Function to extract datetime from filename (assuming format: 'IR1_YYYYMMDDHHMMSS')
def extract_datetime_from_filename(filename):
    basename = os.path.basename(filename)  # Extracts just the filename (no path)
    datetime_str = basename[4:].replace('.bin', '')  # e.g., '20230315000614'
    return datetime.strptime(datetime_str, '%Y%m%d%H%M%S')

# Define the folder path containing the .bin files
folder_path = '/home/kirtan/github/KTH-EF2260-Space-Environment-and-Spacecraft-Engineering/image_data/20230315_binned_images/'  # Replace with the correct path

# List all .bin files in the folder
bin_files = glob.glob(os.path.join(folder_path, '*.bin'))

# Check if there are any .bin files
if not bin_files:
    print("No .bin files found in the folder.")
    exit()

# Sort files based on the datetime extracted from the filename
bin_files_sorted = sorted(bin_files, key=extract_datetime_from_filename)

# Function to read and store images and dates into separate arrays
def load_images_and_dates(sorted_bin_files):
    dates = []  # List to store the dates extracted from filenames
    images = []  # List to store image data (as numpy arrays)

    # Loop through each .bin file in chronological order
    for bin_file in sorted_bin_files:
        # Extract the date from the filename and append it to the dates list
        capture_datetime = extract_datetime_from_filename(bin_file)
        dates.append(capture_datetime)

        # Read the binary data and reshape it into the expected image shape
        with open(bin_file, 'rb') as file:
            image_data = np.fromfile(file, dtype=np.uint16, count=187 * 44).reshape(187, 44)
            images.append(image_data)

    # Convert lists to numpy arrays
    dates_array = np.array(dates)
    images_array = np.array(images)

    return dates_array, images_array



# Define the folder path containing the .bin files
folder_path = '/home/kirtan/github/KTH-EF2260-Space-Environment-and-Spacecraft-Engineering/image_data/20230315_binned_images/'  # Replace with the correct path

# List all .bin files in the folder
bin_files = glob.glob(os.path.join(folder_path, '*.bin'))

# Check if there are any .bin files
if not bin_files:
    print("No .bin files found in the folder.")
    exit()

# Sort files based on the datetime extracted from the filename
bin_files_sorted = sorted(bin_files, key=extract_datetime_from_filename)

# Load images and dates into separate arrays (in chronological order)
dates_array, images_array = load_images_and_dates(bin_files_sorted)

(data_length, image_width, image_height) = images_array.shape
# print(data_length)
# print(image_width)
# print(image_height)

'''
# Optionally, display the first image (now in chronological order)
plt.figure(figsize=(10, 6))
plt.imshow(images_array[0], cmap='viridis', aspect='auto')
plt.colorbar()
plt.title(f"First Image (Chronologically): {os.path.basename(bin_files_sorted[0])}")
plt.show()
'''

# Save the date_time array in .npy format (in chronological order)
np.save('date_time.npy', dates_array)
# Save the image_data array in .npy format (in chronological order)
np.save('image_data.npy', images_array)

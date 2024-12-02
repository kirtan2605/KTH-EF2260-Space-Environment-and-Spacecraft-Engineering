import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects
from datetime import datetime

def load_full_frame_image(filepath):
    """
    Load a full-frame image from a binary file.
    Args:
        filepath (str): Path to the .bin file.
    Returns:
        numpy.ndarray: The loaded image as a 2D array.
    """
    with open(filepath, 'rb') as file:
        # Read binary data and reshape to 2048x511
        image = np.fromfile(file, dtype=np.uint16, count=2048 * 511).reshape(511, 2048)
    return image

def detect_particle_impacts(image, threshold=500):
    """
    Detect particle impacts in a full-frame image.
    Args:
        image (numpy.ndarray): The input image.
        threshold (int): Pixel value threshold for detection.
    Returns:
        numpy.ndarray: Binary image with detected impacts marked.
    """
    # Create a binary mask of pixels above the threshold
    binary_image = (image > threshold).astype(np.int8)
    return binary_image

def analyze_tracks(binary_image, original_image):
    """
    Analyze particle tracks in a binary image.
    Args:
        binary_image (numpy.ndarray): Binary image with particle tracks.
        original_image (numpy.ndarray): Original full-frame image.
    Returns:
        list: A list of dictionaries containing track information.
    """
    # Label connected components in the binary image
    labeled_image, num_features = label(binary_image)
    track_data = []

    # Loop through each detected feature
    for i in range(1, num_features + 1):
        # Extract pixels belonging to the current track
        track_pixels = (labeled_image == i)
        track_indices = np.argwhere(track_pixels)

        # Compute properties of the track
        intensity = np.sum(original_image[track_pixels])  # Total intensity
        length = len(track_indices)  # Number of pixels
        bounding_box = find_objects(labeled_image == i)[0]

        # Approximate track angle
        y_coords, x_coords = zip(*track_indices)
        angle = np.rad2deg(np.arctan2(max(y_coords) - min(y_coords),
                                      max(x_coords) - min(x_coords)))

        track_data.append({
            'intensity': intensity,
            'length': length,
            'angle': angle,
            'bounding_box': bounding_box
        })

    return track_data

def extract_timestamp(filename):
    """
    Extract timestamp from filename in the format IR1_YYYYMMDDHHMMSS.bin.
    Args:
        filename (str): Filename of the image.
    Returns:
        datetime: Extracted timestamp as a datetime object.
    """
    timestamp = filename.split('_')[1][:14]  # Extract "YYYYMMDDHHMMSS"
    return datetime.strptime(timestamp, "%Y%m%d%H%M%S")

def plot_seus_over_time(timestamps, seu_counts):
    """
    Plot SEUs over time.
    Args:
        timestamps (list): List of datetime objects.
        seu_counts (list): List of SEU counts.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(timestamps, seu_counts, width=0.03, align='center', color='skyblue', edgecolor='black')
    plt.xlabel("Time")
    plt.ylabel("Number of SEUs")
    plt.title("SEUs Over Time")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.show()

# Main script
folder_path = 'c:/Users/saras/Desktop/Space Environment/Lab D/full_frame'
threshold = 500  # SEU detection threshold
seu_counts = []
timestamps = []

for filename in os.listdir(folder_path):
    if filename.endswith('.bin'):
        filepath = os.path.join(folder_path, filename)
        
        # Load and process the full-frame image
        full_frame_image = load_full_frame_image(filepath)
        binary_image = detect_particle_impacts(full_frame_image, threshold)
        
        # Count SEUs (white pixels in binary image)
        seus = np.sum(binary_image)
        seu_counts.append(seus)
        
        # Extract timestamp
        timestamp = extract_timestamp(filename)
        timestamps.append(timestamp)

# Sort data by time
timestamps, seu_counts = zip(*sorted(zip(timestamps, seu_counts)))

# Plot SEUs over time
plot_seus_over_time(timestamps, seu_counts)

def plot_detected_tracks(original_image, binary_image, tracks):
    """
    Plot the original image with detected tracks highlighted.
    Args:
        original_image (numpy.ndarray): Original full-frame image.
        binary_image (numpy.ndarray): Binary image with detected impacts.
        tracks (list): List of detected track information.
    """
    plt.figure(figsize=(12, 6))
    plt.imshow(original_image, cmap='viridis', aspect='auto', alpha=0.8)
    plt.imshow(binary_image, cmap='Reds', aspect='auto', alpha=0.4)
    plt.colorbar()
    plt.title("Detected Particle Tracks")
    plt.show()

    # Optionally, annotate tracks with information
    for track in tracks:
        print(f"Track: Intensity={track['intensity']}, Length={track['length']}, Angle={track['angle']:.2f}°")


# Example usage
# Path to your full-frame .bin file
full_frame_file = 'c:/Users/saras/Desktop/Space Environment/Lab D/full_frame/IR1_20230819214402.bin'

# Load and process the full-frame image
full_frame_image = load_full_frame_image(full_frame_file)
binary_image = detect_particle_impacts(full_frame_image, threshold=500)
tracks = analyze_tracks(binary_image, full_frame_image)

# Display the results
plot_detected_tracks(full_frame_image, binary_image, tracks)


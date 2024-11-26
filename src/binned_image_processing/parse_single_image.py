import numpy as np
import matplotlib.pyplot as plt

# Define the folder path containing the .bin files
folder_path = '/home/kirtan/github/KTH-EF2260-Space-Environment-and-Spacecraft-Engineering/image_data/20230315_binned_images/'
image_name = 'IR1_20230315000614.bin'

filepath = folder_path + image_name

# Read binary data from the file
ff = open(filepath, 'rb')
image = np.fromfile(ff, dtype=np.uint16, count=187*44).reshape(187, 44)
ff.close()

plt.figure(figsize=(10, 6))
plt.imshow(image, cmap='viridis', aspect='auto')
plt.colorbar()
plt.show()

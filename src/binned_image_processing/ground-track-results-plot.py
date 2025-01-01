import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from skyfield.api import load, EarthSatellite
import csv
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def load_data_from_csv(dates_file_path, sums_file_path):
    dates = []
    with open(dates_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            dates.append(row[0])
    
    sums = []
    with open(sums_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            sums.append(float(row[0]))

    return np.array(dates), np.array(sums)

def parse_tle_file(tle_file_path):
    tles = []
    with open(tle_file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]
        
        for i in range(0, len(lines), 2):
            if i + 1 >= len(lines):
                break
                
            line1 = lines[i]
            line2 = lines[i + 1]
            
            if not (line1.startswith('1') and line2.startswith('2')):
                continue
            
            try:
                epoch_year = int(line1[18:20])
                epoch_day = float(line1[20:32])
                
                full_year = 2000 + epoch_year if epoch_year < 50 else 1900 + epoch_year
                date = datetime(full_year, 1, 1) + timedelta(days=epoch_day - 1)
                
                tles.append((line1, line2, date))
                
            except (ValueError, IndexError) as e:
                print(f"Error parsing TLE pair at index {i}: {e}")
                print(f"Line 1: {line1}")
                print(f"Line 2: {line2}")
    
    if not tles:
        raise ValueError("No valid TLEs found in file")
        
    tles.sort(key=lambda x: x[2])
    return tles

def plot_ground_track_with_seus(tles, seu_dates, seu_sums):
    ts = load.timescale()
    
    # Create figure with Cartopy projection
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add natural Earth features
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.OCEAN, color='lightblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.gridlines(draw_labels=True, dms=True, alpha=0.3)
    
    # Create logarithmic normalization for color scale
    # Add small constant to avoid log(0)
    min_nonzero = np.min(seu_sums[seu_sums > 0])  # Find minimum non-zero value
    norm = LogNorm(vmin=min_nonzero, vmax=seu_sums.max())
    cmap = plt.cm.viridis

    lons = []
    lats = []
    values = []

    for date_str, seu_sum in zip(seu_dates, seu_sums):
        try:
            current_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            
            t = ts.utc(current_date.year, current_date.month, current_date.day,
                       current_date.hour, current_date.minute, current_date.second)
            
            closest_tle = select_tle_for_time(tles, current_date)
            satellite = EarthSatellite(closest_tle[0], closest_tle[1])
            
            # Get geocentric position
            geometric = satellite.at(t)
            subpoint = geometric.subpoint()
            
            # Extract latitude and longitude in degrees
            lat = float(subpoint.latitude.degrees)
            lon = float(subpoint.longitude.degrees)
            
            lons.append(lon)
            lats.append(lat)
            values.append(seu_sum)
            
        except Exception as e:
            print(f"Error plotting point for date {date_str}: {e}")
            continue

    # Plot all points at once for better performance
    scatter = ax.scatter(lons, lats, c=values, cmap=cmap, norm=norm, 
                        s=100, transform=ccrs.PlateCarree())

    # Add colorbar with logarithmic scale
    cbar = plt.colorbar(scatter, ax=ax, label='SEU Count',
                       format='%.0e')
    cbar.ax.tick_params(labelsize=10)
    
    # Set title and layout
    plt.title('Spacecraft Ground Track with SEU Count', 
             pad=20)
    
    plt.tight_layout()
    plt.show()

def select_tle_for_time(tles, current_date):
    valid_tles = [tle for tle in tles if tle[2] <= current_date]
    if not valid_tles:
        valid_tles = [tles[0]]
    return max(valid_tles, key=lambda x: x[2])

# Define file paths
folder_path = '/home/kirtan/local-repository/KTH-EF2260-Space-Environment-and-Spacecraft-Engineering/image-data/python-parsed-data-files/'
dates_file_path = folder_path + 'seu_identifiable_dates.csv'
sums_file_path = folder_path + 'seu_sums.csv'
tle_file_path = '/home/kirtan/local-repository/KTH-EF2260-Space-Environment-and-Spacecraft-Engineering/satellite-data/sat000054227.txt'

# Load the data
seu_dates, seu_sums = load_data_from_csv(dates_file_path, sums_file_path)
tles = parse_tle_file(tle_file_path)

# Plot ground track
plot_ground_track_with_seus(tles, seu_dates, seu_sums)
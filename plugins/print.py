import os
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pandas as pd
import numpy as np
import contextily as ctx

# Set environment variable to restore or create SHX file
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# Load the CSV data
csv_paths = [
    '/workspace/devkit23/plugins/plugin2/NDVI_California_Dataset/NDVI_California_Dataset/Third Part/Transformed_NDVI_Third_Part_2020_April.csv',
    '/workspace/devkit23/plugins/plugin2/NDVI_California_Dataset/NDVI_California_Dataset/Fourth Part/Transformed_NDVI_Fourth_Part_2020_April.csv',
    '/workspace/devkit23/plugins/plugin2/NDVI_California_Dataset/NDVI_California_Dataset/Fifth_Part/Transformed_NDVI_Fifth_Part_2020_April.csv',
    # Add more CSV paths as needed
]
for csv_path in csv_paths:
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV file loaded successfully: {csv_path}")
    except FileNotFoundError:
        df = pd.DataFrame({
            'lat': np.random.uniform(32, 42, 1000),  # Generating sample data for latitude
            'lon': np.random.uniform(-124, -114, 1000)  # Generating sample data for longitude
        })
        print("CSV file not found. Using sample data.")


# Extract the latitude and longitude values
latitudes = df['lat']
longitudes = df['lon']

# Print the values to check for mistakes
print("Latitudes:", latitudes.head())
print("Longitudes:", longitudes.head())

# Create a list of points
points = [Point(lon, lat) for lat, lon in zip(latitudes, longitudes)]

# Create a GeoDataFrame with these points
gdf_points = gpd.GeoDataFrame(geometry=points)

# Load the California shapefile (make sure to use the correct path)
shapefile_path = '/workspace/devkit23/plugins/plugin2/cb_2021_us_state_20m/cb_2021_us_state_20m.shp'
try:
    california = gpd.read_file(shapefile_path)
    california = california[california['NAME'] == 'California']
    print(f"Shapefile loaded successfully: {shapefile_path}")
except Exception as e:
    california = gpd.GeoDataFrame()
    print(f"Error loading shapefile: {e}")

# Ensure the coordinate reference system (CRS) matches
gdf_points.set_crs(epsg=4326, inplace=True)
if not california.empty:
    california = california.to_crs(epsg=4326)

# Plot the map
fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the figure size as needed
if not california.empty:
    california.boundary.plot(ax=ax, linewidth=1, edgecolor='white')
gdf_points.plot(ax=ax, color='red', markersize=0.1)  # Reduced marker size for better visibility

# Zoom in on a specific region (e.g., Northern California)
minx, miny, maxx, maxy = -122, 37, -121, 39  # Adjust these values to zoom in on the desired region
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

# Add satellite basemap
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, crs=gdf_points.crs.to_string())

plt.title('Map of California with Points from CSV (Zoomed In)')

# Save the plot as an image file
output_path = 'official_plotting.png'
plt.savefig(output_path, dpi=300)
print(f"Map saved as {output_path}")

# Display the plot (optional)
plt.show()

from osgeo import gdal, gdalconst
import os
import numpy as np

# Path to the large GeoTIFF file
input_file = '1464-1145_quad.tif'

# Output directory to save the tiles
output_dir = 'clip'

tile_size_x = 256
tile_size_y = 256

# Open the large GeoTIFF file
dataset = gdal.Open(input_file, gdalconst.GA_ReadOnly)

if dataset is None:
    print("Error: Could not open the file.")
    exit(1)

# Get the geotransform parameters and SRS
geotransform = dataset.GetGeoTransform()
srs = dataset.GetProjection()

# Loop through the large GeoTIFF file and create tiles
for y in range(0, dataset.RasterYSize, tile_size_y):
    for x in range(0, dataset.RasterXSize, tile_size_x):
        # Define the tile's boundaries
        x_offset = x
        y_offset = y
        x_size = min(tile_size_x, dataset.RasterXSize - x)
        y_size = min(tile_size_y, dataset.RasterYSize - y)

        # Read the data from the large GeoTIFF file
        data = dataset.ReadAsArray(x_offset, y_offset, x_size, y_size)

        # Ensure the data array is two-dimensional
        if len(data.shape) == 3 and data.shape[0] == 1:
            data = data[0]

        # Create a new GeoTIFF file for the tile
        driver = gdal.GetDriverByName('GTiff')
        output_tile_path = os.path.join(output_dir, f'tile_{x}_{y}.tif')
        output_tile = driver.Create(output_tile_path, x_size, y_size, 1, gdalconst.GDT_Float32)
        output_tile.SetGeoTransform((geotransform[0] + x_offset * geotransform[1],
                                      geotransform[1],
                                      geotransform[2],
                                      geotransform[3] + y_offset * geotransform[5],
                                      geotransform[4],
                                      geotransform[5]))
        output_tile.SetProjection(srs)
        output_tile.GetRasterBand(1).WriteArray(data)

        # Close the output tile dataset
        output_tile = None

# Close the input dataset
dataset = None
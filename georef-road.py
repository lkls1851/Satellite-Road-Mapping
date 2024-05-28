import cv2
import numpy as np
import ezdxf
from osgeo import gdal

input_road_tif='DoLR/merged256_new_nov.tif'
dxf_path=((input_road_tif.split('.'))[0]+'.dxf')
INPUT_SIZE=4096

image = cv2.imread(input_road_tif)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a binary image
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

binary = cv2.bitwise_not(binary)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

doc = ezdxf.new(dxfversion='R2010')

msp = doc.modelspace()
for contour in contours:
    points = [tuple(point[0]) for point in contour]
    msp.add_lwpolyline(points)

doc.saveas(dxf_path)


input_ds = gdal.Open(input_road_tif)
geotransform = input_ds.GetGeoTransform()
block_width = input_ds.RasterXSize
block_height = input_ds.RasterYSize

for block_x in range(0, block_width, INPUT_SIZE):
    for block_y in range(0, block_height, INPUT_SIZE):

        output_path = dxf_path
        output_ds = gdal.GetDriverByName('GTiff').Create(output_path, INPUT_SIZE, INPUT_SIZE, input_ds.RasterCount, input_ds.GetRasterBand(1).DataType)
        output_ds.SetGeoTransform((
            geotransform[0] + block_x * geotransform[1],
            geotransform[1],
            0.0,
            geotransform[3] + block_y * geotransform[5],
            0.0,
            geotransform[5]
        ))
        for band in range(1, input_ds.RasterCount + 1):
            input_band = input_ds.GetRasterBand(band)
            output_band = output_ds.GetRasterBand(band)
            output_band.ReadAsArray(block_x, block_y, INPUT_SIZE, INPUT_SIZE)
            output_band.WriteArray(input_band.ReadAsArray(block_x, block_y, INPUT_SIZE, INPUT_SIZE))
        output_ds = None
        input_band = None
        output_band = None
input_ds = None
print('DXF File saved successfully')

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image


image_path='DoLR/merged256_new_nov.tif'

image=cv2.imread(image_path)
_, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)


kernel = np.ones((10,10),np.uint8)

negative=255-image
dilated_image = cv2.dilate(negative, kernel, iterations=1)

broad=Image.fromarray(dilated_image)
save_path='thick_'+image_path
broad.save(save_path)
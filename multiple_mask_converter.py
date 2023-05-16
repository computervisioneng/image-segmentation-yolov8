"""
Author: RomanGoEmpire
Description: This Python script converts segmentation masks (format 1.1) downloaded from the app.cvat.ai website
into a format compatible with training YOLOv8 models.
"""

import os
import cv2
import numpy as np
import random

# Directory paths for input masks and output labels
input_dir = './tmp/masks'
output_dir = './tmp/labels'

# Path to the input image for automatic color extraction
image_for_automatic_path = 'input_image.png'  # Change this path if you want to use a different image

# Read the input image for automatic color extraction
image = cv2.imread(image_for_automatic_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixels = image_rgb.reshape(-1, 3)
unique_colors = np.unique(pixels, axis=0)
print(unique_colors)
# TODO: If you want to enter the colors manually, uncomment the line below and add the colors as a list of RGB values
# unique_colors = [[255, 255, 255], [200, 200, 200]]

# Process each mask image in the input directory
for mask_filename in os.listdir(input_dir):
    mask_path = os.path.join(input_dir, mask_filename)

    # Read the mask image
    mask_image = cv2.imread(mask_path)
    mask_image_rgb = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    detected_polygons = []

    # Process each unique color in the input mask image
    for color in unique_colors:
        if np.array_equal(color, np.zeros(3, dtype='int8')):
            continue

        # TODO: Uncomment the lines below to visualize the contours
        # Generate a random contour color for visualization
        # contour_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        lower = np.array(color, dtype=np.uint8)
        upper = np.array(color, dtype=np.uint8)
        mask = cv2.inRange(mask_image_rgb, lower, upper)
        H, W = mask.shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []

        # Extract polygons from contours
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                polygon = []
                for point in cnt:
                    x, y = point[0]
                    polygon.append(x / W)
                    polygon.append(y / H)
                polygons.append(polygon)

        detected_polygons.append(polygons)

        # TODO: Uncomment the lines below to visualize the contours
        # cv2.drawContours(mask_image, contours, -1, contour_color, 2)
        # cv2.imshow("Image with Contours", mask_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Write polygons to a text file
        with open('{}.txt'.format(os.path.join(output_dir, mask_filename)[:-4]), 'w') as f:
            for index, polygons in enumerate(detected_polygons):
                for polygon in polygons:
                    for p_, p in enumerate(polygon):
                        if p_ == len(polygon) - 1:
                            f.write('{}\n'.format(p))
                        elif p_ == 0:
                            f.write('{} {} '.format(index, p))
                        else:
                            f.write('{} '.format(p))

    print(f'Created label file: {mask_path}')
    f.close()

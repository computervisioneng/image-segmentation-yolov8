from ultralytics import YOLO

import cv2


model_path = '/home/phillip/Desktop/train2/weights/last.pt'

image_path = '/home/phillip/Desktop/todays_tutorial/41_image_segmentation_yolov8/code/data/images/val/1be566eccffe9561.png'

img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)

for result in results:
    for j, mask in enumerate(result.masks.data):

        mask = mask.numpy() * 255

        mask = cv2.resize(mask, (W, H))

        cv2.imwrite('./output.png', mask)


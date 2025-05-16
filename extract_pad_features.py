import cv2
import numpy as np
import os
import csv
from ultralytics import YOLO

def findCorners(image, model):
    results = model(image)
    result = results[0]
    if len(result.obb.xyxyxyxy) == 0:
        return None 
    points = result.obb.xyxyxyxy.cpu().numpy()
    x1, y1, x2, y2, x3, y3, x4, y4 = int(points[0][0][0]), int(points[0][0][1]), \
         int(points[0][1][0]), int(points[0][1][1]), \
         int(points[0][2][0]), int(points[0][2][1]), \
         int(points[0][3][0]), int(points[0][3][1])
    points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    return points

def findArea(points):
    x_dist = abs(points[0][0] - points[2][0])
    y_dist = abs(points[0][1] - points[2][1])
    area = x_dist * y_dist 
    return area

def findCenter(points):
    xSum = points[0][0] + points[1][0] + points[2][0] + points[3][0]
    ySum = points[0][1] + points[1][1] + points[2][1] + points[3][1]
    centerCoords = [int(xSum/4), int(ySum/4)]
    return centerCoords

csv_file = 'obb_data.csv'
model = YOLO('best_obb.pt')  # load YOLO model
write_header = not os.path.exists(csv_file)  # create csv if not already made

 # iterate through each .jpg in Frames foldwr
for filename in os.listdir('Frames'):
    image_path = 'Frames/' + filename
    img = cv2.imread(image_path)

    coords = findCorners(img, model)
    if coords is None:
        continue
    area = findArea(coords) 
    centerCoords = findCenter(coords)
    target = int(filename.split('_')[0])

        # Write to CSV
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['Target', 'x_center', 'y_center', 'Area', 'Name'])
            write_header = False
        writer.writerow([target] + [centerCoords[0]] + [centerCoords[1]] + [area] + [filename])

    print(f"Processed and saved: {filename}")

print("Processing complete.")


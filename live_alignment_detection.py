from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np

#functions to extract data needed from YOLO
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

def findCenter(points): #Not using avg since the last point is the same as the first
    xSum = points[0][0] + points[1][0] + points[2][0] + points[3][0]
    ySum = points[0][1] + points[1][1] + points[2][1] + points[3][1]
    centerCoords = [int(xSum/4), int(ySum/4)]
    return centerCoords

#SVM model
df = pd.read_csv('obb_data.csv')
X = df.drop(columns=['Target', 'Name'])
Y = df['Target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.20, random_state=3)
svm = SVC(kernel='linear')
svm.fit(X_train, Y_train)
header = np.array(['x_center', 'y_center', 'Area'])

#YOLO model
yolo = YOLO('best_obb.pt')

#Setting up to read video
input_path = 'HeavyDrift.mp4'
cap = cv2.VideoCapture(input_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("Drift_output_video.mp4", fourcc, fps, (width, height))

#reading/writing video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = yolo(frame)
    annotated_frame = results[0].plot()
    
    #apply svm to the frame
    coords = findCorners(frame, yolo)
    if coords:
        area = findArea(coords)
        centerCoords = findCenter(coords)
        predVal = np.array([centerCoords[0], centerCoords[1], area]).reshape(1,-1)
        predVal_df = pd.DataFrame(predVal, columns=header)
        pred = svm.predict(predVal_df)
        print(pred[0])
        text = f"{'Aligned' if pred[0] == 1 else 'Misaligned'} | Center: {centerCoords} | Area: {area}"

        cv2.putText(
            annotated_frame,
            text,
            org=(20, int(height/7)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=3,
            color=(255, 0, 0),
            thickness=10
        )

    out.write(annotated_frame)
    
cap.release()
out.release()
cv2.destroyAllWindows()











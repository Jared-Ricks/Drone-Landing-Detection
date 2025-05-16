import cv2
import os

path_name = 'Flying'
cap = cv2.VideoCapture(f'{path_name}.mp4')
fps = 30  # 30 frames per second
start = 7 * fps # start at 7 seconds in 
end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 6 * fps
interval = 6  # save every 6 frames = 5 fps from 30 fps video

os.makedirs('Test_Frames', exist_ok=True)

prefix = '1'
frame_num = 0
count = 1

while True:
    success, frame = cap.read()
    if not success or frame_num >= end:
        break
    if frame_num >= start and (frame_num - start) % interval == 0:  
        name = f"Test_Frames/_{path_name}.{count}.jpg"
        cv2.imwrite(name, frame)
        print("Saved", name)
        count += 1
    frame_num += 1

cap.release()
print("Done.")

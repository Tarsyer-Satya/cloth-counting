import os
import random

import cv2
from ultralytics import YOLO

from tracker import Tracker
from collections import defaultdict
import time



def find_centroid(x1,y1,x2,y2):
    return [(x1 + x2)//2, (y1 + y2)//2]

def find_point(l1,l2,x,y):
    x1,y1,x2,y2 = l1[0], l1[1], l2[0], l2[1]
    m = (y2 - y1)/(x2-x1)
    
    c = y1 - m*x1

    y_line = m*x + c

    x_line=(y - c)/m

    if(x_line < x and y_line > y):
        return "right"
    return "left"




start_point = (93,169)
end_point = (795,533)


video_path = os.path.join('.', 'data', 'people.mp4')
video_out_path = os.path.join('.', 'cloth_counting.mp4')

video_path = 'data/video1.mp4'

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model = YOLO("yolov8s-world.pt")
model.set_classes(["book"])

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(20)]
count = 0



track_dict = defaultdict(str)
cloth_count = 0
# cars_left = 0

start_time = time.time()

while ret:

    results = model.predict(frame, conf= 0.003,iou = 0.3)
    count += 1

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            centroid = find_centroid(x1, y1, x2, y2)


            print(track_dict[track_id])
            if(track_dict[track_id] != ""):
                if(track_dict[track_id] != find_point(start_point, end_point,centroid[0], centroid[1])):
                    cloth_count += 1
            

            track_dict[track_id] = find_point(start_point, end_point, centroid[0], centroid[1])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)

        cv2.putText(frame, f"cloth count: {cloth_count}", (50,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (255,0,0), 3)
        # cv2.putText(frame, f"vehicles_left: {cars_left}", (50,100), cv2.FONT_HERSHEY_SIMPLEX , 1, (255,0,0), 3)



    frame = cv2.line(frame, start_point, end_point, (0,155,0), 3)
    cap_out.write(frame)
    # cv2.imshow('image',frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    
    ret, frame = cap.read()

# cv2.imwrite('sample.jpg',frame)

end_time = time.time()


print("time took in seonds: ", end_time-start_time)
cap.release()
cap_out.release()
# cv2.destroyAllWindows()

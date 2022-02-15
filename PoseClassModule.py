import mediapipe as mp
import os
import cv2
import csv
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

file_path = '/home/dayana/anaconda3/envs/myenv/Projects/Mediapipe/pose'

class_name = ['floorwork', 'full_frontal', 'hourglass', 'mermaid', 'sideways']

joints=33 

landmarks =['class']

for n in range (1, joints+1):
    landmarks += ['x{}'.format(n), 'y{}'.format(n), 'z{}'.format(n), 'v{}'.format(n)]

with open('coords.csv', mode='w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(landmarks)

#estimating and exporting poses landmarks to csv

for i in range(5):

    cap = cv2.VideoCapture(os.path.join(file_path,  '{pose_name}.mp4'.format(pose_name = class_name[i])))

    with mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if ret==False: #looping pose videos
                break
            
            frame=cv2.resize(frame, (frame.shape[1] // 2,frame.shape[0] // 2))
            # Recolor to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Estimation
            results = pose.process(image)
        
            # Recolor to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            
            # Drawing
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)         

            try:
                # Extract Pose landmarks
                pose_class = results.pose_landmarks.landmark
                row= []
                for landmark in range(len(pose_class)):
                    row.append([pose_class[landmark].x, pose_class[landmark].y, pose_class[landmark].z, pose_class[landmark].visibility])
                #row=list(np.array(row).flatten())
                # Append class name 
                row.insert(0, class_name[i])
                
                # Export to CSV
                with open('coords.csv', mode='a') as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerow(row) 
                
            except:
                pass      
            
            cv2.imshow('Posing', image)

            k = cv2.waitKey(1) & 0xFF
            if k == 27   :  # wait for ESC key to exit
                break

    cap.release()
    cv2.destroyAllWindows()
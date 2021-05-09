"""
 * @author Faezeh Shayesteh
 * @email shayesteh.fa@gmail.com
 * @create date 2021-04-23 21:26:28
 * @modify date 2021-04-23 21:26:28
 * @desc [description]
"""

import imutils
import time
import dlib
import cv2
import numpy as np
from makeup.eyeshadow import eyeshadow
from makeup.lipstick import lipstick

#initiating camera
prev = 0
frame_rate = 15
detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")
print("[INFO] camera sensor warming up...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)


# applying makup on frames
while True:
    # frame rate and resize frame
    ret, frame = cap.read()
    time_elapsed = time.time() - prev
    frame = imutils.resize(frame, width = 700)

    if(time_elapsed > 1./frame_rate):
        # preparing frame
        prev = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        eye = eyeshadow(frame2)
        lip = lipstick(frame2)
        # detect faces in frame
        detected_faces = detector(gray, 0)
        landmarks_x = []
        landmarks_y = []    
        # get landmarks of the face
        # try:
        for face in detected_faces:
            pose_landmarks = face_pose_predictor(gray, face)
            for i in range(68):
                landmarks_x.append(pose_landmarks.part(i).x)
                landmarks_y.append(pose_landmarks.part(i).y)
            # frame = eye.apply_eyeshadow(landmarks_x,landmarks_y,100,20,90,0.5)
            frame = lip.apply_lipstick(landmarks_x,landmarks_y,100, 20 , 30, "soft", False)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # except Exception as e:
            # print(e)
    # show face with applied makeup
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
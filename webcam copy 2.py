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
from makeup.foundation import foundation
from imutils import face_utils
#initiating camera
prev = 0
frame_rate = 12
detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor("./data/shape_predictor_81_face_landmarks.dat")
face_pose_predictor68= dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")



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
        # eye = eyeshadow(frame2)
        # lip = lipstick(frame2)
        skin = foundation()
        # detect faces in frame
        detected_faces = detector(gray, 0)
        landmarks_x = []
        landmarks_y = []    
        landmarks_x68 = []
        landmarks_y68 = []  
        # get landmarks of the face
        # try:
        for face in detected_faces:
            pose_landmarks = face_pose_predictor(gray, face)
            for i in range(81):
                landmarks_x.append(pose_landmarks.part(i).x)
                landmarks_y.append(pose_landmarks.part(i).y)
            pose_landmarks68 = face_pose_predictor68(gray, face)
            for i in range(68):
                landmarks_x68.append(pose_landmarks68.part(i).x)
                landmarks_y68.append(pose_landmarks68.part(i).y)

        face_top_x = np.r_[landmark_x68[1:17], landmark_x[68:81]]
        # landmark_x[18:81]
        face_top_y = np.r_[landmark_y68[1:17], landmark_y[68:81]]
        shape68 = face_utils.shape_to_np(shape)
        shape81 = face_utils.shape_to_np(shape)
        for (x, y) in pose_landmarks:
		        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        for (x, y) in pose_landmarks68:
		        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

            # frame = skin.apply_foundation(landmarks_x, landmarks_y,landmarks_x68,landmarks_y68, frame2, 100, 20, 50, 81,81,0.8)
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # except Exception as e:
            # print(e)
    # show face with applied makeup
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
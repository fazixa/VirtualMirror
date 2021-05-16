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
from makeup.eyeliner import eyeliner
from makeup.lipstick import lipstick
# from makeup.eye_color import lenses

#initiating camera
prev = 0
frame_rate = 15
detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")
print("[INFO] camera sensor warming up...")
cap = cv2.VideoCapture(1)
time.sleep(2.0)


# applying makup on frames
while True:
    # frame rate and resize frame
    ret, frame = cap.read()
    time_elapsed = time.time() - prev
    # frame = imutils.resize(frame, width = 900)

    try:
        # preparing frame
        prev = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
        # detect faces in frame
        detected_faces = detector(gray, 0)
        landmarks_x = []
        landmarks_y = []    



        #before
        for face in detected_faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
           
            cropped_img = frame2[ y1:y2, x1:x2]
            height, width = frame2.shape[:2]
            cropped_width = x2-x1
            cropped_height = y2-y1
            ratio = 300/cropped_width
            new_width = width*ratio
            cropped_img = imutils.resize(cropped_img, width = 300)

            pose_landmarks = face_pose_predictor(gray, face)   


            for i in range(68):
                landmarks_x.append(int(((pose_landmarks.part(i).x)-x1)*ratio))
                landmarks_y.append(int(((pose_landmarks.part(i).y)-y1)*ratio))
            
            # lip = lipstick(cropped_img)
            eye = eyeliner(cropped_img)
            frame2 = eye.apply_eyeshadow(landmarks_x,landmarks_y,90,20,0,1)
            # frame2 = lip.apply_lipstick(landmarks_x,landmarks_y,100, 20 , 30, "soft", False)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)



            ### after
            frame2 = imutils.resize(frame2, width = cropped_width)
            cheight, cwidth = frame2.shape[:2]
            frame[ y1:y1+cheight, x1:x2] = frame2
            frame = imutils.resize(frame, width = 1000)
       
    except Exception as e:
        print(e)
        

        # except Exception as e:
            # print(e)
    # show face with applied makeup
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
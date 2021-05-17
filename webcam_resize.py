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
from makeup.counsiler import counsiler
# from makeup.eye_color import lenses

#initiating camera
prev = 0
frame_rate = 15
detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")
print("[INFO] camera sensor warming up...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)

padding = 50
face_resized_width = 250

# applying makup on frames
while True:
    # frame rate and resize frame
    ret, frame = cap.read()
    time_elapsed = time.time() - prev
    frame = imutils.resize(frame, width = 1000)

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
            start =time.time()

 

            height, width = frame2.shape[:2]
            orignal_face_width = x2-x1
            ratio = face_resized_width / orignal_face_width
            new_padding = int(padding / ratio)
            new_y1= max(y1-new_padding,0)
            new_y2= min(y2+new_padding,height)
            new_x1= max(x1-new_padding,0)
            new_x2= min(x2+new_padding,width)
            cropped_img = frame2[ new_y1:new_y2, new_x1:new_x2]
            cropped_img = imutils.resize(cropped_img, width = (face_resized_width+2*padding))

            pose_landmarks = face_pose_predictor(gray, face)   


            for i in range(68):
                landmarks_x.append(int(((pose_landmarks.part(i).x)-new_x1)*ratio))
                landmarks_y.append(int(((pose_landmarks.part(i).y)-new_y1)*ratio))
            # lip = lipstick(cropped_img)
            eye = eyeliner(cropped_img)
            # c = counsiler()
            # frame2 = c.apply_blush(cropped_img, landmarks_x, landmarks_y, 100, 20, 40, 51, 51, 0.3)
            frame2 = eye.apply_eyeshadow(landmarks_x,landmarks_y,90,20,0,1)


            frame2 = imutils.resize(frame2, width = new_x2-new_x1)
            cheight, cwidth = frame2.shape[:2]
            frame[ new_y1:new_y1+cheight, new_x1:new_x1+cwidth] = frame2


            frame = frame
       
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
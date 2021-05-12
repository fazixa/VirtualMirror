import imutils
import time
import dlib
import cv2
import numpy as np
from makeup import makeup


lipstick_type = "soft"
gloss = False


detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")


def empty(a):
    pass


print("[INFO] camera sensor warming up...")

frame_rate = 9
prev = 0

cap = cv2.VideoCapture(0)
time.sleep(2.0)
cv2.namedWindow('BGR')
cv2.resizeWindow('BGR', 900, 240)
cv2.createTrackbar('Blue', 'BGR', 34, 255, empty)
cv2.createTrackbar('Green', 'BGR', 21, 255, empty)
cv2.createTrackbar('Red', 'BGR', 124, 255, empty)
previousFrame = None
motion = 1 
moistx = []
moisty = []

while True:
    ret, frame = cap.read()
    time_elapsed = time.time() - prev
 

    if(time_elapsed > 1./frame_rate):
        prev = time.time()


        frame = imutils.resize(frame, width = 700)
        imgOriginal = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.GaussianBlur(gray, (21, 21), 0) 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        b = cv2.getTrackbarPos('Blue', 'BGR')
        g = cv2.getTrackbarPos('Green', 'BGR')
        r = cv2.getTrackbarPos('Red', 'BGR')
        x = []
        y = []


        ####Motion detection########################################################################

        if (previousFrame is None): 
            previousFrame = gray2 
            continue
        diff_frame = cv2.absdiff(previousFrame, gray2) 
    
        # If change in between static background and 
        # current frame is greater than 30 it will show white color(255) 
        thresh_frame = cv2.threshold(diff_frame, 2, 255, cv2.THRESH_BINARY)[1] 
        thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2) 
    
        # Finding contour of moving object 
        cnts, _ = cv2.findContours(thresh_frame.copy(),  
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    
        for contour in cnts: 
            if cv2.contourArea(contour) < 10000: 
                continue
            motion = 1
            print("motion")


        ##########################################################################################
        detected_faces = detector(gray, 0)
        m = makeup(frame)
        dets = detector(gray, 0)
        
        try:
            if (motion == 1):
                for k, d in enumerate(dets):
                    shape = face_pose_predictor(gray, d)

                    i = 0
                    for pt in shape.parts():
                        i = i + 1
                        x.append(pt.x)
                        y.append(pt.y)
                    
                    frame , x2, y2 , moistx, moisty= m.apply_lipstick(x,y,moistx, moisty ,r, g , b, motion, lipstick_type, gloss)
                motion = 0
            else:
                frame , x2, y2 , moistx,moisty= m.apply_lipstick(x2,y2,moistx, moisty, r, g , b, motion,lipstick_type, gloss)
        except Exception as e:
            print(e)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("BGR", frame)

        ##############################################################################################
        previousFrame=gray2.copy()
        ###############################################################################################
    key = cv2.waitKey(1) & 0xFF
    

    if key == ord("q"):
        break
    
cv2.imshow("Original",imgOriginal)
# do a bit of cleanup
cv2.destroyAllWindows()

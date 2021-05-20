"""
 * @author Faezeh Shayesteh
 * @email shayesteh.fa@gmail.com
 * @create date 2021-04-23 03:45:13
 * @modify date 2021-04-23 03:45:27
 * @desc [description]
"""
import cv2
import dlib
import numpy as np
from skimage import io
import imutils
# from makeup.eyeshadow import eyeshadow
from makeup.foundation import Foundation
# from makeup.counsiler import counsiler

# detecting face landmarks of the input image
detector = dlib.get_frontal_face_detector()
detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor("./data/shape_predictor_81_face_landmarks.dat")
face_pose_predictor68= dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")
img = io.imread('./data/input/input.jpg')
img = imutils.resize(img, width = 400)
detected_faces = detector(img, 0)
pose_landmarks = face_pose_predictor(img, detected_faces[0])

height, width = img.shape[:2]
print(height, width)
# preparing landmarks
landmarks_x = []
landmarks_y = []    
landmarks_x68 = []
landmarks_y68 = []  
# get landmarks of the face
# try:
for face in detected_faces:
    pose_landmarks = face_pose_predictor(img, face)
    for i in range(81):
        landmarks_x.append(pose_landmarks.part(i).x)
        landmarks_y.append(pose_landmarks.part(i).y)
    pose_landmarks68 = face_pose_predictor68(img, face)
    for i in range(68):
        landmarks_x68.append(pose_landmarks68.part(i).x)
        landmarks_y68.append(pose_landmarks68.part(i).y)

    skin = Foundation()
    frame = skin.apply_foundation(img, landmarks_x, landmarks_y,landmarks_x68,landmarks_y68, 235, 143, 52, 81,81,0.7)
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
# applying mak
# foundation = foundation()


# writing image
img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
cv2.imshow("i", img)
cv2.waitKey()

# cv2.imwrite('./data/results/counsiler.jpg', frame)
# plt.figure()
# plt.imshow(im)
# plt.show()

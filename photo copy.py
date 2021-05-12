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
from makeup.eyeshadow import eyeshadow
import imutils


# detecting face landmarks of the input image
# detector = dlib.get_frontal_face_detector()
# face_pose_predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")
img = io.imread('./data/input/input.jpg')
# detected_faces = detector(img, 0)
# pose_landmarks = face_pose_predictor(img, detected_faces[0])


face_pose_predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
detected_faces = detector(img, 0)

for faceRect in detected_faces:
    x1 = faceRect.left()
    y1 = faceRect.top()
    x2 = faceRect.right()
    y2 = faceRect.bottom()

cropped_img = img[ y1:y2, x1:x2]
height, width = img.shape[:2]
cropped_width = x2-x1
cropped_height = y2-y1
ratio = 300/cropped_width
new_width = width*ratio
landmarks_x = []
landmarks_y = []    


# detected_faces = detector(cropped_img, 0)
cropped_img = imutils.resize(cropped_img, width = 300)

for face in detected_faces:
    pose_landmarks = face_pose_predictor(img, face)
    landmarks_x = []
    landmarks_y = []
    for i in range(68):
        landmarks_x.append(int(((pose_landmarks.part(i).x)-x1)*ratio))
        landmarks_y.append(int(((pose_landmarks.part(i).y)-y1)*ratio))


eyeshadow = eyeshadow(cropped_img)
cropped_img = eyeshadow.apply_eyeshadow(landmarks_x, landmarks_y, 100, 20, 50, 0.8)
# first method
# x2  = int(x2*ratio)
# x1 = int(x1*ratio)
# y2  = int(y2*ratio)
# y1 = int(y1*ratio)
# frame = imutils.resize(img, width = int(new_width))
# frame[y1:y2-1, x1:x2] = cropped_img

#second method
cropped_img = imutils.resize(cropped_img, width = cropped_width)
cheight, cwidth = cropped_img.shape[:2]
img[ y1:y1+cheight, x1:x2] = cropped_img


cv2.imshow("Cropped Image", img)
cv2.waitKey(0)


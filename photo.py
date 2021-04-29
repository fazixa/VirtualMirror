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

# detecting face landmarks of the input image
detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")
img = io.imread('./data/input/input.jpg')
detected_faces = detector(img, 0)
pose_landmarks = face_pose_predictor(img, detected_faces[0])

# preparing landmarks
landmarks_x = []
landmarks_y = []
for i in range(68):
    landmarks_x.append(pose_landmarks.part(i).x)
    landmarks_y.append(pose_landmarks.part(i).y)

# applying makeup
eyeshadow = eyeshadow(img)
img = eyeshadow.apply_eyeshadow(landmarks_x, landmarks_y, 100, 20, 50, 0.8)

# writing image
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite('./data/results/eyeshadow.jpg', img)
# plt.figure()
# plt.imshow(im)
# plt.show()

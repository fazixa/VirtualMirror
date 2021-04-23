/**
 * @author Faezeh Shayesteh
 * @email shayesteh.fa@gmail.com
 * @create date 2021-04-23 03:45:13
 * @modify date 2021-04-23 03:45:27
 * @desc [description]
 */

import cv2
import dlib
import numpy as np
from skimage import io
from makeup.eyeshadow import eyeshadow

# detecting face lanmarks of the input image
detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")
img = io.imread('./data/input.jpg')
detected_faces = detector(img, 0)
pose_landmarks = face_pose_predictor(img, detected_faces[0])

landmark = np.empty([68, 2], dtype=int)
for i in range(68):
    x.append(pose_landmarks.part(i).x)
    y.append(pose_landmarks.part(i).y)

m = eyeshadow(img)
im = m.apply_eyeshadow(landmark)
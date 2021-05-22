import cv2
import math
from pylab import mean
from skimage import color
import mediapipe as mp
import imutils
from typing import List, Tuple, Union
from skimage.draw import line, polygon, polygon_perimeter
import numpy as np
from numpy import c_
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
VISIBILITY_THRESHOLD = 0.5
PRESENCE_THRESHOLD = 0.5




drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)


# LANDMARKS
LOWER_LIP_INNER = [308, 324, 318,402, 317, 14, 87, 178, 88, 95, 78]
LOWER_lIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]


UPPER_LIP_OUTER = [61, 185, 40, 39, 37, 11, 267, 269, 270, 409, 291]
UPPER_LIP_INEER = [308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]


LEFT_EYE_UPPER = [263, 249, 390, 373, 374, 380, 381, 382, 362]
LEFT_EYE_LOWER = [362, 398, 384, 385, 386, 387, 388, 466, 263]

LEFT_EYEBROW_LOWER = [276, 283, 282, 295, 285]

LEFT_CHEEK_BONE = [454, 447, 345, 280, 425, 426, 436, 432, 430, 394, 379, 365, 397, 288, 361]



# POLYGONS
LOWER_LIP = LOWER_lIP_OUTER + LOWER_LIP_INNER
UPPER_LIP = LOWER_lIP_OUTER + UPPER_LIP_INEER
# LEFT_EYE = [LEFT_EYE_LOWER + LEFT_EYE_UPPER]
LEFT_EYESHADOW = LEFT_EYE_LOWER + LEFT_EYEBROW_LOWER


# MAKEUPS
LIPS = [LOWER_LIP, UPPER_LIP]
EYESHADOW = [LEFT_EYESHADOW]
CONCEALER = [LEFT_CHEEK_BONE]

# ALL MAKEUPS
MAKEUP = [LIPS, EYESHADOW]





M=[LIPS]
FACE_CONNECTIONS = frozenset([
    # Lips.
    # (61, 146),
    # (146, 91),
    # (91, 181),
    # (181, 84),
    # (84, 17),
    # (17, 314),
    # (314, 405),
    # (405, 321),
    # (321, 375),
    # (375, 291),
    # (61, 185),
    # (185, 40),
    # (40, 39),
    # (39, 37),
    # (37, 0),
    # (0, 267),
    # (267, 269),
    # (269, 270),
    # (270, 409),
    # (409, 291),
    # (78, 95),
    # (95, 88),
    # (88, 178),
    # (178, 87),
    # (87, 14),
    # (14, 317),
    # (317, 402),
    # (402, 318),
    # (318, 324),
    # (324, 308),
    # (78, 191),
    # (191, 80),
    # (80, 81),
    # (81, 82),
    # (82, 13),
    # (13, 312),
    # (312, 311),
    # (311, 310),
    # (310, 415),
    # (415, 308),
    # # Left eye.
    # (263, 249),
    # (249, 390),
    # (390, 373),
    # (373, 374),
    # (374, 380),
    # (380, 381),
    # (381, 382),
    # (382, 362),
    # (263, 466),
    # (466, 388),
    # (388, 387),
    # (387, 386),
    # (386, 385),
    # (385, 384),
    # (384, 398),
    # (398, 362),
    # # Left eyebrow.
    (276, 283),
    (283, 282),
    (282, 295),
    (295, 285),
    # (300, 293),
    # (293, 334),
    # (334, 296),
    # (296, 336),
    # # Right eye.
    # (33, 7),
    # (7, 163),
    # (163, 144),
    # (144, 145),
    # (145, 153),
    # (153, 154),
    # (154, 155),
    # (155, 133),
    # (33, 246),
    # (246, 161),
    # (161, 160),
    # (160, 159),
    # (159, 158),
    # (158, 157),
    # (157, 173),
    # (173, 133),
    # # Right eyebrow.
    # (46, 53),
    # (53, 52),
    # (52, 65),
    # (65, 55),
    # (70, 63),
    # (63, 105),
    # (105, 66),
    # (66, 107),
    # # Face oval.
    # (10, 338),
    # (338, 297),
    # (297, 332),
    # (332, 284),
    # (284, 251),
    # (251, 389),
    # (389, 356),
    # (356, 454),
    # (454, 323),
    # (323, 361),
    # (361, 288),
    # (288, 397),
    # (397, 365),
    # (365, 379),
    # (379, 378),
    # (378, 400),
    # (400, 377),
    # (377, 152),
    # (152, 148),
    # (148, 176),
    # (176, 149),
    # (149, 150),
    # (150, 136),
    # (136, 172),
    # (172, 58),
    # (58, 132),
    # (132, 93),
    # (93, 234),
    # (234, 127),
    # (127, 162),
    # (162, 21),
    # (21, 54),
    # (54, 103),
    # (103, 67),
    # (67, 109),
    # (109, 10)
])





# UPPER_LIPS = []

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def apply_color(image, x, y):
    im_copy = image.copy()
    height, width = image.shape[:2]
    r = 20
    g = 20
    b = 200
    intensity = 0.9

    # converting desired parts of the original image to LAB color space
    lip_LAB = color.rgb2lab((im_copy[x, y] / 255.).reshape(len(x), 1, 3)).reshape(len(x), 3)
    # calculating mean of each channel
    L, A, B = mean(lip_LAB[:, 0]), mean(lip_LAB[:, 1]), mean(lip_LAB[:, 2])
    # converting the color of the makeup to LAB
    L1, A1, B1 = color.rgb2lab(np.array((r / 255., g / 255., b / 255.)).reshape(1, 1, 3)).reshape(
        3, )
    # applying the makeup color on image
    # L1, A1, B1 = color.rgb2lab(np.array((self.r / 255., self.g / 255., self.b / 255.)).reshape(1, 1, 3)).reshape(3, )
    G = L1 / L
    lip_LAB = lip_LAB.reshape(len(x), 1, 3)
    lip_LAB[:, :, 1:3] = intensity * np.array([A1, B1]) + (1 - intensity) * lip_LAB[:, :, 1:3]
    lip_LAB[:, :, 0] = lip_LAB[:, :, 0] * (1 + intensity * (G - 1))
    # converting back toRGB
    # print(self.r,self.g,self.b)
    im_copy[x, y] = color.lab2rgb(lip_LAB).reshape(len(x), 3) * 255

    return im_copy


def apply_blur(image, image2, x, y):
    # image is orignal
    # image2 is withe applied color
    # image2 = image.copy()
    height, width = image.shape[:2]
    filter = np.zeros((height, width))
    cv2.fillConvexPoly(filter, np.array(c_[y, x], dtype='int32'), 1)
    # # for x_cor in x:
    # filter[x, y] =1
    
    # Erosion to reduce blur size

    filter = cv2.GaussianBlur(filter, (51, 51), 0)
    kernel = np.ones((30,15), np.uint8)
    filter = cv2.erode(filter, kernel, iterations=1)

    
    alpha = np.zeros([height, width, 3], dtype='float64')

    alpha[:, :, 0] = filter
    alpha[:, :, 1] = filter
    alpha[:, :, 2] = filter



    im_copy = (alpha * image2 + (1 - alpha) *image).astype('uint8')
    # im_copy = alpha.astype('uint8')
    return im_copy

with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    image = imutils.resize(image, width = 1000)
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    
    # if results.multi_face_landmarks:
    #   for face_landmarks in results.multi_face_landmarks:
    #     mp_drawing.draw_landmarks(
    #         image=image,
    #         landmark_list=face_landmarks,
    #         connections=mp_face_mesh.FACE_CONNECTIONS,
    #         landmark_drawing_spec=drawing_spec,
    #         connection_drawing_spec=drawing_spec)




    # landmark_list  = face_landmarks


    if results.multi_face_landmarks:
      for landmark_list in results.multi_face_landmarks:
        eye_x = []
        eye_y = []

        image_rows, image_cols, _ = image.shape
        idx_to_coordinates = {}
        for idx, landmark in enumerate(landmark_list.landmark):
            
            if ((landmark.HasField('visibility') and
                landmark.visibility < VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                landmark.presence < PRESENCE_THRESHOLD)):
                continue
            landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                        image_cols, image_rows)

            if landmark_px:
                idx_to_coordinates[idx] = landmark_px

        for i in range(len(idx_to_coordinates.values())):
            cv2.circle(image, (idx_to_coordinates[i][0],idx_to_coordinates[i][1]), 1,
            (0, 255, 0), 1)
            cv2.putText(image, str(i), (idx_to_coordinates[i][0],idx_to_coordinates[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 
               0.3, (255, 255, 0), 1, cv2.LINE_AA)


        for region in LIPS:
            for point in region:
                eye_x.append(idx_to_coordinates[point][0])
                eye_y.append(idx_to_coordinates[point][1])

            if FACE_CONNECTIONS:
                num_landmarks = len(landmark_list.landmark)
                # Draws the connections if the start and end landmarks are both visible.
                for connection in FACE_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                        raise ValueError(f'Landmark index is out of range. Invalid connection '
                                    f'from landmark #{start_idx} to landmark #{end_idx}.')
                    if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                        A= 1
                        cv2.putText(image, str(start_idx), (idx_to_coordinates[start_idx][0],idx_to_coordinates[start_idx][1]), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.3, (255, 255, 0), 1, cv2.LINE_AA)
                        rr, cc = line(idx_to_coordinates[start_idx][1],idx_to_coordinates[start_idx][0], idx_to_coordinates[end_idx][1], idx_to_coordinates[end_idx][0])

                        image[rr,cc]= (255,255,255)


            margin = 40
            top_x = min(eye_x)-margin
            top_y = min(eye_y)-margin
            bottom_x = max(eye_x)+margin
            bottom_y = max(eye_y)+margin


            rr, cc = polygon_perimeter(eye_x, eye_y)
            # image[cc,rr,1] = 255 
            rr, cc = polygon(rr, cc)

            crop = image [top_y:bottom_y, top_x:bottom_x, ]
            crop_colored = apply_color(crop, cc-top_y,rr-top_x)

            # crop = image [top_y:bottom_y, top_x:bottom_x, ]
            # crop1 = image1 [top_y:bottom_y, top_x:bottom_x, ]

            image2 = apply_blur(crop,crop_colored,cc-top_y,rr-top_x)

            image [top_y:bottom_y, top_x:bottom_x, ] =image2

            # image[cc,rr,1] = 255 

            # cv2.fillConvexPoly(image, eye, (255, 255, 255))

    
    cv2.imshow('MediaPipe FaceMesh', image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cap.release()
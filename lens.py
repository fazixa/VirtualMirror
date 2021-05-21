from re import X
import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
import time
import cv2
from pylab import *
import numpy as np
from numpy import c_
from skimage import color
from irisSeg import irisSeg
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d, splprep, splev
def eyeAspectRatio(points):
    A = dist.euclidean(points[1], points[5])
    B = dist.euclidean(points[2], points[4])
    C = dist.euclidean(points[0], points[3])
    
    return (A + B) / (2.0 * C)

def getROI(frame, image, landmarks, eye):
    if eye == 0:
        points = [36, 37, 38, 39, 40, 41]
    else:
        points = [42, 43, 44, 45, 46, 47]
        
    region = np.array([[landmarks.part(point).x, landmarks.part(point).y] for point in points])
    margin = 17
    
    left = np.min(region[:, 0])
    top = np.min(region[:, 1])
    right = np.max(region[:, 0])
    bottom = np.max(region[:, 1])
    
    height = abs(top - bottom)
    width = abs(left - right)	
    grayEye = image[top-10:bottom+10, left-10:right+10]

    coord_iris, coord_pupil, output_image = irisSeg(grayEye, 40, 70)
    print(coord_iris) # radius and the coordinates for the center of iris 
    print(coord_pupil) # radius and the coordinates for the center of pupil 
    # plt.imshow(output_image)
    # plt.show()
    cv2.imshow('ddd', output_image)
    # cv2.imwrite('file.jpg', grayEye)
    roi = frame[top:bottom, left+margin:right-margin]
    thresh = calibrate(grayEye)
    _, threshEye = cv2.threshold(grayEye, thresh, 255, cv2.THRESH_BINARY)
    prepEye = preprocess(threshEye)
    x, y = getIris(prepEye, roi)
    eye_radius = np.uint16((landmarks.part(38).x - landmarks.part(37).x) / 2)
    center= np.empty([2], dtype=int)
    center[0] = x+left+3
    center[1] = y+top+3


    # for x, y in zip(x_fill, y_fill):
    #     frame = cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    #text = str((x*left)/(width*100.0))
    #cv2.putText(frame, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    #print(height)
    # cv2.circle(frame, (x+left, y+top), 3, (0, 255, 0), -1)
    
    ear = eyeAspectRatio(region)
    
    return center, eye_radius

def fill(r, center):
    points_1 = [center[0] - r, center[1]]
    points_2 = [center[0], center[1] - r]
    points_3 = [center[0] + r, center[1]]
    points_4 = [center[0], center[1] + r]
    points_5 = points_1

    points = np.array([points_1, points_2, points_3, points_4, points_5])

    x, y = points[0:5, 0], points[0:5, 1]

    tck, u = splprep([x, y], s=0, per=1)
    unew = np.linspace(u.min(), u.max(), 1000)
    xnew, ynew = splev(unew, tck, der=0)
    tup = c_[xnew.astype(int), ynew.astype(int)].tolist()
    coord = list(set(tuple(map(tuple, tup))))
    coord = np.array([list(elem) for elem in coord])
    return np.array(coord[:, 0], dtype=np.int32), np.array(coord[:, 1], dtype=np.int32)
def get_interior_points( x, y):
    intx = []
    inty = []
    print('start get_interior_points')

    def ext(a, b, i):
        a, b = round(a), round(b)
        intx.extend(arange(a, b, 1).tolist())
        inty.extend((ones(b - a) * i).tolist())

    x, y = np.array(x), np.array(y)
    print('x,y get_interior_points')
    xmin, xmax = amin(x), amax(x)
    xrang = np.arange(xmin, xmax + 1, 1)
    print(type(xrang))
    print('x-rang')
    print(xrang)
    for i in xrang:
        try:
            ylist = y[where(x == i)]
            ext(amin(ylist), amax(ylist), i)
        except ValueError:  # raised if `y` is empty.
            pass

    print('xrang2 get_interior_points')
    return np.array(intx, dtype=np.int32), np.array(inty, dtype=np.int32)


def apply_color( y, x, frame):
    r = 20
    g = 220
    b = 20
    intensity = 0.3

    x1 = []
    y1 = []

    for i, (x_, y_) in enumerate(zip(x, y)):
        if frame[x_,y_,0]<80:
            x1.append(x_)
            y1.append(y_)


    print("________________________________________",x, x.shape)
    # t = np.array(list(zip(x, y)))
    # t = t[frame[t[0, :]] > 200]
    # x = t[0, :]
    # y = t[:, 0]
                    # converting desired parts of the original image to LAB color space
    lip_LAB = color.rgb2lab((frame[x1, y1] / 255.).reshape(len(x1), 1, 3)).reshape(len(x1), 3)
    # calculating mean of each channel
    L, A, B = mean(lip_LAB[:, 0]), mean(lip_LAB[:, 1]), mean(lip_LAB[:, 2])
    # converting the color of the makeup to LAB
    L1, A1, B1 = color.rgb2lab(np.array((r / 255., g / 255.,b / 255.)).reshape(1, 1, 3)).reshape(
        3, )
    # applying the makeup color on image
    # L1, A1, B1 = color.rgb2lab(np.array((self.r / 255., self.g / 255., self.b / 255.)).reshape(1, 1, 3)).reshape(3, )
    G = L1 / L
    lip_LAB = lip_LAB.reshape(len(x1), 1, 3)
    lip_LAB[:, :, 1:3] = intensity * np.array([A1, B1]) + (1 - intensity) * lip_LAB[:, :, 1:3]
    lip_LAB[:, :, 0] = lip_LAB[:, :, 0] * (1 + intensity * (G - 1))
    # converting back toRGB
    # print(self.r,self.g,self.b)
    frame[x1,y1] = color.lab2rgb(lip_LAB).reshape(len(x1), 3) * 255
    # cv2.imshow("Frame", frame)
    
    return frame

def apply_blur( y, x, frame, farme_c):
    # gussian blur
    height,width = frame.shape[:2]
    filter = np.zeros((height,width))
    
    # kernel = np.ones((15, 15), np.uint8)
    # filter = cv2.erode(filter, kernel, iterations=1)
    cv2.fillConvexPoly(filter, np.array(c_[y, x], dtype='int32'), 1)
    filter = cv2.GaussianBlur(filter, (21, 21), 0)
    # Erosion to reduce blur size
    kernel = np.ones((2,2), np.uint8)
    filter = cv2.erode(filter, kernel, iterations=1)
    
    alpha = np.zeros([height, width, 3], dtype='float64')
    alpha[:, :, 0] = filter
    alpha[:, :, 1] = filter
    alpha[:, :, 2] = filter
    frame = (alpha * frame + (1 - alpha) * frame_c).astype('uint8')   
    # frame = alpha.astype('uint8')   
    return frame


def apply_blur2(y, x, frame):
    intensity = 0.1
    r = 20
    g = 220
    b = 20
    # Create blush shape
    height,width = frame.shape[:2]
    mask = np.zeros((height, width))
    cv2.fillConvexPoly(mask, np.array(c_[y, x], dtype='int32'), 250)
    
    mask = mask* intensity
    # kernel = np.ones((15, 15), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=1)
    # print(np.array(c_[x_right, y_right])[:, 0])


    # alpha = np.zeros([height, width, 3], dtype='float64')
    # alpha[:, :, 0] = mask
    # alpha[:, :, 1] = mask
    # alpha[:, :, 2] = mask


    # frame = alpha.astype('uint8') 
    val = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(float)

    val[:, :, 0] = val[:, :, 0] / 255. * 100.
    val[:, :, 1] = val[:, :, 1] - 128.
    val[:, :, 2] = val[:, :, 2] - 128.
    print(r)
    LAB = color.rgb2lab(np.array((float(r) / 255., float(g) / 255., float(b) / 255.)).reshape(1, 1, 3)).reshape(3, )
    mean_val = np.mean(np.mean(val, axis=0), axis=0)


    mask = np.array([mask, mask, mask])
    mask = np.transpose(mask, (1, 2, 0))

    lab = np.multiply((LAB - mean_val), mask)

    val[:, :, 0] = np.clip(val[:, :, 0] + lab[:, :, 0], 0, 100)
    val[:, :, 1] = np.clip(val[:, :, 1] + lab[:, :, 1], -127, 128)
    val[:, :, 2] = np.clip(val[:, :, 2] + lab[:, :, 2], -127, 128)
    
    frame = (color.lab2rgb(val) * 255).astype(np.uint8)
    return frame

def getSize(eye, t):
    height, width = eye.shape
    _, thresh = cv2.threshold(eye, t, 255, cv2.THRESH_BINARY)
    n_pixels = height*width
    #print(n_pixels)
    
    black_pixels = n_pixels - cv2.countNonZero(thresh)
    #print("->", black_pixels)
    try:
        ratio = black_pixels * 1.0 / n_pixels
        return ratio
    except ZeroDivisionError:
        return None
    

def calibrate(eye):
    iris_size = 0.48
    trials = {}
    
    for t in range(5, 100, 5):
        trials[t] = getSize(eye, t)
    
    try:
        best_threshold, size = min(trials.items(), key = lambda x : abs(x[1] - iris_size))
        #print(best_threshold, size)
        return best_threshold
    except TypeError:
        return None
    

def preprocess(image):
    kernel = np.array([[0., 1., 0.], [1., 2., 1.], [0., 1., 0.]], dtype = np.uint8)
    blur = cv2.bilateralFilter(image, 5, 10, 10)
    #leftEroded = cv2.erode(leftBlur, kernel, iterations = 1) 
    dilated = cv2.dilate(blur, kernel)
    return cv2.bitwise_not(dilated)


def getIris(image, roi):
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = sorted(contours, key = cv2.contourArea, reverse = True)
    margin = 5
    #return max_contour
    #for contour in contours: 
    #	cv2.drawContours(roi, contour, -1, (255, 0, 0), 2)
    #cv2.drawContours(roi, max_contour, -1, (255, 0, 0), 2)
    try:
        max_contour = all_contours[0]
        M = cv2.moments(max_contour)
        x = int(M['m10'] / M['m00']) + margin
        y = int(M['m01'] / M['m00'])
        roi = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        cv2.circle(roi, (x, y), 3, (0, 0, 255), -1)
        #cv2.imshow("ROI", roi)
        return x, y
    except (IndexError, ZeroDivisionError):
        return 0, 0
        
    
    
def printText(frame, text):
    width, height, _ = frame.shape
    cv2.putText(frame, text, (width // 2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")
    total = 0
    previousRatio = 1
    while True:
        retr, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_c = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            faces = detector(gray)
        
            landmarks = predictor(gray, faces[0])
            #cv2.circle(frame, (landmarks.part(0).x, landmarks.part(1).y), 3, (255, 0, 0), -1)
        except: 
            continue
        margin = 7
        center , r = getROI(frame, gray, landmarks, 0)
        
        # getROI(frame, gray, landmarks, 1)


        x_fill, y_fill = fill(r, center )
        y_fill, x_fill = get_interior_points(x_fill, y_fill)
        frame = apply_color(x_fill, y_fill, frame)
        frame = apply_blur(x_fill, y_fill, frame, frame_c)

        # cv2.imshow("ff",frame)
            
        # avgEAR = (Lear + Rear) / 2.0
        
        # avgHori = (Lhori + Rhori) / 2.0
        # avgVerti = (Lverti + Rverti) / 2.0
        
        # if avgHori < 0.8:
        #     printText(frame, "LEFT")
        # elif avgHori > 1.70:
        #     printText(frame, "RIGHT")
        # elif avgVerti < 0.60:
        #     printText(frame, "UP")
        # else:
        #     printText(frame, "CENTER")
        
        # if(avgEAR < 0.20):
        #     if(previousRatio >= 0.20):
        #         total += 1
        # previousRatio = avgEAR
            
        # cv2.putText(frame, "Counter: " + str(total), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # cv2.imshow("Frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()	
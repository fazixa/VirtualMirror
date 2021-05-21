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
import math


def houghTransformCircleDetector(image, eye_radius, eye_distance):
    # parameters
    edge_sensitivity_threshold = 450
    edge_accumulator_threshold = 10
    inverse_accumulator_ratio = 1
    min_circle_distance = eye_distance / 2
    # Convert to gray-scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray =image
    # img = cv2.medianBlur(gray, 5)
    # Apply hough transform on the images
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        inverse_accumulator_ratio,
        min_circle_distance,
        param1=edge_sensitivity_threshold,
        param2=edge_accumulator_threshold,
        minRadius=int(eye_radius - eye_radius / 4),
        maxRadius=int(eye_radius + eye_radius / 2),
    )
    return circles

def filterOutCircles(circles, region_x, region_y):
    # circles = np.uint16(np.around(circles))
    # average x & y of eye region landmarks
    left_distances = []
    right_distances = []
    left_x = left_y = right_x = right_y = 0



    for x in region_x:
        left_x += x

    for y in region_y:
        left_y += y

    # for i in rightEyeLandmarks:
    #     right_x += landmarks[i].x
    #     right_y += landmarks[i].y


    left_x /= 4
    right_x /= 4
    left_y /= 4
    right_y /= 4
    # find the circle with the smallest distance from the left & right eye region
    for circle in circles[0, :]:
        left_distances.append(
            math.sqrt((circle[0] - left_x) ** 2 + (circle[1] - left_y) ** 2)
        )
        right_distances.append(
            math.sqrt((circle[0] - right_x) ** 2 + (circle[1] - right_y) ** 2)
        )
    left_eye = circles[0, :][left_distances.index(min(left_distances))]
    right_eye = circles[0, :][right_distances.index(min(right_distances))]
    return [left_eye, right_eye]


def createHistogramOfColorsInEyeRegion(img, circles, eye_radius, test=False):
    # Convert image to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert image to float so to create a mask
    gray = gray / 255.0
    for circle in circles:
        center = (circle[0], circle[1])
        # get all pixels inside eye radius
        cv2.circle(gray, center, int(eye_radius + eye_radius / 2), 2, -1)
    if test:
        cv2.imshow('',gray)
    eye_pixels = np.where(gray == 2)
    # get histrogram of colors for eye_pixels
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    eye_pixel_hue = h[eye_pixels]
    # h[0]/h[180] = red, h[60] = green, h[120] = blue
    histogram = np.histogram(eye_pixel_hue, np.arange(180))
    return histogram, eye_pixels, eye_pixel_hue, h, s, v


def findLargestNConsecutiveBins(histogram, n):
    bins = histogram[0]
    maxSum = 0
    maxBins = []
    lengthOfBins = len(bins)
    for i in range(0, lengthOfBins):
        # max numbers can start at the end of the array and wrap to the beggining so we need to account for those
        # ex: [44, 2, 3, 40, 42] -> we need two arrays [0] & [3:4]
        overflow = i + n > lengthOfBins
        maxNum = i + n if not overflow else lengthOfBins
        maxNum2 = 0 if not overflow else (i + n) % lengthOfBins
        binRange = np.concatenate((np.arange(i, maxNum), np.arange(0, maxNum2)))
        newSum = sum(bins[binRange])

        if newSum > maxSum:
            maxSum = newSum
            maxBins = binRange

    return maxBins


def changeHueOfHistogram(
    histogram, eye_pixels, eye_pixel_hue, h, s, v, color, test=False
):
    largestBins = findLargestNConsecutiveBins(histogram, 30)
    if test:
        print(histogram[0])
        print(largestBins)
    if color == "brown":
        destinationHue = np.concatenate((np.arange(170, 180), np.arange(0, 20)))
    elif color == "blue":
        destinationHue = np.arange(100, 131)
    elif color == "green":
        destinationHue = np.arange(40, 71)
    # loop through the larget bins of hue colors, and map them to the destination color
    for i, bin in enumerate(largestBins):
        eye_pixel_hue[eye_pixel_hue == bin] = destinationHue[i]
    # update hue of eye pixel region
    h[eye_pixels] = eye_pixel_hue
    # recreate hsv image with updated hues, covert back to BGR and display
    newImage = cv2.merge([h, s, v])
    newImage = cv2.cvtColor(newImage, cv2.COLOR_HSV2BGR)
    # showImage(newImage)
    return newImage



# detecting face landmarks of the input image
detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")
img = io.imread('./data/input/input.jpg')
detected_faces = detector(img, 0)
pose_landmarks = face_pose_predictor(img, detected_faces[0])

eye_points = [36, 37, 38, 39, 40, 41]
margin = 10
# preparing landmarks
landmarks_x = []
landmarks_y = []
for i in range(68):
    landmarks_x.append(pose_landmarks.part(i).x)
    landmarks_y.append(pose_landmarks.part(i).y)


region_x = landmarks_x[36:42]
region_y = landmarks_y[36:42]

region_x = landmarks_x[43:47]
region_y = landmarks_y[43:47]

left = np.min(region_x)
top = np.min(region_y)
right = np.max(region_x)
bottom = np.max(region_y)
eye_radius = ((landmarks_x[38] - landmarks_x[37]) / 2)
eye_distance = np.uint16(landmarks_x[43] - landmarks_x[38])
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eye = img[top-margin:bottom+margin, left-margin:right+margin]

circles = houghTransformCircleDetector(eye, eye_radius, eye_distance)


circles = filterOutCircles(circles, region_x,region_y)

print(circles)

histogram, eye_pixels, eye_pixel_hue, h, s, v = createHistogramOfColorsInEyeRegion(
    eye, circles, eye_radius, False
)

eye_color = 'green'
img = changeHueOfHistogram(
            histogram, eye_pixels, eye_pixel_hue, h, s, v, eye_color, True
        )


# applying makeup
# eyeshadow = eyeshadow(img)
# img = eyeshadow.apply_eyeshadow(landmarks_x, landmarks_y, 100, 20, 50, 0.8)

# writing image
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow("Cropped Image", img)
cv2.waitKey(0)


cv2.imwrite('./data/results/eyeshadow.jpg', img)
# plt.figure()
# plt.imshow(im)
# plt.show()



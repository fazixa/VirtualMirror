

from skimage import io
import cv2
img = io.imread('./data/input/input.jpg')
height, width = img.shape[:2]
img = img[0:10, 0:100]

cv2.imshow("Cropped Image", img)
cv2.waitKey(0)
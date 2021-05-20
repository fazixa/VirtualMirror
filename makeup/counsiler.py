from __future__ import division
from itertools import zip_longest
# import scipy.interpolate
from scipy.interpolate import interp1d
import cv2
import numpy as np
from skimage import color
from PIL import Image
from imutils import face_utils
import os
import dlib
from pylab import *
from skimage import io

from scipy import interpolate


class Concealer(object): 


    def apply_blush(self, img, landmark_x, landmark_y, r_value, g_value, b_value, ksize_h, ksize_w, intensity):
        self.r= int(r_value)
        self.g = int(g_value)
        self.b = int(b_value)
        self.intensity = intensity
        self.image = img
        self.im_copy = img.copy()
        self.height, self.width = self.image.shape[:2]

        indices_left = [1, 2, 3, 4, 48, 31, 36]
        # indices_face_bottom = range(1, 27)
        left_cheek_x = [landmark_x[i] for i in indices_left]
        left_cheek_y = [landmark_y[i] for i in indices_left]

        left_cheek_x, left_cheek_y = self.get_boundary_points(left_cheek_x, left_cheek_y)
        left_cheek_y, left_cheek_x = self.get_interior_points(left_cheek_x, left_cheek_y)


        indices_right = [15, 14, 13, 12, 54, 35, 45]
        right_cheek_x = [landmark_x[i] for i in indices_right]
        right_cheek_y = [landmark_y[i] for i in indices_right]
        right_cheek_x, right_cheek_y = self.get_boundary_points(right_cheek_x, right_cheek_y)
        right_cheek_y, right_cheek_x = self.get_interior_points(right_cheek_x, right_cheek_y)

        self.y_all = np.concatenate((left_cheek_x, right_cheek_x))
        self.x_all = np.concatenate((left_cheek_y, right_cheek_y))

        self.apply_color(self.x_all, self.y_all )
        self.__smoothen_blush(self.x_all, self.y_all )

        # self.apply_color(left_cheek_x,left_cheek_y )
        # self.apply_blur(left_cheek_x, left_cheek_y )
        # self.apply_color(right_cheek_x, right_cheek_y )
        # self.apply_blur(right_cheek_x, right_cheek_y )

        return self.im_copy
        
    def get_boundary_points(self, x, y):
        tck, u = interpolate.splprep([x, y], s=0, per=1)
        unew = np.linspace(u.min(), u.max(), 1000)
        xnew, ynew = interpolate.splev(unew, tck, der=0)
        tup = c_[xnew.astype(int), ynew.astype(int)].tolist()
        coord = list(set(tuple(map(tuple, tup))))
        coord = np.array([list(elem) for elem in coord])
        return np.array(coord[:, 0], dtype=np.int32), np.array(coord[:, 1], dtype=np.int32)

    def get_interior_points(self, x, y):
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
  

        for i in xrang:
            try:
                ylist = y[where(x == i)]
                ext(amin(ylist), amax(ylist), i)
            except ValueError:  # raised if `y` is empty.
                pass

        print('xrang2 get_interior_points')
        return np.array(intx, dtype=np.int32), np.array(inty, dtype=np.int32)

    def apply_color(self, x, y):
        # converting desired parts of the original image to LAB color space
        lip_LAB = color.rgb2lab((self.im_copy[x, y] / 255.).reshape(len(x), 1, 3)).reshape(len(x), 3)
        # calculating mean of each channel
        L, A, B = mean(lip_LAB[:, 0]), mean(lip_LAB[:, 1]), mean(lip_LAB[:, 2])
        # converting the color of the makeup to LAB
        L1, A1, B1 = color.rgb2lab(np.array((self.r / 255., self.g / 255., self.b / 255.)).reshape(1, 1, 3)).reshape(
            3, )
        # applying the makeup color on image
        # L1, A1, B1 = color.rgb2lab(np.array((self.r / 255., self.g / 255., self.b / 255.)).reshape(1, 1, 3)).reshape(3, )

        G = L1 / L
        lip_LAB = lip_LAB.reshape(len(x), 1, 3)
        lip_LAB[:, :, 1:3] = self.intensity * np.array([A1, B1]) + (1 - self.intensity) * lip_LAB[:, :, 1:3]
        lip_LAB[:, :, 0] = lip_LAB[:, :, 0] * (1 + self.intensity * (G - 1))
        # converting back toRGB
        # print(self.r,self.g,self.b)
        self.im_copy[x, y] = color.lab2rgb(lip_LAB).reshape(len(x), 3) * 255

        # self.im_copy = cv2.cvtColor(self.im_copy, cv2.COLOR_BGR2RGB)
        # cv2.imwrite('./eyeshadow2.jpg', self.im_copy)


    def apply_blur(self, x, y):
        # gussian blur
        filter = np.zeros((self.height, self.width))
        cv2.fillConvexPoly(filter, np.array(c_[y, x], dtype='int32'), 1)
        
        # Erosion to reduce blur size
        kernel = np.ones((25,25), np.uint8)
        filter = cv2.erode(filter, kernel, iterations=1)
        filter = cv2.GaussianBlur(filter, (91, 91), 0)
        alpha = np.zeros([self.height, self.width, 3], dtype='float64')
        alpha[:, :, 0] = filter
        alpha[:, :, 1] = filter
        alpha[:, :, 2] = filter
        self.im_copy = (alpha * self.im_copy + (1 - alpha) * self.image).astype('uint8')
        # self.im_copy = alpha.astype('uint8')

    def __smoothen_blush(self, x, y):
        # imgBase = np.zeros((self.height, self.height))
        # cv2.fillConvexPoly(imgBase, np.array(np.c_[x, y], dtype='int32'), 1)
        # imgMask = cv2.GaussianBlur(imgBase, (81, 81), 0)

        # imgBlur3D = np.ndarray(
        #     [self.height, self.width, 3], dtype='float')
        # imgBlur3D[:, :, 0] = imgMask
        # imgBlur3D[:, :, 1] = imgMask
        # imgBlur3D[:, :, 2] = imgMask
        # self.image_copy = (
        #     imgBlur3D*self.image + (1 - imgBlur3D)*self.image_copy).astype('uint8')

        img_base = np.zeros((self.height, self.width))
        cv2.fillConvexPoly(img_base, np.array(
            np.c_[x, y], dtype='int32'), 1)
        img_mask = cv2.GaussianBlur(
            img_base, (51, 51), 0)  # 51,51 81,81
        img_blur_3d = np.ndarray(
            [self.height, self.width, 3], dtype='float')
        img_blur_3d[:, :, 0] = img_mask
        img_blur_3d[:, :, 1] = img_mask
        img_blur_3d[:, :, 2] = img_mask
        self.im_copy = (
            img_blur_3d * self.image + (1 - img_blur_3d) * self.im_copy).astype('uint8')
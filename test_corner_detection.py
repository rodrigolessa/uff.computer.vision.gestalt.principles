# Cast Matlab R2013a to Python 3.7.0

#clc; clear all; close all;
import numpy as np
import cv2
import imutils
from scipy import signal

# Parâmetros
Threshold = 1.25 * 10 ** -2
sigma = 1
k = 0.12

# read source JPEG image into 3D array
img = cv2.imread('test.png')
# gray scale value
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

limiar = grayscale.copy()

#binary[binary > 0] = 255

kernel = np.ones((3, 3), np.uint8)

erosion = cv2.erode(limiar, kernel, iterations = 1)

Ic = cv2.bitwise_not(limiar - erosion)

#dilation = cv2.dilate(Ic, kernel, iterations = 1)

cv2.imshow("I - Phi(I)", Ic)
cv2.waitKey(0)

print('open image')

# Shifts
Ix, Iy = np.gradient(Ic)

print('gradient')

Ix2 = np.square(Ix)
Iy2 = np.square(Iy)
Ixy = np.multiply(Ix, Iy)

print('multiply square of gradient')

windowsize = 3 * sigma
Wx, Wy = np.meshgrid(list(range(-windowsize, windowsize)), list(range(-windowsize, windowsize)))
w = np.exp(-(np.square(Wx) + np.square(Wy)) / (2 * sigma ** 2))

print('meshgrid')

# Convolutions
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
A = signal.convolve2d(w, Ix2)
B = signal.convolve2d(w, Iy2)
C = signal.convolve2d(w, Ixy)

print('convolve 2D meshgrid')

# x,y is the width and length of the image
(x, y) = grayscale.shape
# Initialize Corner response
R = np.zeros(grayscale.shape, dtype = "uint64")

# Loop for each pixel
for xRows in range(0, x - 1):
    for yColumns in range(0, y - 1):
        # get M at each pixel
        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.matrix.html
        M = np.matrix([[A[xRows, yColumns], C[xRows, yColumns]], [C[xRows, yColumns], B[xRows, yColumns]]])
        # compute R at each pixel
        # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.det.html
        # a.trace(offset=0)	Sum along diagonal elements
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.trace.html
        R[xRows, yColumns] = np.linalg.det(M) - k * (np.trace(M) ** 2)

print('Matrix M')

img_result = img.copy()
# Corner response map
img_map = np.zeros(grayscale.shape)

#for xRows = 1:x
for xRows in range(0, x - 1):
    #for yColumns = 1:y
    for yColumns in range(0, y - 1):
        # For those corner response R larger than Threshold
        if R[xRows, yColumns] > Threshold:
        	# Mark corner point by blue
            img_result[xRows, yColumns] = (0, 0, 255)
        	# Mark corner point by white
            img_map[xRows, yColumns] = 255

# results
cv2.imshow('Corner Detected', img_result)
cv2.waitKey(0)

cv2.imshow('Corner Response Map', img_map)
cv2.waitKey(0)

cv2.destroyAllWindows()
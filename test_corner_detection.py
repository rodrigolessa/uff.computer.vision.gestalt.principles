# Cast Matlab R2013a to Python 3.7.0

#clc; clear all; close all;
import numpy as np
import cv2
import imutils
from scipy import signal

# Settings
Threshold = 1/2000
# Testar com = 1.25 * 10 ** -2
sigma = 1
k = 0.12

#IMG = imread('stars.jpg');                  % read source JPEG image into 3D array
#IMG = cv2.imread('referencias\\stars.jpg')
IMG = cv2.imread('test.png')
#IMG_hsv = rgb2hsv(IMG);                     % convert image from RGB to HSV

#IMG_hsv_value = IMG_hsv(:,:,3);             % gray scale value
IMG_hsv_value = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)

print('open image')

#[Ix, Iy] = gradient(IMG_hsv_value);         % Shifts
Ix, Iy = np.gradient(IMG_hsv_value)

print('gradient')

#Ix2 = Ix.^2;
Ix2 = np.square(Ix)
#Iy2 = Iy.^2;
Iy2 = np.square(Iy)
#Ixy = Ix.*Iy;
Ixy = np.multiply(Ix, Iy)

print('multiply square of gradient')

# Window Function
windowsize = 3*sigma
#[Wx, Wy] = meshgrid(-windowsize : windowsize, -windowsize : windowsize);
Wx, Wy = np.meshgrid(list(range(-windowsize, windowsize)), list(range(-windowsize, windowsize)))
#w = exp(-(Wx .^2 + Wy .^2) / (2 * sigma ^ 2))
w = np.exp(-(np.square(Wx) + np.square(Wy)) / (2 * sigma ** 2))

print('meshgrid')

# Convolutions
#A = conv2(w, Ix2);
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
A = signal.convolve2d(w, Ix2)
#B = conv2(w, Iy2);
B = signal.convolve2d(w, Iy2)
#C = conv2(w, Ixy);
C = signal.convolve2d(w, Ixy)

print('convolve 2D meshgrid')

#[x,y] = size(IMG_hsv_value);            % x,y is the width and length of the image
(x, y) = IMG_hsv_value.shape
#R = zeros(x, y);                        % Initialize Corner response R
R = np.zeros(IMG_hsv_value.shape, dtype = "uint64")

# Loop for each pixel
#for xRows = 1:x
for xRows in range(0, x - 1):
    #for yColumns = 1:y  
    for yColumns in range(0, y - 1):
        #M = [A(xRows, yColumns), C(xRows, yColumns); C(xRows, yColumns), B(xRows, yColumns)];   % get M at each pixel
        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.matrix.html
        M = np.matrix([[A[xRows, yColumns], C[xRows, yColumns]], [C[xRows, yColumns], B[xRows, yColumns]]])
        #R(xRows,yColumns) = det(M) - k * (trace(M) ^ 2);                                        % compute R at each pixel
        # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.det.html
        # a.trace(offset=0)	Sum along diagonal elements
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.trace.html
        R[xRows, yColumns] = np.linalg.det(M) - k * (np.trace(M) ** 2)
    #end
#end

print('Matrix M')

#IMG_result = IMG;
IMG_result = IMG.copy()
#IMG_map = zeros(x, y, 3);                                   % Corner response map
IMG_map = np.zeros(IMG_hsv_value.shape, dtype = "uint64")

#for xRows = 1:x
for xRows in range(0, x - 1):
    #for yColumns = 1:y
    for yColumns in range(0, y - 1):
        #if ((R(xRows, yColumns) > Threshold))               % For those corner response R larger than Threshold
        if R[xRows, yColumns] > Threshold:
        	#IMG_result(xRows, yColumns, :) = [0, 0, 255];   % Mark corner point by blue
            IMG_result[xRows, yColumns] = (0, 0, 255)
        	#IMG_map(xRows, yColumns, :) = 255;              % Mark corner point by white
            IMG_map[xRows, yColumns] = 255
        #end
    #end
#end

# Show results
#figure('Name', 'Corner Detected');  
#imshow(IMG_result);
cv2.imshow('Corner Detected', IMG_result)
cv2.waitKey(0)

#figure('Name', 'Corner Response Map');
#imshow(IMG_map);
cv2.imshow('Corner Response Map', IMG_map)
cv2.waitKey(0)

cv2.destroyAllWindows()
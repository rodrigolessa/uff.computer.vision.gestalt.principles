import numpy as np
import cv2
import imutils
from scipy import signal

threshold = 1/2000
sigma = 1
k = 0.12

original = cv2.imread('referencias\\boundaries.png')
grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
grayscale[grayscale > 0] = 255

#cv2.imshow('grayscale', grayscale)
#cv2.waitKey(0)

# find where the black pixels are
points = np.argwhere(grayscale == 0)
print('argwhere:')
print(points[:5])

# store them in x,y coordinates instead of row, col indices
#points = np.fliplr(points)
#print('fliplr:')
#print(points)

a = cv2.arcLength(points, False)
print('arcLength')
print(a)

# Python: cv2.approxPolyDP(curve, epsilon, closed[, approxCurve]) → approxCurve

for x, y in points[:100]:
    original[x, y] = (0, 0, 255)

cv2.imshow('arcLength', original)
cv2.waitKey(0)


# Mahotas
# def circle_se(radius):
#     '''
#     circle = circle_se(radius)
#     Build a circular structuring element of a given radius
#     Parameters
#     ----------
#     radius : int
#         Radius of circle
#     Returns
#     -------
#     circle : boolean ndarray
#     '''
#     if not (radius > 0):
#         raise ValueError('mahotas.morph.circle: radius must be positive')
#     X = np.arange(-radius, +radius+1)
#     X,Y = np.meshgrid(X,X)
#     return (X**2 + Y**2) < radius**2
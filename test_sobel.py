import numpy as np
import cv2
import imutils
from logo_pre_processing import LogoPreProcessing

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html

def findGradients(img):
    #cv2.CV_32F
    #cv2.CV_64F
    I_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
    I_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5)

    return I_x, I_y

src = cv2.imread('test.png', 0)
blockSize = 3

print("Gradient Matrix")

size_y, size_x = src.shape

# Return object
scored_image_gradient = np.zeros(src.shape)

# http://stackoverflow.com/q/11331830/
gradient_x, gradient_y = findGradients(src)

gradient_xx = np.square(gradient_x)
gradient_yy = np.square(gradient_y)
gradient_xy = np.multiply(gradient_x, gradient_y)

for y in np.arange(size_y):
    for x in np.arange(size_x):
        M = np.zeros((2, 2))

        # Use a blockSize x blockSize window (except for pixels too close to the edges)
        ymin = max(0, y - blockSize / 2)
        ymax = min(size_y, y + blockSize / 2)
        xmin = max(0, x - blockSize / 2)
        xmax = min(size_x, x + blockSize / 2)

        for v in np.arange(ymin, ymax):
            for u in np.arange(xmin, xmax):
                M[0, 0] += gradient_xx[v, u]
                M[0, 1] += gradient_xy[v, u]
                M[1, 1] += gradient_yy[v, u]
            M[1, 0] = M[0, 1]

print(M)
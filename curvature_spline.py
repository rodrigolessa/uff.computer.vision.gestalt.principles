from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils
import os

debugging = False

clear = lambda: os.system('cls')
clear()


class CurvatureDefinition:

    def __init__(self):
        """
        Inicio
        """
        self.path = ''

    # Um spline é uma curva definida matematicamente por dois ou mais pontos de controle. 
    # Os pontos de controle que ficam na curva são chamados de nós.
    def curvature_spline(self, x, y, error = 0.1):
        """
        Calculate the signed curvature of a 2D curve at each point
        using interpolating splines.

        Parameters
        ----------
        x, y  : numpy.array(dtype = float) shape (n_points, )
        error : float - The admisible error when interpolating the splines

        Returns
        -------
        curvature: numpy.array shape (n_points, )

        Note: This is 2-3x slower (1.8 ms for 2000 points) than `curvature_gradient`
        but more accurate, especially at the borders.
        """

        # handle list of complex case
        #if y is None:
        #x, y = x.real, x.imag

        # create range for array length
        t = np.arange(x.shape[0])
        std = error * np.ones_like(x)

        fx = UnivariateSpline(t, x, k=4, w=1 / np.sqrt(std))
        fy = UnivariateSpline(t, y, k=4, w=1 / np.sqrt(std))

        xl = fx.derivative(1)(t)
        xl2 = fx.derivative(2)(t)
        yl = fy.derivative(1)(t)
        yl2 = fy.derivative(2)(t)

        curvature = (xl * yl2 - yl * xl2) / np.power(xl ** 2 + yl ** 2, 3/2)

        return curvature



##########################################################

cd = CurvatureDefinition()

##########################################################

# Load
img = cv2.imread('test.png', 0)

# Binarization
_, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Boundaries using Mathematical Morphological Operators
kernel = np.ones((3, 3), np.uint8)

# Erosion: Shrinking the foreground
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/erode.htm
#erosion = cv2.erode(threshold, kernel, iterations = 1)

# Dilation: Expanding the foreground
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm
dilation = cv2.dilate(threshold, kernel, iterations = 1)

# Boundaries extraction
edges = dilation - threshold

#edges = cv2.dilate(edges, kernel, iterations = 1)

if debugging == True:
    cv2.imshow('edges', edges)
    cv2.waitKey(0)

print('binary image matrix')
print(edges)

w, h = edges.shape

print('binary image matrix size')
print(w)
print(h)

px = edges[13,93] # line, column
print('Test específic coordinate')
print(px)

print(zip(*np.nonzero(edges)))
#[(0, 0), (0, 1), (1, 0), (2, 3), (3, 0), (3, 1), (3, 3)]

#threshold[threshold > 0] = 1

rows, cols = np.nonzero(edges)

print('Lines and columns')
print(rows)
print(cols)

#for r, c in zip(rows, cols):
#    if threshold[r][c] != 255:
#        print(threshold[r][c])

t_ = np.arange(rows.shape[0])

print('create range for array length')
print(rows.shape[0])
print(t_)

print('np.ones_like(rows)')
print(np.ones_like(rows))

stdErro = 0.1 * np.ones_like(rows)

print('std erro')
print(stdErro)

w_ = 1 / np.sqrt(stdErro)

print('spline weight')
print(w_)

fx_ = UnivariateSpline(t_, rows, k = 3, w = w_)

#print('Spline of x')
#print(fx_)

x, y = np.array([1, 2, 3, 4]), np.array([1, np.nan, 3, 4])
w = np.isnan(y)
y[w] = 0.
spl = UnivariateSpline(x, y, w=~w)

#print('Spline of 1, 2, 3')
#print(spl)

#x = np.linspace(-3, 3, 50)

#print('linspace of -3, 3')
#print(x)

#y = np.exp(-x**2) + 0.1 * np.random.randn(50)


points = np.argwhere(edges == 255)
print('argwhere 255')
print(points)

# store them in x,y coordinates instead of row, col indices
points = np.fliplr(points)
print('fliplr:')
print(points)

plt.plot(cols, rows, 'ro', ms = 5)
#Use the default value for the smoothing parameter:
spl = UnivariateSpline(cols, rows)
#xs = np.linspace(-3, 3, 1000)
#plt.plot(xs, spl(xs), 'g', lw=3)
#Manually change the amount of smoothing:
spl.set_smoothing_factor(0.5)
#plt.plot(xs, spl(xs), 'b', lw=3)
plt.show()

# cs = cd.curvature_spline(rows, cols)

# print('splines')
# print(cs)

#I_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
#I_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5)

#cs = cd.curvature_spline(I_x, I_y)

#print(cs)

x_space = np.linspace(0, 10, w)

print('linspace')
print(x_space)

y = np.sin(x)
spl = UnivariateSpline(x, y, k=4, s=0)

#print(spl.derivative().roots() / np.pi)

x_ = spl.derivative(1)(x)
x__ = spl.derivative(2)(x)
y_ = 1 #fy.derivative(1)(t)
y__ = 1 #fy.derivative(2)(t)

curv = (x_ * y__ - y_ * x__) / np.power(x_ ** 2 + y_ ** 2, 3/2)

#print(curv)


import matplotlib.pyplot as plt
N = 8
y = np.zeros(N)
x1 = np.linspace(0, 10, N, endpoint=True)
x2 = np.linspace(0, 10, N, endpoint=False)
plt.plot(x1, y, 'o')
plt.plot(x2, y + 0.5, 'o')
plt.ylim([-0.5, 1])
#plt.show()
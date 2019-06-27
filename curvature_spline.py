import numpy as np
import cv2
import imutils
from scipy.interpolate import UnivariateSpline

class CurvatureDefinition:

    def __init__(self):
        """
        Inicio
        """
        self.path = ''

    # Um spline é uma curva definida matematicamente por dois ou mais pontos de controle. 
    # Os pontos de controle que ficam na curva são chamados de nós.
    def curvature_spline(self, x, y = None, error = 0.1):
        """
        Calculate the signed curvature of a 2D curve at each point
        using interpolating splines.

        Parameters
        ----------
        x, y: numpy.array(dtype = float) shape (n_points, )
        or
        y = None and
        x is a numpy.array(dtype=complex) shape (n_points, )
        In the second case the curve is represented as a np.array
        of complex numbers.
        error : float
        The admisible error when interpolating the splines

        Returns
        -------
        curvature: numpy.array shape (n_points, )

        Note: This is 2-3x slower (1.8 ms for 2000 points) than `curvature_gradient`
        but more accurate, especially at the borders.
        """

        # handle list of complex case
        #if y is None:
        #x, y = x.real, x.imag

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

cd = CurvatureDefinition()

img = cv2.imread('test.png', 0)

I_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
I_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5)

cs = cd.curvature_spline(I_x, I_y)

print(cs)
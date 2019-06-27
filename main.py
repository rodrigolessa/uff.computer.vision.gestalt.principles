# Reprodução do Artigo:
## Decomposition and Construction of Object Based on Law of Closure in Gestalt Psychology
## Yi-Chong Zeng
## Data Analytics Technology and Applications Research Institute
## Institute for Information Industry
## Taipei, Taiwan
## yichongzeng@iii.org.tw

# REFs:
# https://www.codingame.com/playgrounds/38470/how-to-detect-circles-in-images
# Find the curvature of the curve
# Cited article
# Automatic Detections of Nipple and Pectoralis Major in Mammographic Images


import numpy as np
import cv2
import imutils
from scipy.interpolate import UnivariateSpline
from logo_pre_processing import LogoPreProcessing

# GOAL:
# A method to implement object reification based on 
# law of closure in Gestalt psychology.

# Construct object via combination of discontinuous edges, 
# detects key vertices and then divides boundary into small edges. 
# Virtual edge is generated to connect two edges as 
# straight line, ellipse, or circle. Phases of the project:

######################################################

# Um spline é uma curva definida matematicamente por dois ou mais pontos de controle. 
# Os pontos de controle que ficam na curva são chamados de nós.
def _curvature_spline(x, y = None, error = 0.1):
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

######################################################
# Initial Parameters
imageName = 'test.png' #'gestalt_triangle.png' #gestalt-triangle-630x659.jpg'
imageSize = 180
imageBorderSize = 5

# Instantiate preprocessing functions
lpp = LogoPreProcessing(imageName, imageSize, imageBorderSize, False)

# Load the image
normalized = lpp.normalize

# Crop and get only important shape from image
# Centralizar o logo na imagem deixando somente 
# a área da imagem que possui forma
normalized = lpp.crop(normalized)

# Resize and add border
normalized = lpp.scale(normalized)

# Flip values
# Nossa imagem de teste possui fundo preto, então invertemos para ficar igual ao artigo:
# Flip the values of the pixels 
# (black pixels are turned to white, and white pixels to black)
flipped = cv2.bitwise_not(normalized)

# Binarization
binaryImg = flipped.copy()

# Esse pre-processamento termina transformando a imagem em binária, 
# pois o artigo só considera isso como entrada para o processo abordado.
# Any pixel with a value greater than zero (black) is set to 255 (white)
binaryImg[binaryImg > 0] = 255

# Debugging: Show the original image
debug1 = np.hstack((normalized, flipped, binaryImg))

cv2.imshow("Preprocessing: Resize + Bitwise + Binarization", debug1)
cv2.waitKey(0)

# Accessing Image Properties
# Image properties include number of rows, columns and channels, 
# type of image data, number of pixels etc.
# Returns a tuple of number of rows, columns and channels (if is color)
outlineBase = np.zeros(normalized.shape, dtype = "uint8")


######################################################
# A - Edge and Boundary Detection

# In this work, binary images are analyzed for object
# detection. The boundaries and edges using morphology
# operations which is defined as Ic = I - Phi(I). The denotations I
# and Ic represent the original binary image and its boundary,
# respectively. The function Phi(I) generate the result by applying
# erosion operation or dilation operation to I.

# Ic = I - Phi(I)
# I = original binary image
# Ic = image boundary

# Boundaries using Mathematical Morphological Operators

kernel = np.ones((3, 3), np.uint8)

# ? - Optimal edge detection:
#     * Canny edge detection

# 1 - Erosion: Shrinking the foreground
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/erode.htm
# Redução das bordas do objeto. Consegue eliminar objetos 
# muito pequenos mantendo somente pixels de elementos estruturantes.
erosion = cv2.erode(binaryImg, kernel, iterations = 1)

# 2 - Dilation: Expanding the foreground
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm
# Expandir as bordas do objeto, podendo preencher pixels faltantes.
# Completar a imagem com um objeto estruturante.
dilation = cv2.dilate(binaryImg, kernel, iterations = 1)

# Teste: gradient = cv2.bitwise_not(cv2.morphologyEx(binaryImg, cv2.MORPH_GRADIENT, kernel))

# The function Phi(I) generate the result by applying
# erosion operation or dilation operation to I.
#Ic = cv2.subtract(binaryImg, dilation) # for negative values
Ic = cv2.bitwise_not(binaryImg - erosion)

# TODO: Verificar se a imagem do resultado da erosão possui limiar
# senão aplicar dilatação

# Write a image for documentation
#cv2.imwrite("boundaries.png", Ic)

# Debugging: Show the original binary image and its boundary
debug2 = np.hstack((erosion, dilation, Ic))

#(cnts, hierarchy) = cv2.findContours(Ic.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#for c in cnts[:2]:
#    print(c)
#    cv2.drawContours(outlineBase, [c], -1, (255, 255, 255), 2)
#cv2.imshow("outlineBase", outlineBase)

cv2.imshow("Morphological Transformations: erode + dilation + I - Phi(I)", debug2)
cv2.waitKey(0)


######################################################
# B - Key Vertex Detection

# Let Psi be coordinate set of N vertices on edge of I, and Psi =
# {(x[n], y[n]) | n={1, 2, ..., N}}. The definition of curvature [8]
# is formulated below, 

# k = ((x' * y'') - (x'' * y')) / (((x' ** 2) + (y' ** 2)) ** (3/2))

# [8] Wikipedia: Curvature, [online]
# https://en.wikipedia.org/wiki/Curvature
# k = - (-2*Lx*Lxy*Ly + Lxx*Ly^2 + Lyy*Lx^2 ) / ((Lx^2 + Ly^2)^(3/2))

# where k is curvature. The denotations x' and x'' are, respectively, 
# first-order differential of x and 
# second-order differential of x. 
# The definition is the same as the variable y. The determination of 
# key point detection is defined as, 

# D(x, y)   = 1, if |K x,y| > Tau1
#           = 0, if else

# where Tau1 is a threshold to determine key vertex, 
# and Tau1 is set to 1.25 x 10 ** -2 in the experiment

# Detection of Edges Using Mathematical Morphological Operators

# 1 - First order derivative / gradient methods are as follows:
#     * Roberts operator
#     * TODO: Sobel operator (write a code)
#     * Prewitt operator

# 2 - Second order derivative:
#     * Laplacian
#     * Laplacian of Gaussian
#     * Difference of Gaussian

# Finding x and y derivatives
#dx, dy = np.gradient(Ic)
# find where the black pixels are
gray = np.float32(normalized)
#points = np.argwhere(Ic == 0)

#for x, y in points[:50]:

#x = np.array([0, 1, 2, 3])
#y = np.array([0.1, 0.2, 0.9, 2.1])
x=np.arange(0,4)
y=np.array([0,1,1.9,3.1])
print(np.round(np.polyfit(x,y,1)))

_curvature_spline(x, y)

######################################################
# C - Virtual Edge Generation

# In order to determine relationship between two edges, the
# proposed scheme utilizes approximate line and approximate
# ellipse/circle to edges. Let x[n] and y[n] be, respectively,
# coordinates of x-direction and y-direction, and 0 >= x[n], y[n] <= 1.
# Approximate line is formulated as c1x[n]+c2y[n]=1, and two
# parameters c1 and c2 are estimated by

# where N is total number of coordinate. Average error (ε)
# between original coordinate and transferred coordinate (namely,
# xt[n] and yt[n]) is calculated according to

# where xt[n]=(1-c2y[n])/c1 and yt[n]=(1-c1x[n])/c2. As eps < τ2, a
# virtual edge connects two edges as a line

# Approximate ellipse/circle is formulated as
# d1x2[n]+d2y2[n]+d3x[n]+d4y[n]=1, and four parameters d1, d2, d3
# and d4 are estimated by,

# As -4d1d2<0, approximate ellipse/circle is available.
# Similarly, average error between original coordinate and
# transferred coordinate is computed by (4), and transferred
# coordinate is defined as follows

# As ε<τ3, a virtual edge connects two edges as an ellipse/a
# circle. Moreover, this scheme is capable of examining a single
# edge whether is a part of ellipse/circle or not. 

######################################################
# D - Object Constrution

# Two works are implemented during object detection. 
# The first work is to detect ellipse-like/circle-like object. 
# An edge, which is determined as a part of ellipse/circle, 
# is collected in advanced. 
# Moreover, a group of edges connected via virtual edges 
# is collected as well. 
# Drawing approximate ellipses based on the collected edges, 
# ellipse-like/circle-like objects are detected. 
# The second work is to detect polygon with closured contours. 
# Exclude edges utilized to detect ellipse-like/circlelike object, 
# our scheme searches closed loop among the rest of edges 
# and the virtual edges.

# 1. First work
# detect ellipse-like/circle-like object

# 2. Second work
# detect polygon with closured contours


#cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

cv2.destroyAllWindows()
# Reprodução do Artigo:
## Decomposition and Construction of Object Based on Law of Closure in Gestalt Psychology
## Yi-Chong Zeng
## Data Analytics Technology and Applications Research Institute
## Institute for Information Industry
## Taipei, Taiwan
## yichongzeng@iii.org.tw

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
# Initial Parameters
imageName = 'test.png'
imageSize = 200
imageBorderSize = 5

# Instantiate preprocessing functions
lpp = LogoPreProcessing(imageName, imageSize, imageBorderSize, False)

# Load the image
normalized = lpp.normalize()

# Convert it to grayscale
# ! O próprio artigo não considera as cores ou outras camadas da imagem.
grayscale = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)

# Flip values
# ! Nossa imagem de teste possui fundo preto, então invertemos para ficar igual ao artigo:
# Flip the values of the pixels 
# (black pixels are turned to white, and white pixels to black)
grayscale = cv2.bitwise_not(grayscale)

# Binarization
binary = grayscale.copy()
# Esse pre-processamento termina transformando a imagem em binária, 
# pois o artigo só considera isso como entrada para o processo abordado.
# Any pixel with a value greater than zero (black) is set to 255 (white)
binary[binary > 0] = 255
# Ou obter o thresholder com OpenCV + Otsu

# Debugging: Show the original image
debug1 = np.hstack((grayscale, binary))

cv2.imshow("Preprocessing: Bitwise + Binarization", debug1)
#cv2.waitKey(0)

# Accessing Image Properties
# Image properties include number of rows, columns and channels, 
# type of image data, number of pixels etc.
# Returns a tuple of number of rows, columns and channels (if is color)
outlineBase = np.zeros(normalized.shape, dtype = "uint8")

print('')
print(' - shape size:' + str(outlineBase.shape))

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
erosion = cv2.erode(binary, kernel, iterations = 1)

# 2 - Dilation: Expanding the foreground
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm
# Expandir as bordas do objeto, podendo preencher pixels faltantes.
# Completar a imagem com um objeto estruturante.
dilation = cv2.dilate(binary, kernel, iterations = 1)

# Teste: gradient = cv2.bitwise_not(cv2.morphologyEx(binaryImg, cv2.MORPH_GRADIENT, kernel))

# The function Phi(I) generate the result by applying
# erosion operation or dilation operation to I.
#Ic = cv2.bitwise_not(binary - erosion)
Ic = cv2.subtract(binary, erosion) # for negative values

# TODO: Verificar se a imagem do resultado da erosão possui limiar
# senão aplicar dilatação

# Write a image for documentation
cv2.imwrite("resultado_erosao.png", Ic)

# Debugging: Show the original binary image and its boundary
debug2 = np.hstack((erosion, dilation, Ic))

cv2.imshow("Morphological Transformations: erode + dilation + I - Phi(I)", debug2)
#cv2.waitKey(0)

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

cv2.waitKey(0)
cv2.destroyAllWindows()
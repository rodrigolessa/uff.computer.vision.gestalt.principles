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
imageName = 'gestalt-triangle-630x659.jpg'
imageSize = 300
imageBorderSize = 50

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

# Detection of Edges Using Mathematical Morphological Operators

kernel = np.ones((3, 3), np.uint8)

# 1 - First order derivative / gradient methods are as follows:
#     * Roberts operator
#     * TODO: Sobel operator (write a code)
#     * Prewitt operator

# 2 - Second order derivative:
#     * Laplacian
#     * Laplacian of Gaussian
#     * Difference of Gaussian

# 3 - Optimal edge detection:
#     * Canny edge detection

# 4 - Erosion: Shrinking the foreground
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/erode.htm
# Redução das bordas do objeto. Consegue eliminar objetos 
# muito pequenos mantendo somente pixels de elementos estruturantes.
erosion = cv2.erode(binaryImg, kernel, iterations = 1)

# 5 - Dilation: Expanding the foreground
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm
# Expandir as bordas do objeto, podendo preencher pixels faltantes.
# Completar a imagem com um objeto estruturante.
dilation = cv2.dilate(binaryImg, kernel, iterations = 1)

# Teste = gradient = cv2.bitwise_not(cv2.morphologyEx(binaryImg, cv2.MORPH_GRADIENT, kernel))

# The function Phi(I) generate the result by applying
# erosion operation or dilation operation to I.
#Ic = cv2.subtract(binaryImg, dilation) # for negative values
Ic = cv2.bitwise_not(binaryImg - erosion)

# TODO: Verificar se a imagem do resultado da erosão possui limiar
# senão aplicar dilatação

# Debugging: Show the original binary image and its boundary
debug2 = np.hstack((erosion, dilation, Ic))

cv2.imshow("Morphological Transformations: erode + dilation + I - Phi(I)", debug2)
cv2.waitKey(0)

######################################################
# B - Key Vertex Detection

# Let Psi be coordinate set of N vertices on edge of I, and Psi =
# {(x[n], y[n]) | n={1, 2, ..., N}}. The definition of curvature [8]
# is formulated below, where k is curvature. The denotations x' and x'' 
# are, respectively, first-order differential of x and 
# second-order differential of x. The definition is the same as 
# the variable y. The determination of key point detection is defined 
# as, where Tau1 is a threshold to determine key vertex, 
# and Tau1 is set to 1.25 x 10 ** -2 in the experiment

# k = ((x' * y'') - (x'' * y')) / (((x' ** 2) + (y' ** 2)) ** (3/2))

# D(x, y)   = 1, if |K x,y| > Tau1
#           = 0, if else

######################################################
# C - Virtual Edge Generation

######################################################
# D - Object Constrution


cv2.destroyAllWindows()
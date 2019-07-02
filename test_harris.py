import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test.png')

grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
grayscale = np.float32(grayscale)

cv2.imshow("grayscale", grayscale)

# img       - Input image, it should be grayscale and float32 type.
# blockSize - It is the size of neighbourhood considered for corner detection
# ksize     - Aperture parameter of Sobel derivative used.
# k         - Harris detector free parameter in the equation.
dst = cv2.cornerHarris(grayscale, 3, 3, 0.00001)

cv2.imshow("dist", dst)

# Área de busca
kernel = np.ones((3, 3))

# Dilation: Expanding the foreground
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm
# Expandir as bordas do objeto, podendo preencher pixels faltantes.
# Completar a imagem com um objeto estruturante.
# dst = cv2.dilate(dst, kernel, iterations = 1)

threshold = 1.25 * 10 ** -2

edges = np.zeros(img.shape)

edges[dst > threshold * dst.max()] = [255,255,255]

cv2.imshow("edges", edges)

# Erosion: Shrinking the foreground
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/erode.htm
# Redução das bordas do objeto. Consegue eliminar objetos 
# muito pequenos mantendo somente pixels de elementos estruturantes.
# edges = cv2.erode(edges, kernel, iterations = 1)

#cv2.imwrite('erode.png', cornerErosion)

# # from matplotlib import pyplot as plt
# plt.subplot(121)
# plt.imshow(img, cmap = 'gray')
# plt.title('Edge Image')
# plt.xticks([])
# plt.yticks([])
# plt.subplot(122)
# plt.imshow(dst, cmap = 'gray')
# plt.title('Corner Image')
# plt.xticks([])
# plt.yticks([])
# plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
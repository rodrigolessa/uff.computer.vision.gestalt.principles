import cv2
import numpy as np
img = cv2.imread('test.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray", gray)
#cv2.waitKey(0)

imc = np.zeros(img.shape)

gray = np.float32(gray)

#gray[gray > 0] = 255

cv2.imshow("binary", gray)
cv2.waitKey(0)

kernel = np.ones((3, 3))

#erosion = cv2.erode(gray, kernel, iterations = 1)
#cv2.imshow("erode 1", erosion)
#cv2.waitKey(0)

#Ic = gray - erosion
#cv2.imshow("ic", Ic)
#cv2.waitKey(0)

#dilation = cv2.dilate(Ic, kernel, iterations = 1)
#cv2.imshow("dilation", dilation)
#cv2.waitKey(0)

# img - Input image, it should be grayscale and float32 type.
# blockSize - It is the size of neighbourhood considered for corner detection
# ksize - Aperture parameter of Sobel derivative used.
# k - Harris detector free parameter in the equation.
dst = cv2.cornerHarris(gray, 3, 3, 0.00001)
#cv2.imshow("cornerHarris", dst)
#cv2.waitKey(0)

#dst = cv2.dilate(dst,None)
#cv2.imshow("dilate", dst)
#cv2.waitKey(0)

threshold = 1.25 * 10 ** -2

#img[dst > 0.01 * dst.max()]=[0,0,255]
img[dst > threshold * dst.max()]=[0,0,255]

imc[dst > threshold * dst.max()]=[255,255,255]
#cv2.imshow("corner", imc)
#cv2.waitKey(0)

cv2.imwrite('harris.png', imc)

cornerErosion = cv2.erode(imc, kernel, iterations = 1)

cv2.imshow("corner erosion", cornerErosion)
cv2.waitKey(0)

cv2.imwrite('erode.png', cornerErosion)

cv2.imshow("dist", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# from matplotlib import pyplot as plt
# plt.subplot(121)
# plt.imshow(img,cmap = 'gray')
# plt.title('Edge Image')
# plt.xticks([])
# plt.yticks([])
# plt.subplot(122)
# plt.imshow(dst,cmap = 'gray')
# plt.title('Corner Image')
# plt.xticks([])
# plt.yticks([])
# plt.show() 
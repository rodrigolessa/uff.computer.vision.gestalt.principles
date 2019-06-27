import numpy as np
import cv2
import imutils
from scipy.interpolate import UnivariateSpline
from logo_pre_processing import LogoPreProcessing

def stripX(pts):
    result = []
    for pt in pts:
        #print(pt)
        print(pt[:1])
        #result.append(pt['x'])

    return result

imgPath = 'test.png'
imgSize = 180
imgBorderSize = 5

original = cv2.imread(imgPath)
grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
blur = cv2.bilateralFilter(grayscale, 9, 75, 75)
_, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#THRESH_BINARY = fundo preto or THRESH_BINARY_INV = fundo branco
outline = np.zeros(threshold.shape, dtype = "uint8")
# Shape of image is accessed by img.shape. It returns a tuple of number of rows, 
# columns and channels (if image is color):

cv2.imshow("threshold", threshold)

# cv2.RETR_EXTERNAL - telling OpenCV to find only the outermost contours.
# cv2.CHAIN_APPROX_SIMPLE - to compress and approximate the contours to save memory
#img2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
(contours, hierarchy) = cv2.findContours(threshold.copy(),
    cv2.RETR_EXTERNAL, 
    cv2.CHAIN_APPROX_SIMPLE)

# The outline is drawn as a filled in mask with white pixels:
for cnt in contours:
    if(cv2.contourArea(cnt) > 0):
        sxr = stripX(cnt)
        #print(sxr)
        # cv2.arcLength and cv2.approxPolyDP. 
        # These methods are used to approximate the polygonal curves of a contour.
        #peri = cv2.arcLength(cnt, True)

        # Level of approximation precision. 
        # In this case, we use 2% of the perimeter of the contour.
        # * The Ramer–Douglas–Peucker algorithm, also known as the Douglas–Peucker algorithm and iterative end-point fit algorithm, 
        # is an algorithm that decimates a curve composed of line segments to a similar curve with fewer point
        #approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        cv2.drawContours(outline, [cnt], -1, 255, -1)

cv2.waitKey(0)
cv2.destroyAllWindows()
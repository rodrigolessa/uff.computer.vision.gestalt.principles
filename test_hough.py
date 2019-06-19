import numpy as np
import cv2

# APLICAR TRANFROMADA DE HOUGH

## The edges are detected in the image using Laplace of Gaussian with Zero Crossing. 
# This provides the basic outline in the image. 
# At each point on the edge, voting for all possible circles 
# is performed in the Hough space. The local maxima in the Hough space 
# gives the circles. A threshold is used to identify qualifying local maxima's.

img = cv2.imread('gestalt-triangle-630x659.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

edges = cv2.bitwise_not(edges)

#cv2.imshow('canny', edges)
#cv2.waitKey(0)

output = img.copy()

# Hough Circle Transform

circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,20,
    param1=50,param2=30,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:1]:
    # draw the outer circle
    cv2.circle(output,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(output,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',output)
cv2.waitKey(0)

# Houge Transforme

imgLines = img.copy()

lines = cv2.HoughLines(edges,1,np.pi/180, 200)
for l in lines[:10]:
    for rho, theta in l:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(imgLines,(x1,y1),(x2,y2),(0,0,255),1)

cv2.imshow('hough lines', imgLines)
cv2.waitKey(0)

# Probabilistic Hough Transforme

imgPLines = img.copy()

minLineLength = 50
maxLineGap = 100
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
for l in lines[:50]:
    for x1, y1, x2, y2 in l:
        cv2.line(imgPLines, (x1, y1), (x2, y2), (0, 255, 0), 1)

cv2.imshow('Probabilistic hough lines', imgPLines)
cv2.waitKey(0)

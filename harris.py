import cv2

im = cv2.imread("test.png", 0)

dst_32f = cv2.CreateImage(im.shape, cv2.IPL_DEPTH_32F, 1)

neighbourhood = 3
aperture = 3
k = 0.01
maxStrength = 0.0
threshold = 0.01
nonMaxSize = 3

cv2.CornerHarris(im, dst_32f, neighbourhood, aperture, k)

minv, maxv, minl, maxl = cv.MinMaxLoc(dst_32f)

dilated = cv2.CloneImage(dst_32f)
cv2.Dilate(dst_32f, dilated) # By this way we are sure that pixel with local max value will not be changed, and all the others will

localMax = cv2.CreateMat(dst_32f.height, dst_32f.width, cv.CV_8U)
cv2.Cmp(dst_32f, dilated, localMax, cv.CV_CMP_EQ) #compare allow to keep only non modified pixel which are local maximum values which are corners.

threshold = 0.01 * maxv
cv2.Threshold(dst_32f, dst_32f, threshold, 255, cv.CV_THRESH_BINARY)

cornerMap = cv.CreateMat(dst_32f.height, dst_32f.width, cv.CV_8U)
cv2.Convert(dst_32f, cornerMap) #Convert to make the and
cv2.And(cornerMap, localMax, cornerMap) #Delete all modified pixels

radius = 3
thickness = 2

l = []
for x in range(cornerMap.height): #Create the list of point take all pixel that are not 0 (so not black)
    for y in range(cornerMap.width):
        if cornerMap[x,y]:
            l.append((y,x))

for center in l:
    cv2.Circle(im, center, radius, (255,255,255), thickness)


cv2.ShowImage("Image", im)
cv2.ShowImage("CornerHarris Result", dst_32f)
cv2.ShowImage("Unique Points after Dilatation/CMP/And", cornerMap)

cv2.WaitKey(0)
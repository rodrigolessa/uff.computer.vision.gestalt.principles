import numpy as np
from numpy import array
import cv2
import mahotas as mh
import scipy.ndimage.morphology as mo

def skeletonize(img):
    h1 = np.array([[0, 0, 0],[0, 1, 0],[1, 1, 1]]) 
    m1 = np.array([[1, 1, 1],[0, 0, 0],[0, 0, 0]]) 
    h2 = np.array([[0, 0, 0],[1, 1, 0],[0, 1, 0]]) 
    m2 = np.array([[0, 1, 1],[0, 0, 1],[0, 0, 0]])    
    hit_list = [] 
    miss_list = []
    for k in range(4): 
        hit_list.append(np.rot90(h1, k))
        hit_list.append(np.rot90(h2, k))
        miss_list.append(np.rot90(m1, k))
        miss_list.append(np.rot90(m2, k))    
    img = img.copy()
    while True:
        last = img
        for hit, miss in zip(hit_list, miss_list): 
            hm = mo.binary_hit_or_miss(img, hit, miss) 
            img = np.logical_and(img, np.logical_not(hm)) 
        if np.all(img == last):  
            break
    return img

#input image
nomeimg = 'test.png'
img = cv2.imread(nomeimg)
gray = cv2.imread(nomeimg,0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(6,6)) #con 4,4 si vede tutta la stella e riconosce piccoli oggetti
graydilate = cv2.erode(gray, element) #imgbnbin

ret,thresh = cv2.threshold(graydilate,127,255,cv2.THRESH_BINARY) 
imgbnbin = thresh

cv2.imshow('binaria',imgbnbin)
cv2.waitKey()

#finding a unique contour
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

Areacontours = list()
calcarea = 0.0
unicocnt = 0.0
for i in range (0, len(contours)):
    area = cv2.contourArea(contours[i])
    if (area > 90 ):  #con 90 trova i segni e togli puntini
        if (calcarea<area):
            calcarea = area
            unicocnt = contours[i]

cnt = unicocnt            
print(len(cnt))
cv2.drawContours(thresh,contours,-1,(0,255,0),3)

#fill holes
des = imgbnbin
cv2.drawContours(des,[unicocnt],0,255,-1)

gray = cv2.bitwise_not(des)

cv2.imshow('gray tappabuchi grAY',gray)
cv2.waitKey()


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
res = cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel)

cv2.imshow('res tappabuchi',res)
cv2.waitKey()

rest = cv2.bitwise_not(res)

print(rest)


cv2.imshow('rest tappabuchi 2',rest)
cv2.waitKey()

skel = skeletonize(rest)
skel = skel.astype(np.uint8)*255
print(skel)
cv2.imshow('skel',skel)
cv2.waitKey(0)

imgbnbin = mh.binary(skel)

#FINDING junction and PRUNING
print("mohatas imgbnbin")
print(imgbnbin)

b2 = mh.thin(imgbnbin)
b3 = mh.thin(b2, mh.endpoints('homotopic'), 15) # prune small branches, may need tuning

outputimage = mh.overlay(imgbnbin, b3)
mh.imsave('outputs.png', outputimage)

# structuring elements to search for 3-connected pixels
seA1 = array([[False,  True, False],
       [False,  True, False],
       [ True, False,  True]], dtype=bool)

seB1 = array([[False, False, False],
       [ True, False,  True],
       [False,  True, False]], dtype=bool)

seA2 = array([[False,  True, False],
       [ True,  True,  True],
       [False, False, False]], dtype=bool)

seB2 = array([[ True, False,  True],
       [False, False, False],
       [False,  True, False]], dtype=bool)

# hit or miss templates from these SEs
hmt1 = mh.se2hmt(seA1, seB1)
hmt2 = mh.se2hmt(seA2, seB2)

# locate 3-connected regions
b4 = mh.union(mh.supcanon(b3, hmt1), mh.supcanon(b3, hmt2))

# dilate to merge nearby hits
b5 = mh.dilate(b4, m.sedisk(10))

# locate centroids
b6 = mh.blob(mh.label(b5), 'centroid')

outputimage = mh.overlay(imgbnbin, mh.dilate(b6, mh.sedisk(5)))
mh.imsave('output.png', outputimage)
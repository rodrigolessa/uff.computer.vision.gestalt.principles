from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
from canny import canny_edge_detector
from collections import defaultdict
import cv2
import numpy as np

# Load
img = cv2.imread('test.png', 0)
# Binarization
_, threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Boundaries using Mathematical Morphological Operators
kernel = np.ones((3, 3), np.uint8)
# Erosion: Shrinking the foreground
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/erode.htm
#erosion = cv2.erode(threshold, kernel, iterations = 1)
# Dilation: Expanding the foreground
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm
dilation = cv2.dilate(threshold, kernel, iterations = 1)
# Boundaries extraction
edges = dilation - threshold
# Debbuging
cv2.imshow('edges', edges)
cv2.waitKey(0)
#print('binary image matrix')
#print(edges)
#w, h = edges.shape
#print('binary image matrix size')
#print(w)
#print(h)
#px = edges[13,93] # line, column
#print('Test específic coordinate')
#print(px)
#rows, cols = np.nonzero(edges)

grayscale = np.float32(img)

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

edges[dst > threshold * dst.max()] = 255

cv2.imshow("edges 2", edges)








# Load image:
#input_image = Image.open("test.png")
input_image = Image.open("test.png")

# Output image:
output_image = Image.new("RGB", input_image.size)
output_image.paste(input_image)
draw_result = ImageDraw.Draw(output_image)




# Find circles
rmin = 15
rmax = 50
steps = 100
threshold = 0.3 # 1.25 * 10 ** -2

print('threshold: ' + str(threshold))

points = []
for r in range(rmin, rmax + 1):
    for t in range(steps):
        points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

print('edges')
print(np.nonzero(edges))
print(zip(np.nonzero(edges)))
print(np.argwhere(edges == 255))

acc = defaultdict(int)
for x, y in np.argwhere(edges == 255): #canny_edge_detector(input_image):
    for r, dx, dy in points:
        a = x - dx
        b = y - dy
        acc[(a, b, r)] += 1

circles = []
for k, v in sorted(acc.items(), key=lambda i: -i[1]):
    x, y, r = k
    if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
        print(v / steps, x, y, r)
        circles.append((x, y, r))

# Expected
#0.74 46 140 43
#0.71 152 140 43
#0.7 99 56 43

for x, y, r in circles:
    draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(255,0,0,0))

# Save output image
output_image.save("result.png")
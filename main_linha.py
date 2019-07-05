from skimage import data, io, draw
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

image = io.imread('resultado_circulos_subtraidos.png', as_gray=True)

# Classic straight-line Hough transform
# hspace : 2-D ndarray of uint64 Hough transform accumulator.
# angles : ndarray Angles at which the transform is computed, in radians.
# distances : ndarray Distance values.
h, theta, d = hough_line(image)

#print(h)
#print(theta)
#print(d)

row1, col1 = image.shape

for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    x0 = 0
    x1 = col1 - 1
    y0 = (dist - 0 * np.cos(angle)) / (np.sin(angle) + 1e-8)
    y1 = (dist - (col1-1) * np.cos(angle)) / (np.sin(angle) + 1e-8)
    y0 = np.clip(y0, 0, row1-1)
    y1 = np.clip(y1, 0, row1-1)
    rr, cc = draw.line(int(y0), int(x0), int(y1), int(x1))
    #image[rr, cc] = 1

#io.imshow(image)

#img = np.zeros((10, 10), dtype=np.uint8)
#rr, cc = draw.line(1, 1, 8, 8)
#img[rr, cc] = 1
#io.imshow(img)

# Generating figure 1
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')
ax[0].set_axis_off()

ax[1].imshow(np.log(1 + h),
             extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
             cmap=cm.gray, aspect=1/1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')

ax[2].imshow(image, cmap=cm.gray)
for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
    ax[2].plot((0, image.shape[1]), (y0, y1), '-r')
    print(y0, y1)
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines')

plt.tight_layout()
plt.show()
"""
=========================================
Robust line model estimation using RANSAC
=========================================

In this example we see how to robustly fit a line model to faulty data using
the RANSAC (random sample consensus) algorithm.

Firstly the data are generated by adding a gaussian noise to a linear function.
Then, the outlier points are added to the data set.

RANSAC iteratively estimates the parameters from the data set.
At each iteration the following steps are performed:

1. Select ``min_samples`` random samples from the original data and check
   whether the set of data is valid (see ``is_data_valid`` option).
2. Estimate a model on the random subset
   (``model_cls.estimate(*data[random_subset]``) and check whether the
   estimated model is valid (see ``is_model_valid`` option).
3. Classify all the data points as either inliers or outliers by calculating
   the residuals using the estimated model (``model_cls.residuals(*data)``) -
   all data samples with residuals smaller than the ``residual_threshold``
   are considered as inliers.
4. If the number of the inlier samples is greater than ever before,
   save the estimated model as the best model. In case the current estimated
   model has the same number of inliers, it is considered as the best model
   only if the sum of residuals is lower.

These steps are performed either a maximum number of times or until one of
the special stop criteria are met. The final model is estimated using all the
inlier samples of the previously determined best model.

"""

from matplotlib import pyplot as plt
from skimage.measure import LineModelND, ransac
import numpy as np
import cv2
import os

clear = lambda: os.system('cls')
clear()

np.random.seed(seed=1)

# generate coordinates of line
x = np.arange(-200, 200)
y = 0.2 * x + 20
data = np.column_stack([x, y])

# add gaussian noise to coordinates
noise = np.random.normal(size=data.shape)
data += 0.5 * noise
data[::2] += 5 * noise[::2]
data[::4] += 20 * noise[::4]

# add faulty data
faulty = np.array(30 * [(180., -100)])
faulty += 10 * np.random.normal(size=faulty.shape)
data[:faulty.shape[0]] = faulty

# data = np.array([
#  [1.54703769e+02,4.98475460e+01],
#  [1.55188662e+02,5.40684374e+01],
#  [1.57144039e+02,5.21490541e+01],
#  [1.56344631e+02,5.60410075e+01],
#  [1.59156119e+02,5.17033392e+01],
#  [1.07041548e+02,7.61612798e+01],
#  [1.60748013e+02,5.22089779e+01],
#  [1.55012466e+02,5.39564754e+01],
#  [1.63054015e+02,5.26147088e+01],
#  [1.60562775e+02,7.95550664e+01],
#  [1.65483110e+02,5.33629584e+01],
#  [1.84265933e+02,4.98987607e+01],
#  [1.66810241e+02,5.28925982e+01],
#  [1.79117648e+02,3.60690715e+01],
#  [1.67650819e+02,5.31933309e+01],
#  [1.70397385e+02,5.95538303e+01],
#  [1.70221529e+02,5.38937789e+01],
#  [1.68446529e+02,3.58222940e+01],
#  [1.73265582e+02,5.46020004e+01],
#  [1.75766963e+02,5.08113179e+01]
# ])

img = cv2.imread('test2.png', 0)

data = []
rows, cols = np.nonzero(img)

#print(rows, cols)
print(rows.shape, cols.shape)
print(rows.max(), cols.max())

data = np.column_stack([rows, cols])

#for i in range(0, rows.shape[0]):
#        data2.append([rows[i], cols[i]])

#mat = np.hstack((rows, cols))
#print(mat)

#mylist = []
#for item in data:
#mylist.append(item)
#data = np.array(mat)

#print(data)
#print(data2)

# fit line using all data
#model = LineModelND()
#model.estimate(data)

# robustly fit line only using inlier data with RANSAC algorithm
model_robust, inliers = ransac(data, LineModelND, min_samples=3,
                               residual_threshold=1, max_trials=1000)
outliers = inliers == False

# generate coordinates of estimated models
line_x = np.arange(0, 250)
#line_y = model.predict_y(line_x)
line_y_robust = model_robust.predict_y(line_x)

fig, ax = plt.subplots()
ax.plot(data[inliers, 0], data[inliers, 1], '.b', alpha=0.6,
        label='Inlier data')
ax.plot(data[outliers, 0], data[outliers, 1], '.r', alpha=0.6,
        label='Outlier data')
#ax.plot(line_x, line_y, '-k', 
#        label='Line model from all data')
ax.plot(line_x, line_y_robust, '-b', 
        label='Robust line model')
ax.legend(loc='lower left')
plt.show()

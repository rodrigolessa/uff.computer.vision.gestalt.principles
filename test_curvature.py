import math
import numpy
from scipy.integrate import cumtrapz
from scipy.interpolate import UnivariateSpline

def linspace(a,b,n):
    return numpy.linspace(a,b,n)

def lissajousCurve(delta=0):
    noOfPoints=30
    a=9
    b=6
    pi = math.pi
    div = a
    if a>b :
        div = a
    else:
        div = b

    end = -float(pi*(float(b)/float(a)))
    offset = -float(pi*(float(b)/float(a)))/4
    t = linspace(offset,end+offset,noOfPoints)
    ret = []
    x = []
    y = []

    for tVal in t:
        x.append(math.sin(a * tVal + delta))
        y.append(math.sin(b * tVal))

    for i in range(len(x)):
        ret.append( {
            "x": x[i]*10 + 10,
            "y": y[i]*10 + 10
        } )

    return ret

def getUniformIndexes(arr):
    return range(len(arr))

def stripX(pts):
    result = []
    for pt in pts:
        result.append(pt['x'])

    return result

def stripY(pts):
    result = []
    for pt in pts:
        result.append(pt['y'])

    return result

def getDerivative(fx, xVal, derivative):
    if derivative == 0:
        return fx([xVal])[0]

    return fx.derivative(derivative)([xVal])[0]

#yArr MUST BE increasing
def getSpline(yArr, xArr):
    return UnivariateSpline(yArr, xArr, k=3, s=0)

def lengthRateOfChangeFunc(t, fx, fy):
    dxdt = getDerivative(fx, t, 1)
    dydt = getDerivative(fy, t, 1)
    val = math.sqrt(dxdt**2 + dydt**2)
    return val

def reparametriseWRTArcLength(fx_t, fy_t, tVals):

    rateOfChangePointsY = []
    rateOfChangePointsX = tVals
    for tVal in tVals:
        rateOfChangePointsY.append(lengthRateOfChangeFunc(tVal, fx_t, fy_t))

    tListReparametrised = cumtrapz(rateOfChangePointsY, rateOfChangePointsX, initial=0)
    return tListReparametrised

def getArcLengthSpline(shape):
    xArr = stripX(shape)
    yArr = stripY(shape)
    fx = getSpline(getUniformIndexes(xArr), xArr)
    fy = getSpline(getUniformIndexes(yArr), yArr)

    #Reparametrise WRT arc length
    tListReparametrised = reparametriseWRTArcLength(fx, fy, getUniformIndexes(xArr))
    
    fx = getSpline(tListReparametrised, xArr)
    fy = getSpline(tListReparametrised, yArr)
    return fx, fy, tListReparametrised

def calcCurvatureWithSplinesAtTVal(fx_arcLength, fy_arcLength, tVal):
    x_  = getDerivative(fx_arcLength, tVal, 1)
    x__ = getDerivative(fx_arcLength, tVal, 2)
    y_  = getDerivative(fy_arcLength, tVal, 1)
    y__ = getDerivative(fy_arcLength, tVal, 2)
    curvature = abs(x_* y__ - y_* x__) / math.pow(math.pow(x_, 2) + math.pow(y_, 2), 3.0 / 2.0)
    return 1.0/curvature

#a = [1,0,0], b = [[1],[0],[0]]
#[1,0,0]*[[1],[0],[0]] = [1]
def matrixMultiply(a, b):
    aNumRows = len(a)
    aNumCols = len(a[0])
    bNumRows = len(b)
    bNumCols = len(b[0])
    m =  [None] * aNumRows # initialize array of rows
    for r in range(aNumRows):
        m[r] = [None] * bNumCols # initialize the current row
        for c in range(bNumCols):
            m[r][c] = 0             # initialize the current cell
            for i in range(aNumCols):
                m[r][c] += a[r][i] * b[i][c]

    return m

def applyTransformationMatToSingleKeypoint(keypoint, transformationMat):
    return matrixMultiply(transformationMat, keypoint)

def applyTransformationMatrixToAllKeypoints(keypoints, transformationMat):
    ret = []
    for kp in keypoints:
        transformedKeypoint = applyTransformationMatToSingleKeypoint(kp, transformationMat)
        ret.append(transformedKeypoint)

    return ret

def convertSingleKeypointToMatrix(keypoint):
    return [[keypoint['x']], [keypoint['y']], [1]]

def convertKeypointsToMatrixKeypoints(keypoints):
    ret = []
    for kp in keypoints:
        newKeypoint = convertSingleKeypointToMatrix(kp)
        ret.append(newKeypoint)
    
    return ret

def convertSingleMatrixKeypoinToKeypointObject(arrayKeypoint):
    xVal = None
    if type(arrayKeypoint[0]) is list:
        xVal = arrayKeypoint[0][0]
    else: 
        xVal = arrayKeypoint[0] 

    yVal = None
    if type(arrayKeypoint[1]) is list: 
        yVal = arrayKeypoint[1][0]
    else: 
        yVal = arrayKeypoint[1] 
    
    return {
        'x': xVal,
        'y': yVal
    }

def _centeroidnp(arr):
    length = arr.shape[0]
    sum_x = numpy.sum(arr[:, 0])
    sum_y = numpy.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def getCenterPointOfShape_int(shape):
    c_pnt = getCenterPointOfShape_float(shape)
    return ( int(c_pnt[0]), int(c_pnt[1]))

def getCenterPointOfShape_float(shape):
    return _centeroidnp(numpy.asarray(shape))

def convertMatrixKeypointsToKeypointObjects(keypoints):
    ret = []
    for kp in keypoints:
        ret.append(convertSingleMatrixKeypoinToKeypointObject(kp))

    return ret

def applyTransformationMatrixToAllKeypointsObjects(keypoints, transformationMat):
    keypointsToken1 = convertKeypointsToMatrixKeypoints(keypoints)
    keypointsToken2 = applyTransformationMatrixToAllKeypoints(keypointsToken1, transformationMat)
    keypointsToken3 = convertMatrixKeypointsToKeypointObjects(keypointsToken2)
    return keypointsToken3

def getTranslateMatrix(x, y):
    return [
        [1, 0, x],
        [0, 1, y],
        [0, 0, 1]
    ]

def getScaleMatrix(scaleX, scaleY):
    return [[scaleX, 0, 0], [0, scaleY, 0], [0, 0, 1]]

#Get a plot of the change in curvature
def calcCurvatureGraphWRTTransformations(shape, pointIdx):
    result = []

    numberOfPoints = 100
    for i in range(numberOfPoints):
        percentageDone = float(i)/float(numberOfPoints)

        #set the current state of the shape
        scale = .1+(percentageDone*4.0)
        transShape1 = shape

        cntPntTemp = getCenterPointOfShape_float(convertKeypointsToMatrixKeypoints(transShape1))
        cntPnt = {
            'x': cntPntTemp[0],
            'y': cntPntTemp[1]
        }
        transShape1 = applyTransformationMatrixToAllKeypointsObjects(transShape1, getTranslateMatrix(-cntPnt['x'], -cntPnt['y']))
        transShape1 = applyTransformationMatrixToAllKeypointsObjects(transShape1, getScaleMatrix(math.sqrt(scale), 1/math.sqrt(scale)))
        transShape1 = applyTransformationMatrixToAllKeypointsObjects(transShape1, getTranslateMatrix( cntPnt['x'],  cntPnt['y']))

        #get the current curvature point
    
        fx, fy, tListReparametrised = getArcLengthSpline(shape)
        curvatureValue = calcCurvatureWithSplinesAtTVal(fx, fy, tListReparametrised[pointIdx])
        result.append( {'x': percentageDone, 'y': curvatureValue} )
    
    return result

def getMin2(inresult1, inresult2):

    x0 = [1.0, .1, .1]

    startX = inresult1[0].x
    endX = inresult1[inresult1.length - 1].x
    inputRes1Fx.fx = toArcLengthSpline_monotonicallyIncreasingX(inresult1)
    inputRes1Fx.startX = startX
    inputRes1Fx.endX = endX

    startX = inresult2[0].x
    endX = inresult2[inresult2.length - 1].x
    inputRes2Fx.fx = toArcLengthSpline_monotonicallyIncreasingX(inresult2)
    inputRes2Fx.startX = startX
    inputRes2Fx.endX = endX

    solution = nelderMead(functionToMin, x0, {maxIterations: 100})
    return solution

def generateAllTheInfo(shape, pointIdx):
    g_shape1 = shape
    g_shape2 = shape

    scale = 2.0
    cntPntTemp = getCenterPointOfShape_float(convertKeypointsToMatrixKeypoints(g_shape2))
    # cntPnt = {
    #     x: cntPntTemp[0],
    #     y: cntPntTemp[1]
    # }
    g_shape1 = applyTransformationMatrixToAllKeypointsObjects(g_shape1, getTranslateMatrix(-cntPntTemp[0], -cntPntTemp[1]))
    g_shape1 = applyTransformationMatrixToAllKeypointsObjects(g_shape1, getScaleMatrix(math.sqrt(scale), 1/math.sqrt(scale)))
    g_shape1 = applyTransformationMatrixToAllKeypointsObjects(g_shape1, getTranslateMatrix( cntPntTemp[0],  cntPntTemp[1]))

    inresult1 = calcCurvatureWithSplinesAtTVal(g_shape1, pointIdx, None)

    for i in range(1, g_shape1.length - 1):
        inresult2 = calcCurvatureWithSplinesAtTVal(g_shape2, i, None)

        g_min = getMin2(inresult1, inresult2)
        console.log("Check: " + i + " - " + g_min.fx)

        if g_min.fx < 0.01 :
            g_foundKeypoints.push(g_shape2[i])
            print("found: " + str(i))

print(generateAllTheInfo(lissajousCurve(), 5))
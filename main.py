# GOAL:
# A method to implement object reification based on 
# law of closure in Gestalt psychology.

# Construct object via combination of discontinuous edges, 
# detects key vertices and then divides boundary into small edges. 
# Virtual edge is generated to connect two edges as 
# straight line, ellipse, or circle. Phases of the project:

# Parameters
imgName = "gestalt-triangle-630x659.jpg"

# A - Edge and Boundary Detection

# In this work, binary images are analyzed for object
# detection. The boundaries and edges using morphology
# operations which is defined as Ic = I - Phi(I). The denotations I
# and Ic represent the original binary image and its boundary,
# respectively. The function Phi(I) generate the result by applying
# erosion operation or dilation operation to I.

# Ic = I - Phi(I)
# I = original binary image
# Ic = image boundary

# Detection of Edges Using Mathematical Morphological Operators

# 1 - First order derivative / gradient methods are as follows:
#     * Roberts operator
#     * Sobel operator
#     * Prewitt operator

APLICAR SOBEL

# 2 - Second order derivative:
#     * Laplacian
#     * Laplacian of Gaussian
#     * Difference of Gaussian

# 3 - Optimal edge detection:
#     * Canny edge detection

# 4 - Erosion: Shrinking the foreground

# 5 - Dilation: Expanding the foreground

# B - Key Vertex Detection

# Let Psi be coordinate set of N vertices on edge of I, and Psi =
# {(x[n], y[n]) | n={1, 2, ..., N}}. The definition of curvature [8]
# is formulated below, where k is curvature. The denotations x' and x'' 
# are, respectively, first-order differential of x and 
# second-order differential of x. The definition is the same as 
# the variable y. The determination of key point detection is defined 
# as, where Tau1 is a threshold to determine key vertex, 
# and Tau1 is set to 1.25 x 10 ** -2 in the experiment

# k = ((x' * y'') - (x'' * y')) / (((x' ** 2) + (y' ** 2)) ** (3/2))

# D(x, y)   = 1, if |K x,y| > Tau1
#           = 0, if else

# C - Virtual Edge Generation

# D - Object Constrution

APLICAR TRANFROMADA DE HOUGH
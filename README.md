# uff.computer.vision.gestalt.principles
A method to implement object reification based on law of closure in Gestalt psychology.

Construct object via combination of discontinuous edges, detects key vertices and then divides boundary into small edges. Virtual edge is generated to connect two edges as straight line, ellipse, or circle. Phases of the project:

A - Edge and Boundary Detection

Detection of Edges Using Mathematical Morphological Operators

1 - First order derivative / gradient methods are as follows:
    * Roberts operator
    * Sobel operator
    * Prewitt operator

2 - Second order derivative:
    * Laplacian
    * Laplacian of Gaussian
    * Difference of Gaussian

3 - Optimal edge detection:
    * Canny edge detection

4 - Erosion: Shrinking the foreground

5 - Dilation: Expanding the foreground

B - Key Vertex Detection

C - Virtual Edge Generation

D - Object Constrution

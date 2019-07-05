from numpy import *
from matplotlib.pyplot import *
from math import *

t=arange(0,4*pi/2,0.0001)
print(t)
#first set of parameters
A=1
B=1
a1=1
b1=2
d=pi/2
#second set of parameters
a2=2
b2=3
X1=[]
Y1=[]
Y2=[]
X2=[]
for i in range(len(t)):
    x1=A*sin(a1*t[i]+d)
    y1=B*sin(b1*t[i])
    x2=A*sin(a2*t[i]+d)
    y2=B*sin(b2*t[i])
    X1=append(X1,x1)
    Y1=append(Y1,y1)
    X2=append(X2,x2)
    Y2=append(Y2,y2)
figure()
plot(t,X1, color='blue')
plot(t,Y1, color='pink')
plot(t,X2, color='purple')
plot(t,Y2, color='green')
show()
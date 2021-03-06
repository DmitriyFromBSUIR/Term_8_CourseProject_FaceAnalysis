import numpy as n
import math
import matplotlib.pyplot as plt

A = n.matrix([[1,1/2,1/3,1/4],[1/1,1/3,1/4,1/5],[1/2,1/4,1/5,1/6],[1/3,1/5,1/6,1/7]])
B = n.matrix([[1, 0.0500, 0.3333, 0.2500], [0.500, 0.3333, 0.2500, 0.2000], [0.3333, 0.2500, 0.2000, 0.1667], [0.2500, 0.2000, 0.1667, 0.1429]])
C = n.matrix([[16, -129, 240, -140], [-120, 1.200,-2700, 1680], [240,-2700, 6.480,-4.200], [-140, 1.680,-4.200, 2.800]])


'''Square matrix: no rows = no columns'''
D = n.matrix([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
E = n.matrix([[1,2],[3,4]])

P = n.matrix([[1,1, -1,-1],[1,-1,-1,1]])

'''Matrix multiplication is not commutative its associative'''
#print A*C
#print B*C 
'''Using numpy for Inverse and Transpose'''
#print E.T
#print E.I

def rotate2D(angle):
    R = n.matrix([[n.cos(angle), n.sin(angle)],[-n.sin(angle), n.cos(angle)]])
    return n.asarray(R).reshape(-1) #converting matrix to array

#---------------------------------------------------------------------------

#plt.plot(n.multiply(rotate2D(0),P))
plt.plot(P)


#plt.ylabel("Rotation")
#plt.show()


#Solve
#http://stackoverflow.com/questions/6789927/is-there-a-python-module-to-solve-linear-equations
#2x + 3y + 4z = 2
#x + 4z + y = 2
#4z + 5y + 2x = 3 

a = [[2,1,2],[3,1,5],[4,4,4]]
b = [2,2,3]
#print n.linalg.solve(a, b)
S, V, D = n.linalg.svd(a, b)
#print S
#print V
#print D

#Year Population
#1960 3.0
#1970 3.7
#1975 4.1
#1980 4.5
#1985 4.8

a = [1960, 1970, 1975, 1980, 1985]
b = [3.0, 3.7, 4.1, 4.5, 4.8]

#plt.plot(a,b,"rx-")
#plt.show()

y = n.array([[1960,1],[1970,1],[1980,1],[1985,1]]) #3
p = n.array([3.0, 3.7, 4.1, 4.5, 4.8])

print n.linalg.lstsq(y, p)









'''
Created on 19-02-2013

@author: Wiktor
'''
import Test1
import cv2
from pylab import *

def greyLevelMapLambdaWay(II,b,a):
    (M,N)=II.shape
    cc=zeros((M,N),uint8)
    for i in range(len[II]):
        cc[i] = map(lambda x: max(min(x*a+b,255),0), II[i]) # MAP FUNCTION!!!!!!!
    return cc

I1=cv2.imread("GreenTest.jpg")
I2 = cv2.cvtColor(I1, cv2.COLOR_RGB2GRAY)

I2 = greyLevelMapLambdaWay(I2, 2, 1)
Test1.show1_OpenCV(I2)








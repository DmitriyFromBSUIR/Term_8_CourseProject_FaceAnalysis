'''
Created on 18-02-2013

@author: Wiktor
'''

import Exercises4_Tools as tools
import cv2
import numpy as np
from pylab import*

#-------------------------------------------------------------------------

def BoxFilter(I,n):
    kernel = np.ones((n,n), dtype=np.float)/float(n*n) #Creating normalized kernel
    #kernel = np.array([[3,10,3],[0,0,0],[-3,-10,-3]])/9.0 #Creating kernel manually and applying some cool filters
    Dst = cv2.filter2D(I, -1,kernel)
    return Dst

def addRandomUniformNoise(I):
    M,N=I.shape
    noise = np.random.rand(M,N)*60
    noise=tools.to_uint8(noise)#
    I += noise
    return I

def addGaussianRandomNoise(I, mu, sigma):
    M,N=I.shape
    gaussianNoise = np.random.normal(mu, sigma,(M,N))
    gaussianNoise=tools.to_uint8(gaussianNoise)
    I += gaussianNoise
    return I 

def addSaltNPepperNoise(I, density):
    M,N=I.shape
    I[((np.random.rand(M,N))<density)] = [0]
    I[((np.random.rand(M,N))<density)] = [255]
    return I





#------------------------------BODY---------------------------------------

I = cv2.imread("lena.png")
I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)

#Dst = BoxFilter(I,7)
#tools.show_Multiple_images_OpenCV(Oryginal=I, Filtered = Dst, Difference = cv2.absdiff(I, Dst))

#tools.show_1_image_OpenCV(addSaltNPepperNoise(I, 0.07))

Iun = addRandomUniformNoise(copy(I))
Igrn = addGaussianRandomNoise(copy(I), 0, 50)
Isp = addSaltNPepperNoise(copy(I), 0.07)

    
tools.show_Multiple_images_OpenCV(Iun=BoxFilter(Iun, 5), Igrn=BoxFilter(Igrn, 5), SnP=BoxFilter(I, 5))






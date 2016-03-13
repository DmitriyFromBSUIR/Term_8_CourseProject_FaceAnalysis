'''
Created on 18-02-2013

@author: Wiktor
'''

import Exercises4_Tools as tools
import cv2
import numpy as np
from pylab import*

#-------------------------------------------------------------------------


def MatchAll(I, template, thr):
    Ibcp = copy(I)
    (w,h,t) = shape(template)
    CrossCorr = cv2.matchTemplate(I, template, cv2.TM_CCORR_NORMED)
    cv2.normalize(CrossCorr,CrossCorr,0,255,cv2.NORM_MINMAX)
    
    for idxr, r in enumerate(CrossCorr):
        for idxc, c in enumerate(r):
            if c>= thr:
                x=idxc
                y=idxr
                x1=idxc+h
                x2=idxr+w
                cv2.rectangle(Ibcp, (x,y), (x1,x2),(60,45,255),2)
        
        
    tools.show_Multiple_images_pylab(Oryginal=I, CrossCor=CrossCorr, result = Ibcp)
        




#------------------------------BODY---------------------------------------
"""
I = cv2.imread("lena.png")
Ibcp = copy(I)
#I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
region = I[248:280,245:288]
(w,h,t) = shape(region)

#M1 = cv2.matchTemplate(I, region, cv2.TM_CCORR_NORMED)
#M2 = cv2.matchTemplate(I,region, cv2.TM_SQDIFF_NORMED)
M3 = cv2.matchTemplate(I,region, cv2.TM_CCOEFF_NORMED)

#Img = cv2.cvGetRectSubPix()

#tools.show_Multiple_images_pylab(d1=M1,d2=M2,d3=M3)

min_val, max_val, min_loc,max_loc = cv2.minMaxLoc(M3)
#cv2.normalize(M1,M1,0,255,cv2.NORM_MINMAX)

cv2.rectangle(Ibcp, max_loc, (max_loc[0]+h,max_loc[1]+w),(60,45,255),3)

#tools.show_1_image_pylab(Ibcp)
tools.show_Multiple_images_pylab(d1=Ibcp)
"""

I = cv2.imread("pattern1.jpg")

template = I[367:426,4:58]

MatchAll(I, template, 130)

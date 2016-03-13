#import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
from pylab import *
import cv2
import Test1

def displayHistogram(n,bins):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # get the corners of the rectangles for the histogram
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n
    
    
    # we need a (numrects x numsides x 2) numpy array for the path helper
    # function to build a compound path
    XY = np.array([[left,left,right,right], [bottom,top,top,bottom]]).T
    
    # get the Path object
    barpath = path.Path.make_compound_path_from_polys(XY)
    
    # make a patch out of it
    patch = patches.PathPatch(barpath, facecolor='blue', edgecolor='gray', alpha=0.8)
    ax.add_patch(patch)
    
    # update the view limits
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())
    
    plt.show()




#-------------------------------Main body
#loading an image
#I=cv2.imread("colorProblem.jpg") 
#R,G,B = I[:,:,0], I[:,:,1], I[:,:,2]
#for i in [R,G,B]:
#    I_eq = cv2.equalizeHist(i)
#    n,bins = histogram(array(I_eq),256,normed=True)
#    displayHistogram(n,bins)

#Equalizing blue channel and merging
I=cv2.imread("colorProblem.jpg")
R,G,B = I[:,:,0], I[:,:,1], I[:,:,2]
B_eq = cv2.equalizeHist(B)
I_mod = cv2.merge([R,G, B_eq])

Test1.show2_OpenCV(I, I_mod)

#------Example of histogram equalization

#loading an image
#I=cv2.imread("Eye3.jpg") 
#I=cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
#I_eq = cv2.equalizeHist(I)
#Test1.show2_OpenCV(I,I_eq)

#n,bins = histogram(array(I),256,normed=True)
#n2,bins2= histogram(array(I_eq),256,normed=True) 
#displayHistogram(n,bins)
#displayHistogram(n2,bins2)






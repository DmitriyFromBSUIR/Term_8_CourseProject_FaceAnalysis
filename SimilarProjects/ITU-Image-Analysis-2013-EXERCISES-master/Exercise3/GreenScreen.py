from pylab import *
import cv2
import Test1 as t
from numpy.core.fromnumeric import nonzero

def ChangeGreenToWhite(I, Gthr):
    R,G,B = cv2.split(I)
    I2 = nonzero((G>=Gthr)) #Zwraca indexy wszystkich elementow dla ktorych condition is true
    I[I2] = [255, 255, 255]
    return I

def ChangeGreenScreenRGB(I, Ibcg, Gthr):
    R,G,B = cv2.split(I)
    I2 = nonzero((G>=Gthr))
    I[I2] = Ibcg[I2]
    return I
    
    
    
    
I = cv2.imread("GreenTest.jpg")
Ibcg = cv2.imread("GreenTest_Background.jpg")


t.show1_OpenCV(ChangeGreenScreenRGB(I, Ibcg, 180))

"""
#Replace all numbers >2 and <10 with 1
T = np.array(range(0,20))
T1 = nonzero((T>3) & (T<10))
T[T1] = 1

print T
"""    
    



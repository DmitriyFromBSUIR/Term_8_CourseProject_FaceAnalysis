from pylab import *
import cv2 

#----------------------------Functions
#When a double-starred parameter is declared such as $**imgs$, 
#then all the keyword arguments from that point till the end are collected as a dictionary called $'imgs'$.
def showFigures(**imgs):
    figure()
    gray()
    
    for (counter, (k,v)) in enumerate(imgs.items()): #See how to use the $enumerate()$ function for creating a counter in a loop.
        subplot(1,len(imgs)  ,counter);imshow(v);title(k)

    show()
    
def DarkPupil(matrix,tr):
    temp=copy(matrix)
    I2 = nonzero(matrix>=tr)
    print I2[1][1]
    for i in range(len(I2[0])): #len() zwraca dlugosc arraya
        x=I2[0][i]
        y=I2[1][i]
        #print x,y
        temp[x][y]=255 #I[x][y]
    return temp

def BrightPupil(matrix,tr):
    temp=copy(matrix)
    I2 = nonzero(matrix<tr) #nonzero 
    print I2[1][1]
    for i in range(len(I2[0])): #len() zwraca dlugosc arraya
        x=I2[0][i]
        y=I2[1][i]
        #print x,y
        temp[x][y]=0 #I[x][y]
    return temp

def Glint(matrix,tr):
    temp=copy(matrix)
    I2 = nonzero(matrix<tr) #nonzero 
    print I2[1][1]
    for i in range(len(I2[0])): #len() zwraca dlugosc arraya
        x=I2[0][i]
        y=I2[1][i]
        #print x,y
        temp[x][y]=0 #I[x][y]
    return temp

    
        



#------------------------------Main body

I=cv2.imread("interlacedEye.jpg") # Loading an image using openCV

dark=I[1::2] #[start:end:step]
bright=I[::2]


#showFigures(Original_image=I,Dark_image=dark,Bright_image=bright)

showFigures(Original_image=I
            ,Dark_image=dark,
            Bright_image=bright,
            dark=DarkPupil(I, 30),
            bright=BrightPupil(I, 30),
            glintDark=Glint(dark, 30), 
            glintBright=Glint(bright, 30))



from pylab import *
import cv2 

#----------------------------Functions

def show1_OpenCV(image):
    ##This function define a window by namedWindow() and then show the image in that window
    image=array(image)
    cv2.namedWindow('By OpenCV', cv2.WINDOW_AUTOSIZE)## create a window called 'By OpenCV'Using the \nw{cv2.WINDOW_AUTOSIZE} parameter when defining a window display the image with its actual size in the window.
    cv2.imshow('By OpenCV', image) ## show the image in 'By OpenCV' window
    cv2.waitKey(0) ## the window will be closed with a (any)key press
    
def show2_OpenCV(*image):
    ## Showing the image using OpenCV
    im=[]
    for i in image:
        im.append(array(i))
    
    cv2.namedWindow('1', cv2.WINDOW_AUTOSIZE)## create a window called 'By OpenCV' Using the \nw{cv2.WINDOW_AUTOSIZE} parameter when defining a window display the image with its actual size in the window.
    cv2.imshow('1', im[0]) ## show the image in 'By OpenCV' window
    cv2.namedWindow('2', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('2', im[1]) 
    cv2.waitKey(0) ## the window will be closed with a (any)key press

          
def show1_pylab(image):
    ## Showing the image using pylab
    figure("By pylab")## Create a figure
    gray()
    title("1"); imshow(image)
    show()

def show2_pylab(*images):
    ## Showing the image using pylab
    figure("By pylab")## Create a figure
    gray()
    subplot(1,2,1);title("1"); imshow(images[0])## more about 'subplot()' : <http://www.scipy.org/Cookbook/Matplotlib/Multiple_Subplots_with_One_Axis_Label>
    subplot(1,2,2);title("2"); imshow(images[1])
    show()  
    
def showAll_OpenCV(**images): #DICTIONARY!!!! wtedy 
    for k,v in images.items():
        cv2.namedWindow(str(k), cv2.WINDOW_AUTOSIZE)## create a window called 'By OpenCV' Using the \nw{cv2.WINDOW_AUTOSIZE} parameter when defining a window display the image with its actual size in the window.
        cv2.imshow(str(k), array(v)) ## show the image in 'By OpenCV' window
    cv2.waitKey(0)
    
def showAll_PyLab(**images):
    figure('Window')
    gray()
    for (i,(k,v))in enumerate(images.items()): #Enumerating!!!!!!!
        subplot(1,len(images),i);title(k);imshow(v)
    show()
    
def displayFunc(vector):
    points=[]
    for i in range(len(vector)):
        points.append((i,vector[i]))
    bins=range(256)
    n=vector
    plt.grid(None, 'major', 'both')
    plot(bins, n, 'k-', linewidth = 5)
    axis([-2, 256, -2, 256])
    plt.show()
        
def grayLevelMap2(I,vector):
    for i in range(len(I)):
        I[i] = map(lambda x: max(min(vector[x], 255),0), I[i])
    return I


def grayLevelMap(I,b,a):
    newImage=[]
    for t in I:
        newBlock=[]
        for i in t:
            if a==0: a=1
            i=i*a+b
            if i>255: i = 255
            if i<0: i = 0 
            newBlock.append(i)
        newImage.append(newBlock)
    return newImage

def greyLevelMapLambdaWay(II,b,a):
    I=copy(II)
    for i in range(len(II)):
        I[i] = map(lambda x: max(min(x*a+b,255),0), II[i]) # MAP FUNCTION!!!!!!!
    return I

def resizeImgOdd(I, sampleFactor):
    return I[1::sampleFactor, 1::sampleFactor]    
    

def resizeWithOpenCV(I, *size):
    return cv2.resize(I, *size)   
        


##------------------------------Main body

# Loading an image using openCV
I1=cv2.imread("zone.jpg")
I2 = cv2.cvtColor(I1, cv2.COLOR_RGB2GRAY)#Changing color to gray scale

#show2_OpenCV(I2, resizeImgOdd(I2,0.5))

#show2_OpenCV(I2, resizeWithOpenCV(I2, (100,50)))


#show1_OpenCV(image = I1)
#show1_pylab(I1)
#show2_pylab(I1, I2)
#showAll_OpenCV(image1 = I1, image2 = I2, image3= I2)
#showAll_PyLab(image = I1, image2 = I2, image3 = I2)
#cv2.imwrite("image.tif", I1)

#vector = [255-x for x in range(0,256)]
#I2 = grayLevelMap2(I2, vector)
#I2 = greyLevelMapLambdaWay(I2, 2, 1)
#I3 = grayLevelMap(I2,90,0)

#show1_OpenCV(I2)
#R,G,B = I1[:,:,0], I1[:,:,1], I1[:,:,2]

#displayFunc(vector)





 
 
 
 
 
 
 
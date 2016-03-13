from pylab import *
import cv2 

#----------------------------Functions
def GetCoordinates(I):
    fig = figure(1)
    gray()
    imshow(I)
    
    fig.hold('on')
    X = ginput(2)
    print list(X)
    
    for (x,y) in X:            
        plt.plot(x,y)

    show(False)
    return X

def showFigures(**imgs):
    figure()
    gray()
    
    for (counter, (k,v)) in enumerate(imgs.items()):
        subplot(1,len(imgs) ,counter);imshow(v);title(k)

    show()
    
def extractChannels(I):
    R,G,B = I[:,:,0], I[:,:,1], I[:,:,2]
    return (R, G, B)

    
#----------------------------------Main body
I=cv2.imread("Flag.jpg")
I=cv2.cvtColor(I, cv2.COLOR_RGB2BGR)

ps = GetCoordinates(I)
flag = I[ int(ps[0][1]):int(ps[1][1]),int(ps[0][0]):int(ps[1][0]),:]
#chnl = extractChannels(I)
#GetCoordinates(I)
#showFigures(flag=I, Red=chnl[0], Green=chnl[1], Blue=chnl[2])
showFigures(Oryginal=I,Flag=flag)
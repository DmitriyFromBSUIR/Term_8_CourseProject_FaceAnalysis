import cv2 as cv3
import numpy as np

if __name__ == '__main__':
    print("Face & Eyes Analysis is running")
    #img = cv2.imread('opencv_logo.png',0)

    while 1:
        cap = cv3.VideoCapture(0)
        ret, img = cap.read()
        '''
        capture = cv3.CaptureFromCAM(0)
        img = cv3.QueryFrame(capture)
        gray = cv3.CreateImage(cv3.GetSize(img), 8, 1)
        edges = cv3.CreateImage(cv3.GetSize(img), 8, 1)
        '''
        filteredImg = cv3.medianBlur(img,5)
        grayImg = cv3.cvtColor(filteredImg, cv3.COLOR_BGR2GRAY)
        #grayImg = cv2.cvtColor(fimg, cv2.COLOR_BGR2GRAY)

        circles = cv3.HoughCircles(grayImg,cv3.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)

        #result = np.around(circles, decimals=0)
        circles = np.uint64(circles)
        for i in circles[0,:]:
            # draw the outer circle
            cv3.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv3.circle(img,(i[0],i[1]),2,(0,0,255),3)

        cv3.imshow('detected circles',img)
        cv3.waitKey(0)

    cv2.destroyAllWindows()
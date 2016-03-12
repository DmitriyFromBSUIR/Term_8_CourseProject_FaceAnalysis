from PIL import *
#from numpy import *
import numpy as np
import cv2

if __name__ == '__main__':
    print("Face Analysis is running")


    # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier('/home/dmitry/Projects/JetBrains_Workspace/PyCharm/Term_8_CourseProject_FaceAnalysis/haarcascades/haarcascade_frontalface_default.xml')
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    eye_cascade = cv2.CascadeClassifier('/home/dmitry/Projects/JetBrains_Workspace/PyCharm/Term_8_CourseProject_FaceAnalysis/haarcascades/haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)

    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)


        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                cv2.circle(roi_color, (ex+int(ew/2), ey+int(eh/2)), int(ew/2), (0,255,0), 2)

        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
        #print("PAM clustering algorithm is active")

    '''
        nimg = cv2.medianBlur(img, 5)
        cimg = cv2.cvtColor(nimg, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(nimg,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(roi_color,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(roi_color,(i[0],i[1]),2,(0,0,255),3)
    '''
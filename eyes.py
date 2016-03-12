import sys, os
from cv2 import *
#from highgui import *
#import opencv

def detectObjects(image):
    grayscale = cvCreateImage(cvSize(image.width, image.height), 8, 1)
    cvCvtColor(image, grayscale, CV_BGR2GRAY)

    storage = cvCreateMemStorage(0)
    cvClearMemStorage(storage)
    cvEqualizeHist(grayscale, grayscale)
    cascade = cvLoadHaarClassifierCascade('/usr/share/opencv/haarcascades/haarcascade_mcs_righteye.xml',
                                          cvSize(1, 1))
    faces = cvHaarDetectObjects(grayscale, cascade, storage, 1.2, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(50, 50))
    if faces:
        for i in faces:
            cvRectangle(image, cvPoint( int(i.x), int(i.y)),
                         cvPoint(int(i.x + i.width), int(i.y + i.height)),
                         CV_RGB(0, 255, 0), 3, 8, 0)

def run():
    cvNamedWindow("cam", 1)
    capture = cvCreateCameraCapture(0)
    orig = cvQueryFrame(capture)
    grey = cvCreateImage(cvGetSize(orig), IPL_DEPTH_8U, 1)
    face = cvCreateImage(cvGetSize(grey), IPL_DEPTH_8U, 1)
    while True:
        orig = cvQueryFrame(capture)
        cvCvtColor(orig, grey, CV_RGB2GRAY)
        detectObjects(orig)
        cvShowImage("cam", orig)
        k = cvWaitKey(10)
        if k == 'q': # press q to quit
            break

if __name__ == "__main__":
    run()
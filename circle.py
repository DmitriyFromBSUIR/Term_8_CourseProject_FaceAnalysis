# Project for Eye's Pupil detection and tracking (included some math analysis)
import sys
import math
import cv2 as cv3 # use OpenCV 3.0 with Python 3.4
import numpy as np

EDGE_DETECTOR_OF_LIE = 100

PATCH_OF_REFLECTED_LIGHT = 300

'''
    Experimental Data:

    ---1---
    Conditions:
        Distance from face to web-camera is approximatly 10 sm
        Evening, low light + additional light
    Result
        The square of eye pupil is 300 - 700 pixels
    ---2---
    Conditions:
        Distance from face to web-camera is approximatly 5 sm
        Evening, low light + additional light
    Result
        The square of eye pupil is approximatly 1100 - 1300 pixels (jumps 1600-1800)
'''

def LieDetectorByEyePupilResize(squareStart, squareResult):
    print("Time Interval is over")
    #squareResult = LieDetectorByEyePupilResize(circles)

    if math.fabs(squareResult - squareStart) > EDGE_DETECTOR_OF_LIE:
        print("He/She is liar and forger")
    else:
        print("He/She is fine human")


def circlesSquareCalculation(circlesCenterX, circlesCenterY, circleRadius):
    square = 0.0
    #for eyePupil in circles:
    square = math.pi * circleRadius * circleRadius
    print("")
    if square > PATCH_OF_REFLECTED_LIGHT:
        print("[DEBUG:] ", square)
    print("")
    return square


if __name__ == '__main__':
    print("Face & Eyes Analysis is running")
    #img = cv2.imread('opencv_logo.png',0)

    # set camera number to 1 for use additional usb camera
    cap = cv3.VideoCapture(1)
    if not cap.isOpened():
        print("Error! Can't capture the video")
        cap.release()
        cv3.destroyAllWindows()

    measureTimeInterval = int(sys.argv[1])

    # main loop
    while True:

        squareStart = 0.0
        squareResult = 0.0

        ret, frame = cap.read()
        img = frame.copy()

        gray = cv3.cvtColor(img,cv3.COLOR_BGR2GRAY)
        filtered_gray = cv3.medianBlur(gray,5)
        #filtered_gray = cv.GaussianBlur(gray,(5,5),0)
        cv3.imshow('Filtered_Grayscale_Image',filtered_gray)
        '''
            Python: cv2.HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) → circles
            Parameters:
            image – 8-bit, single-channel, grayscale input image.
            circles – Output vector of found circles. Each vector is encoded as a 3-element floating-point vector (x, y, radius)
            method – Detection method to use. Currently, the only implemented method is CV_HOUGH_GRADIENT , which is basically 21HT , described in [Yuen90].
            dp – Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height.
            minDist – Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
            param1 – First method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the higher threshold of the two passed to the Canny() edge detector (the lower one is twice smaller).
            param2 – Second method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
            minRadius – Minimum circle radius.
            maxRadius – Maximum circle radius.

        '''
        # detecting the cicles from videostream
        circles = cv3.HoughCircles(filtered_gray,cv3.HOUGH_GRADIENT,1,25,
                                    param1=220,param2=20,minRadius=5,maxRadius=25)

        #if measureTimeInterval == int(sys.argv[1]):
            #squareStart = LieDetectorByEyePupilResize(circles)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # if the detected circle square > PATCH_OF_REFLECTED_LIGHT (filtering REFLECTED_LIGHT circles)
                if circlesSquareCalculation( i[0], i[1], i[2] ) > PATCH_OF_REFLECTED_LIGHT:
                    # draw the outer circle
                    cv3.circle(frame,(i[0],i[1]),i[2],(0,255,0),1)
                    # draw the center of the circle
                    cv3.circle(frame,(i[0],i[1]),2,(0,0,255),2)
                # save measure in the begining of time interval
                if measureTimeInterval == int(sys.argv[1]):
                    if circlesSquareCalculation( i[0], i[1], i[2] ) > PATCH_OF_REFLECTED_LIGHT:
                        squareStart = circlesSquareCalculation( i[0], i[1], i[2] )
                # save measure in the end of time interval
                if circlesSquareCalculation( i[0], i[1], i[2] ) > PATCH_OF_REFLECTED_LIGHT:
                    squareResult = circlesSquareCalculation( i[0], i[1], i[2] )
        cv3.imshow('Videonistagmography System - Circles Detected', frame)

        # decrease the timer counter
        measureTimeInterval -= 1

        # in the end of time interval - analysis the eye pupil circle resize
        if measureTimeInterval == 0:
            LieDetectorByEyePupilResize(squareStart, squareResult)
            # regenerate the timer
            measureTimeInterval = int(sys.argv[1])

        #cv3.waitKey(25)
        k = cv3.waitKey(30) & 0xff
        if k == 27:
            break

    #
    cap.release()
    cv3.destroyAllWindows()

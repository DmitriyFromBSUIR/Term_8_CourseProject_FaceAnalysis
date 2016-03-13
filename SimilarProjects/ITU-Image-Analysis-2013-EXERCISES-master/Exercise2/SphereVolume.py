'''
Created on 07-02-2013

@author: wzdziechowski
'''
import math as m
import fractions as f




def calculateSphere (radius):
    while True:
        try:
            print "Radius:"
            radius = int(raw_input())
            result = f.Fraction(4,3)*m.pi*radius**3
            print result
        except ValueError:
            print "This is not a number try again." 

calculateSphere(5)    
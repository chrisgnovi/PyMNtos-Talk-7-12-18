#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 22:38:13 2018

@author: christophernovitsky
"""

import math as m
import numpy as np 
import math as m
import numpy as np 
import  scipy.optimize
import sympy as sp


def yAxis(utm):
    count = 0
    ang = [];
    yAxis = np.zeros(shape=(96,32))
    for i in range(0,32,2):
    
        P0 = [utm[0,i], utm[0,i+1]] 
        P1 = [utm[0,i], utm[0,i+1]+20] 
        P = [utm[1,i], utm[1,i+1]] 
   
        v1 = np.asarray(P0)-np.asarray(P1)
        v2 = np.asarray(P0)-np.asarray(P)
        a = np.dot(v1,v2)
        b = np.linalg.norm(v1)*np.linalg.norm(v2)
  
        ang = np.arccos(a/b)*180/m.pi
        
        x = np.array(np.linspace(0,360,96)+ ang)
        for k in range(0,len(x)): 
            if (x[k] > 360): x[k] -= 360
     
        yAxis[:,count] = x
        count +=1
    return yAxis

def sin2fit(y, x):
    # sin least-square fit function 
    # The elements of output parameter vector, s ( b in the function ) are:

    # s(1): sine wave amplitude (in units of y)
    # s(2): phase (phase is s(2)/(2*s(3)) in units of x)
    # s(3): offset (in units of y)

    yu = max(y)
    yl = min(y)
    yr = (yu-yl)                    # Range of y
    ym = np.mean(y)
    fit = lambda b,x: b[0]*(np.sin(2*m.pi*x/180 + 2*m.pi/b[1])) + b[2]
    fcn = lambda b: sum((fit(b,x) - y)**2)
    s =  scipy.optimize.fmin(fcn, [yr,  -1,  ym]);       # Minimise Least-Squares
    xp = np.linspace(min(x),max(x),1036);
    yp = fit(s,xp);
    
    return [xp,yp,s];

def sind(x):
    return sp.sin(x * sp.pi / 180)

def cosd(x):
    return sp.cos(x * sp.pi / 180)   
    
def tand(x):
    return sp.tan(x * sp.pi / 180)
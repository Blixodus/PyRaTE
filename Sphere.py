#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:13:22 2019

@author: atte
"""

import numpy as np
import math

import Ray

class Sphere:
    
    def __init__(self, centre, radius, colour, refr_index):
        self.centre = centre
        self.radius = radius
        self.colour = colour
        self.refr_index = refr_index
    
    
    # Calculate the intersection points between a ray and a sphere
    def intersection(self, ray : Ray):
        e = ray.eye
        d = ray.vect
        c = self.centre
        r = self.radius
        # Calculate discriminant
        dis = (d@(e-c))**2-(d@d)*((e-c)@(e-c)-r**2)
        # If discriminant is 0, return unique solution
        if(dis == 0):
            return (-d@(e-c))/(d@d)
        # If discriminant is positive, return minimum distance
        elif(dis > 0):
            dist1 = ((-d)@(e-c)+math.sqrt(dis))/(d@d)
            dist2 = ((-d)@(e-c)-math.sqrt(dis))/(d@d)
            # If one of the distances is negative, return the positive one
            if(dist1 < 0 or dist2 < 0):
                return max(dist1, dist2)
            return min(dist1, dist2)
        # Otherwise return value behind camera
        return -1
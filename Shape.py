#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 18:34:39 2019

@author: atte
"""

import numpy as np
import math

from Ray import *

class Shape:
    
    # Compute reflection of the ray coming from the currently in use eye
    def calculate_reflection(self, t, ray : Ray):
        # Calculate reflection point (new eye)
        r_point = ray.eye+t*ray.vect
        # Calculate normal vector
        norm_vect = self.calculate_norm_vect(r_point, ray)
        # Calculate new ray
        r_ray_vect = ray.vect-2*(ray@norm_vect)*norm_vect
        # Return new eye and ray
        return Ray(r_point, r_ray_vect)

class Sphere(Shape):
    
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
    
    def calculate_norm_vect(self, r_point, ray : Ray):
        # Normal vector is vector between point and centre
        norm_vect = r_point-self.centre
        if(norm_vect@ray<0):
            norm_vect = -norm_vect
        return norm_vect
    
class Plane(Shape):
    
    def __init__(self, point, norm_vect, colour):
        self.point = point
        self.norm_vect = norm_vect
        self.colour = colour
        
    def intersection(self, ray : Ray):
        p = self.point
        norm_vect = self.norm_vect
        e = ray.eye
        d = ray.vect
        # If ray is not parallel to the plane
        if(d@norm_vect != 0):
            # Return distance
            return -((e-p)@norm_vect)/(d@norm_vect)
        return -1
    
    def calculate_norm_vect(self, r_point, ray : Ray):
        if(self.norm_vect@ray<0):
            return -self.norm_vect
        return self.norm_vect
        
        
class Triangle(Shape):
    
    def __init__(self, p1, p2, p3, colour):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.colour = colour
        self.plane = Plane(np.cross(p1-p2, p1-p2))
        
    # TODO
    def intersection(self, ray : Ray):
        return -1
        
    def calculate_norm_vect(self, r_point, ray : Ray):
        return self.plane.calculate_norm_vect(r_point, ray)
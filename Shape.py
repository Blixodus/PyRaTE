#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 18:34:39 2019

@author: atte
"""

import numpy as np
import math
from abc import abstractmethod

from Ray import *

class Shape:
    
    # Compute reflection of the ray coming from the currently in use eye
    def calculate_reflection(self, t : float, ray : Ray) -> Ray:
        # Calculate reflection point (new eye)
        r_point = ray.eye+t*ray.vect
        # Calculate normal vector
        norm_vect = self.calculate_norm_vect(r_point, ray)
        # Calculate new ray
        r_ray_vect = ray.vect-2*(ray@norm_vect)*norm_vect
        # Return new eye and ray
        return Ray(r_point, r_ray_vect)
    
    @abstractmethod
    def intersection(self, ray : Ray) -> float:
        pass
    
    @abstractmethod
    def calculate_norm_vect(self, r_point : np.ndarray(3), ray : Ray) -> np.ndarray(3):
        pass
    
    @abstractmethod
    def edge_points(self) -> (np.ndarray(3), np.ndarray(3)):
        pass
    
    @abstractmethod
    def compute_color(self, r_point : np.ndarray(3), ray : Ray) -> np.ndarray(3):
        pass
    
class Sphere(Shape):
    
    def __init__(self, centre, radius, colour, refr_index):
        self.centre = centre
        self.radius = radius
        self.colour = colour
        self.refr_index = refr_index
    
    # Calculate the intersection points between a ray and a sphere
    def intersection(self, ray):
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
    
    def calculate_norm_vect(self, r_point, ray):
        # Normal vector is vector between point and centre
        norm_vect = r_point-self.centre
        if(norm_vect@ray<0):
            norm_vect = -norm_vect
        return norm_vect
    
    def edge_points(self):
        return self.centre-self.radius, self.centre+self.radius
    
class Plane(Shape):
    
    def __init__(self, norm_vect, point=np.array([0,0,0]), colour=[0,0,0]):
        self.point = point
        self.norm_vect = norm_vect/np.linalg.norm(norm_vect)
        self.colour = colour
        
    def intersection(self, ray):
        p = self.point
        norm_vect = self.norm_vect
        e = ray.eye
        d = ray.vect
        # If ray is not parallel to the plane
        if(d@norm_vect != 0):
            # Return distance
            return -((e-p)@norm_vect)/(d@norm_vect)
        return -1
    
    def calculate_norm_vect(self, r_point, ray):
        if(self.norm_vect@ray<0):
            return -self.norm_vect
        return self.norm_vect
    
    # TODO (?)
    def edge_points(self):
        pass
        
class Triangle(Shape):
    
    def __init__(self, p1, p2, p3, colour):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.colour = colour
        self.plane = Plane(np.cross(p1-p2, p1-p2))
        
    # TODO
    def intersection(self, ray):
        return -1
        
    def calculate_norm_vect(self, r_point, ray):
        return self.plane.calculate_norm_vect(r_point, ray)
    
    # TODO
    def edge_points(self):
        pass

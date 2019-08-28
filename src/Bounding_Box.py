#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 13:18:50 2019

@author: Atte Torri
"""

import numpy as np

from Shape import *
from Ray import *

class Bounding_Box:
    
    def __init__(self):
        self.items = []
        self.min_point = np.array([1000,1000,1000])
        self.max_point = np.array([-1000,-1000,-1000])
    
    def update_size(self):
        # Get all max and min coordinates
        maxs, mins = zip(*[obj.edge_points() for obj in self.items])
        for m in mins:
            self.min_point = np.minimum(m, self.min_point)
        for m in maxs:
            self.max_point = np.maximum(m, self.max_point)
    
    def edge_points(self):
        return self.max_point, self.min_point
    
    def add(self, *items):
        # Add all items into list
        self.items.extend(items)
        self.update_size()
    
    def remove(self, *items):
        # Create new list without removed items
        self.items = [e for e in self.items if e not in items]
        self.update_size()
    
    def intersection(self, ray : Ray) -> (float, 'Shape'):
        # Check if bounding box is intersected
        # Return -1 if not
        # Otherwise return closest intersection inside bounding box
        return self.find_closest(ray)
        pass
    
    # Find the closest object that instersects a ray
    def find_closest(self, ray) -> (float, 'Shape'):
        closest_obj = None
        t = -1
        for obj in self.items:
            t_new, curr_obj = obj.intersection(ray)
            # Change current closest object if this one is closer
            if(t_new>10e-4 and (t_new<t or t<0)):
                t = t_new
                closest_obj = curr_obj
        return t, closest_obj
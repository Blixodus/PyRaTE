#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 13:18:50 2019

@author: atte
"""

import numpy as np

from Shape import *
from Ray import *

class Bounding_Box:
    
    def __init__(self):
        self.items = []
        self.point = np.array([0,0,0])
        self.lengths = np.array([0,0,0])
    
    def update_size(self):
        # Get all max and min coordinates
        maxs, mins = zip(*[obj.edge_points() for obj in self.items])
        self.point = np.minimum(mins)
        self.lengths = np.maximum(maxs)-self.point
    
    def edge_points(self):
        return self.point+self.lengths, self.point
    
    def add(self, *items):
        # Add all items into list
        self.items.extend(items)
        self.update_size()
    
    def remove(self, *items):
        # Create new list without removed items
        self.items = [e for e in self.items if e not in items]
        self.update_size()
    
    def intersection(self, ray : Ray):
        # Check if bounding box is intersected
        # Return -1 if not
        # Otherwise return closest intersection inside bounding box
        pass
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 18:30:27 2019

@author: atte
"""

class Plane:
    
    def __init__(self, point, norm_vect, colour):
        self.point = point
        self.norm_vect = norm_vect
        self.colour = colour
        
    def intersection(self, ray):
        p = np.array(pl.get('p'))
        norm_vect = np.array(pl.get('norm_vect'))
        # If ray is not parallel to the plane
        if(d@norm_vect != 0):
            # Return distance
            return -((e-p)@norm_vect)/(d@norm_vect)
        return -1
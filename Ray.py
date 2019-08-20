#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:13:35 2019

@author: atte
"""

import numpy as np

from Shape import *

class Ray:
    
    def __init__(self, eye, vect, obj):
        self.eye = eye
        self.vect = vect/np.linalg.norm(vect)
        self.obj = obj
        
    def current_refr_index(self):
        if(self.obj == None):
            return 1
        else:
            return self.obj.refr_index
        
    def point(self, t):
        return self.eye+t*self.vect
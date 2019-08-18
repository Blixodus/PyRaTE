#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:13:35 2019

@author: atte
"""

import numpy as np

class Ray:
    
    def __init__(self, eye, vect):
        self.eye = eye
        self.vect = vect/np.linalg.norm(vect)
        
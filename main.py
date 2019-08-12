#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:45:07 2019

@author: Atte Torri
"""

import numpy as np
import PIL
import yaml

# Compute ray e+t*d based on eye position and pixel i, j, returns d
def compute_ray(m, k, i, j, eye, display, px_width, px_height):
    # Calculate the point on the display
    point = np.array([0,(i+0.5-m/2)*px_width,(j+0.5-k/2)*px_height])
    # Calculate location of point in space
    real = point+display
    # Calculate vector in direction of eye to point
    vect = real-eye
    # Normalize
    unit_v = vect/np.linalg.norm(vect)
    return unit_v

# Calculate the intersection points between a ray e+t*d and a sphere (c, r)
def sphere_intersection(e, d, sp):
    c = sp.get('c')
    r = sp.get('r')
    dis = (d@(e-c))**2-(d@d)*((e-c)@(e-c)-r**2)
    if(dis == 0):
        return (-d@(e-c)+np.sqrt(dis))/(d@d)
    elif(dis > 0):
        return np.minimum((-d@(e-c)+np.sqrt(dis))/(d@d), (-d@(e-c)-np.sqrt(dis))/(d@d))
    return -1

# Read variables and other data from YAML file
def read_data(name):
    with open(name, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

# Create file with image based on pixel values in numpy array
def create_image(m, k, pixels):
    im = PIL.Image.new('RGB', (m,k), color = (255, 255, 255))
    for i in range(m):
        for j in range(k):
            im.putpixel((i,j), (pixels[i, j, 0], pixels[i, j, 1], pixels[i, j, 2]))
    im.save("rt_output.png")



def main():
    # Get values from file
    defs = read_data("data.yaml")
    m = defs.get('view').get('m')
    k = defs.get('view').get('k')
    fov_horizontal = defs.get('view').get('fov_horizontal')
    fov_vertical = defs.get('view').get('fov_vertical')
    eye = np.array(defs.get('view').get('eye'))
    display = np.array(defs.get('view').get('display'))
    sp1 = defs.get('spheres').get('sp1')
    
    pixels = np.ones((m, k, 3), dtype = np.int8)*255
    px_width = 2*np.tan(np.deg2rad(fov_horizontal)/2)/m
    px_height = 2*np.tan(np.deg2rad(fov_vertical)/2)/k
    for i in range(m):
        for j in range(k):
            ray = compute_ray(m, k, i, j, eye, display, px_width, px_height)
            t = sphere_intersection(eye, ray, sp1)
            if(t>=0):
                pixels[i, j] = np.asarray([255, 0, 0])
    create_image(m, k, pixels)

if __name__ == "__main__":
    main()
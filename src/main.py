#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:45:07 2019

@author: Atte Torri
"""

# Import dependencies
import numpy as np
import PIL
import yaml
import time
import math
# Import classes
from Ray import *
from Shape import *
from Light import *
from Bounding_Box import *

# Compute ray e+t*d based on eye position and pixel i, j, returns d
def compute_ray(m, k, i, j, eye, display, px_width, px_height):
    # Calculate the point on the display
    point = np.array([0,(i+0.5-m/2)*px_width,(j+0.5-k/2)*px_height])
    # Calculate location of point in space
    real = point+display
    # Calculate vector in direction of eye to point
    vect = real-eye
    return Ray(eye, vect, None)

# Computes colors recursively
def compute_colour(ray, bbox, lights, refl, curr, view_dist, shad_enab, amb_light, amb_bright):
    if(curr <= refl):
        # Find closest object
        t, closest_obj = bbox.find_closest(ray)
        if(t>=0 and (t<=view_dist or view_dist == -1)):
            r_point = ray.point(t)
            return True, closest_obj.compute_colour(r_point, ray)
        # No closest object
        else:
            return False, [0, 0, 0]
    # Reflection limit reached
    return False, [0, 0, 0]
    
def generate_scene_objects(object_defs):
    print(1)
    result = None
    for obj in object_defs:
        if(obj == 'bbox'):
            print(obj)
            bbox = Bounding_Box()
            contents = generate_scene_objects(object_defs.get('bbox'))
            bbox.add(contents)
            return bbox
        elif(obj == 'spheres'):
            c = object_defs.get('spheres').get('sp1').get('c')
            r = object_defs.get('spheres').get('sp1').get('r')
            col = object_defs.get('spheres').get('sp1').get('col')
            refr = object_defs.get('spheres').get('sp1').get('refr')
            sp1 = Sphere(np.array(c), r, np.array(col), refr)
            print(2)
            return sp1
            
def generate_scene_lights(light_defs):
    pass

# Read variables and other data from YAML file
def read_data(name):
    with open(name, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

# Create file with image based on pixel values in numpy array
def create_image(m, k, pixels, name):
    im = PIL.Image.new('RGB', (m,k), color = (255, 255, 255))
    for i in range(m):
        for j in range(k):
            im.putpixel((i,j), (int(pixels[i, j, 0]*255), int(pixels[i, j, 1]*255), int(pixels[i, j, 2]*255)))
    im.save(name)

# Transfrom m*k array to their average values in m_base*k_base array
def super_sample(m, k, m_base, k_base, pixels, ssample):
    real_pixels = np.zeros((m, k, 3), dtype = np.float)
    for i in range(m_base):
        for j in range(k_base):
            average = np.zeros(3)
            for x in range(ssample):
                for y in range(ssample):
                    average = average + pixels[i*ssample+x, j*ssample+y]
            real_pixels[i, j] = average/(ssample**2)
    return real_pixels

def compute_canvas(m, k, eye, display, px_width, px_height, bbox, refl_num, view_distance, lights, shad_enab, amb_light, amb_bright, debug):
    pixels = np.ones((m, k, 3), dtype = np.float)*0.05
    for i in range(m):
        if(i%int(float(m)/10) == 0):
            print("Computing ray {} of {} ({}%)".format(i*k, m*k, i/m*100))
        for j in range(k):
            ray = compute_ray(m, k, i, j, eye, display, px_width, px_height)
            b, colour= compute_colour(ray, bbox, lights, refl_num, 0, view_distance, shad_enab, amb_light, amb_bright)
            if(b):
                pixels[i, j] = colour
    return pixels

def main():
    begin = time.time()
    # Get values from file
    defs = read_data("../data.yaml")
    # Load all settings
    settings = defs.get('settings')
    ssample = int(settings.get('ssample'))
    eps = settings.get('epsilon')
    refl_num = settings.get('refl_num')
    debug = settings.get('debug')
    shad_enab = settings.get('shadows')
    amb_light = settings.get('ambient_light')
    amb_bright = settings.get('ambient_bright')
    # Load all view data
    view = defs.get('view')
    m_base = int(view.get('m'))
    k_base = int(view.get('k'))
    fov_horizontal = view.get('fov_horizontal')
    fov_vertical = view.get('fov_vertical')
    eye = np.array(view.get('eye'))
    display = np.array(view.get('display'))
    view_distance = view.get('dist')
    # Load all objects in scene
    object_defs = defs.get('objects')
    # Load all lights
    light_defs = defs.get('lights')
    
    # Calculate needed values
    m = ssample*m_base
    k = ssample*k_base
    print("Rendering supersample image of size", m, "x", k)
    px_width = 2*np.tan(np.deg2rad(fov_horizontal)/2)/m
    px_height = 2*np.tan(np.deg2rad(fov_vertical)/2)/k
    # Generate scenery
    bbox = generate_scene_objects(object_defs)
    lights = generate_scene_lights(light_defs)
    # Compute the colors for all pixels
    pixels = compute_canvas(m, k, eye, display, px_width, px_height, bbox, refl_num, view_distance, lights, shad_enab, amb_light, amb_bright, debug)
    
    # Create final images                   
    print("Creating file named rt_output_big.png for large aliased image")
    create_image(m, k, pixels, "../outputs/rt_output_big.png")
    print("Rendering final image of size", m_base, "x", k_base)
    real_pixels = super_sample(m, k, m_base, k_base, pixels, ssample)
    print("Creating file named rt_output.png for anti-aliased image")
    create_image(m_base, k_base, real_pixels, "../outputs/rt_output.png")
    
    # Print time spent
    end = time.time()
    print("Rendered in", end-begin, "seconds")

if __name__ == "__main__":
    main()
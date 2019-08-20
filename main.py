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
    return Ray(eye, vect)

# Find the closest object that instersects a ray
def find_closest(eye, ray, obj, eps):
    closest_obj = ('none', 'none')
    t = -1
    # Calculate new ray
    # Test for every type of object
    for curr_type_name in obj:
        curr_type = obj.get(curr_type_name)
        # Test for every object in selected type
        for curr_obj_name in curr_type:
            curr_obj = curr_type.get(curr_obj_name)
            # If the object is a sphere, calculate ray and sphere intersections
            if(curr_type_name == 'spheres'):
                t_new = sphere_intersection(eye, ray, curr_obj)
            # If the object is a triangle, calculate ray and triangle intersections
            elif(curr_type_name == 'triangles'):
                t_new = triangle_intersection()
            # If the object is a plane, calculate ray and plane intersections
            elif(curr_type_name == 'planes'):
                t_new = plane_intersection(eye, ray, curr_obj)
            # Change current closest object if this one is closer
            if(t_new>eps and (t_new<t or t<0)):
                t = t_new
                closest_obj = (curr_type_name, curr_obj_name)
    return (t, closest_obj)

# Computes colors recursively
def compute_colour(ray, n, obj, eps, refl, curr, view_dist, lights, shad_enab, amb_light, amb_bright):
    if(curr <= refl):
        # Find closest object
        t, closest_obj = find_closest(ray, obj, eps)
        if(t>=0 and (t<=view_dist or view_dist == -1)):
            obj_type = closest_obj[0]
            o = obj.get(closest_obj[0]).get(closest_obj[1])
            # Compute reflection eye and ray
            eye_refl, ray_refl = reflection_ray(eye, ray, t, o, obj_type)
            # Compute colour from reflection
            b_refl, color_refl, refl_obj = compute_colour(eye_refl, ray_refl, n, obj, eps, refl, curr+1, view_dist, lights, shad_enab, amb_light, amb_bright)
            # Compute refraction eye and ray
            b_refr, eye_refr, ray_refr, F_refl, F_refr = refraction_ray(eye, ray, t, n, o, obj_type)
            # Compute colour from refraction
            b, color_refr, refr_obj = compute_colour(eye_refr, ray_refr, o.get('refr'), obj, eps, refl, curr+1, view_dist, lights, shad_enab, amb_light, amb_bright)
            # Compute shadow if enabled
            shadow_fact = 1 if (shad_enab == 0) else shadow(eye, ray, t, lights, obj_type, o, obj, eps, amb_light, amb_bright)
            # Return final colour
            if(b_refr and b_refl):
                colour = shadow_fact*(np.array(o.get('col'))+F_refl*o.get('refl')*color_refl+F_refr*np.array(color_refr))
                return True, colour, closest_obj[1]+" ( -> "+refl_obj+") ( -> "+refr_obj+")"
            elif(b_refl):
                colour = shadow_fact*(np.array(o.get('col'))+o.get('refl')*color_refl)
                return True, colour, closest_obj[1]+" -> "+refl_obj
            else:
                colour = shadow_fact*(np.array(o.get('col')))
                return True, colour, closest_obj[1]+" -> None"
        # No closest object
        else:
            return False, [0, 0, 0], "Nothing"
    # Reflection limit reached
    return False, [0, 0, 0], "Limit"
    

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

def compute_canvas(m, k, eye, display, px_width, px_height, objects, eps, refl_num, view_distance, lights, shad_enab, amb_light, amb_bright, debug):
    pixels = np.ones((m, k, 3), dtype = np.float)*0.05
    for i in range(m):
        if(i%int(float(m)/10) == 0):
            print("Computing ray {} of {} ({}%)".format(i*k, m*k, i/m*100))
        for j in range(k):
            ray = compute_ray(m, k, i, j, eye, display, px_width, px_height)
            b, color, obj = compute_colour(ray, 1.0, objects, eps, refl_num, 0, view_distance, lights, shad_enab, amb_light, amb_bright)
            if(b):
                pixels[i, j] = color
            if(debug == 1 and i%(m/10) == 0 and j%(k/10) == 0):
                print("DEBUG:", i, j, obj, j)
            if(debug == 2 and obj != "Nothing"):
                print("DEBUG:", i, j, obj, j)
    return pixels

def main():
    begin = time.time()
    # Get values from file
    defs = read_data("data.yaml")
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
    objects = defs.get('objects')
    # Load all lights
    lights = defs.get('lights')
    
    # Calculate needed values
    m = ssample*m_base
    k = ssample*k_base
    print("Rendering supersample image of size", m, "x", k)
    px_width = 2*np.tan(np.deg2rad(fov_horizontal)/2)/m
    px_height = 2*np.tan(np.deg2rad(fov_vertical)/2)/k
    # Compute the colors for all pixels
    pixels = compute_canvas(m, k, eye, display, px_width, px_height, objects, eps, refl_num, view_distance, lights, shad_enab, amb_light, amb_bright, debug)
    
    # Create final images                   
    print("Creating file named rt_output_big.png for large aliased image")
    create_image(m, k, pixels, "rt_output_big.png")
    print("Rendering final image of size", m_base, "x", k_base)
    real_pixels = super_sample(m, k, m_base, k_base, pixels, ssample)
    print("Creating file named rt_output.png for anti-aliased image")
    create_image(m_base, k_base, real_pixels, "rt_output.png")
    
    # Print time spent
    end = time.time()
    print("Rendered in", end-begin, "seconds")

if __name__ == "__main__":
    main()
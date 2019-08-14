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
    # Calculate discriminant
    dis = (d@(e-c))**2-(d@d)*((e-c)@(e-c)-r**2)
    # If discriminant is 0, return unique solution
    if(dis == 0):
        return (-d@(e-c))/(d@d)
    # If discriminant is positive, return minimum distance
    elif(dis > 0):
        return np.minimum(((-d)@(e-c)+np.sqrt(dis))/(d@d), ((-d)@(e-c)-np.sqrt(dis))/(d@d))
    # Otherwise return value behind camera
    return -1

def triangle_intersection():
    return -1

def plane_intersection(e, d, pl):
    p = np.array(pl.get('p'))
    norm_vect = np.array(pl.get('norm_vect'))
    # If ray is not parallel to the plane
    if(d@norm_vect != 0):
        # Return distance
        return ((e-p)@norm_vect)/(d@norm_vect)
    return -1

# Find the closest object that instersects a ray
def find_closest(eye, ray, obj, eps):
    closest_obj = ('none', 'none')
    t = -1
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

# Compute reflection of the ray coming from the currently in use eye
def reflection_ray(eye, ray, t, obj, obj_type):
    # Calculate reflection point (new eye)
    r_point = eye+t*ray
    # Calculate normal vector
    norm_vect = np.zeros(3)
    if(obj_type == 'spheres'):
        norm_vect = r_point-np.array(obj.get('c'))
        if(norm_vect@ray>0):
            norm_vect = -norm_vect
    elif(obj_type == 'triangles'):
        v1 = np.array(obj.get('p1'))-np.array(obj.get('p2'))
        v2 = np.array(obj.get('p1'))-np.array(obj.get('p3'))
        norm_vect = np.cross(v1, v2)
        if(norm_vect@ray<0):
            norm_vect = np.cross(v2, v1)
    elif(obj_type == 'planes'):
        norm_vect = np.array(obj.get('norm_vect'))
        if(norm_vect@ray>0):
            norm_vect = -norm_vect
    # Calculate new ray
    r_ray = ray+2*(ray@norm_vect)*norm_vect
#    r_parall = (ray@norm_vect)/(norm_vect@norm_vect)*norm_vect
#    r_ortho = ray - r_parall
#    w = np.cross(norm_vect, r_ortho)
#    x1 = np.cos(np.pi)/np.linalg.norm(r_ortho)
#    x2 = np.sin(np.pi)/np.linalg.norm(w)
#    r_ortho_rot = np.linalg.norm(r_ortho)*(x1*r_ortho+x2*w)
#    r_ray = r_ortho_rot+r_parall
    
    if(obj_type == 'sphere'):
        print(r_point, np.array(obj.get('c')), ray, norm_vect, r_ray)
        print(np.arccos(ray@norm_vect), np.arccos(r_ray@norm_vect))
    # Return new eye and ray
    return r_point, -r_ray

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
            im.putpixel((i,j), (pixels[i, j, 0], pixels[i, j, 1], pixels[i, j, 2]))
    im.save(name)

# Transfrom m*k array to their average values in m_base*k_base array
def super_sample(m, k, m_base, k_base, pixels, ssample):
    real_pixels = np.zeros((m, k, 3), dtype = np.uint8)
    for i in range(m_base):
        for j in range(k_base):
            average = np.zeros(3)
            for k in range(ssample):
                for l in range(ssample):
                    average = average + pixels[i*ssample+k, j*ssample+l]
            real_pixels[i, j] = average/(ssample**2)
    return real_pixels



def main():
    # Get values from file
    defs = read_data("data.yaml")
    # Load all view data
    view = defs.get('view')
    m_base = int(view.get('m'))
    k_base = int(view.get('k'))
    fov_horizontal = view.get('fov_horizontal')
    fov_vertical = view.get('fov_vertical')
    eye = np.array(view.get('eye'))
    display = np.array(view.get('display'))
    view_distance = view.get('dist')
    ssample = int(view.get('ssample'))
    eps = view.get('epsilon')
    # Load all objects in scene
    objects = defs.get('objects')
    
    m = ssample*m_base
    k = ssample*m_base
    print(m, k)
    pixels = np.ones((m, k, 3), dtype = np.uint64)*20
    px_width = 2*np.tan(np.deg2rad(fov_horizontal)/2)/m
    px_height = 2*np.tan(np.deg2rad(fov_vertical)/2)/k
        
    
    for i in range(m):
        for j in range(k):
            ray = compute_ray(m, k, i, j, eye, display, px_width, px_height)
            t, closest_obj = find_closest(eye, ray, objects, eps)
            if(t>=0 and (t<=view_distance or view_distance == -1)):
                pixels[i, j] = objects.get(closest_obj[0]).get(closest_obj[1]).get('col')
#                new_eye, new_ray = reflection_ray(eye, ray, t, objects.get(closest_obj[0]).get(closest_obj[1]), closest_obj[0])
#                t2, closest_obj2 = find_closest(new_eye, new_ray, objects, eps)
#                if(t2>=0 and (t2<=view_distance or view_distance == -1)):
#                    pixels[i, j] = pixels[i, j]/2+np.array(objects.get(closest_obj2[0]).get(closest_obj2[1]).get('col'))/2
    real_pixels = super_sample(m, k, m_base, k_base, pixels, ssample)
    create_image(m_base, k_base, real_pixels, "rt_output.png")
    create_image(m, k, pixels, "rt_output_big.png")

if __name__ == "__main__":
    main()
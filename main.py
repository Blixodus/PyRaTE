#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 12:45:07 2019

@author: Atte Torri
"""

import numpy as np
import PIL
import yaml
import time

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
        dist1 = ((-d)@(e-c)+np.sqrt(dis))/(d@d)
        dist2 = ((-d)@(e-c)-np.sqrt(dis))/(d@d)
        # If one of the distances is negative, return the positive one
        if(dist1 < 0):
            return dist2
        if(dist2 < 0):
            return dist1
        return np.minimum(dist1, dist2)
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
        return -((e-p)@norm_vect)/(d@norm_vect)
    return -1

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

# Compute normal vector
def norm_ray(obj_type, obj, r_point, ray):
    norm_vect = np.zeros(3)
    if(obj_type == 'spheres'):
        # Normal vector is vector between point and centre
        norm_vect = r_point-np.array(obj.get('c'))
        if(norm_vect@ray<0):
            norm_vect = -norm_vect
    elif(obj_type == 'triangles'):
        v1 = np.array(obj.get('p1'))-np.array(obj.get('p2'))
        v2 = np.array(obj.get('p1'))-np.array(obj.get('p3'))
        norm_vect = np.cross(v1, v2)
        if(norm_vect@ray<0):
            norm_vect = np.cross(v2, v1)
    elif(obj_type == 'planes'):
        norm_vect = np.array(obj.get('norm_vect'))
        if(norm_vect@ray<0):
            norm_vect = -norm_vect
    # Normalize normal vector
    norm_vect = -norm_vect/np.linalg.norm(norm_vect)
    return norm_vect

# Compute reflection of the ray coming from the currently in use eye
def reflection_ray(eye, ray, t, obj, obj_type):    
    # Calculate reflection point (new eye)
    r_point = eye+t*ray
    # Calculate normal vector
    norm_vect = norm_ray(obj_type, obj, r_point, ray)
    # Calculate new ray
    r_ray = ray-2*(ray@norm_vect)*norm_vect
    # Return new eye and ray
    return r_point, r_ray

# Calculate Fresnel equations
def fresnel(n1, n2, theta, phi):
    # Fresnell parallel
    F_paral = ((n2*np.cos(theta)-n1*np.cos(phi))/(n2*np.cos(theta)+n1*np.cos(phi)))**2
    # Fresnell orthogonal
    F_ortho = ((n1*np.cos(phi)-n2*np.cos(theta))/(n1*np.cos(phi)+n2*np.cos(theta)))**2
    # Combine
    F_refl = 0.5*(F_paral+F_ortho)
    return F_refl, 1-F_refl

# Compute refraction of the ray
def refraction_ray(eye, ray, t, n, obj_enter, obj_type):
    n_t = obj_enter.get('refr')
    if(n_t == 0):
        return False, eye, ray, 1, 0
    # Calculate refraction point (new eye)
    r_point = eye+t*ray
    # Calculate normal vector to the object that will be entered
    norm_vect = norm_ray(obj_type, obj_enter, r_point, ray)
    # Calculate angle between ray and normal vector
    theta = np.pi/2-np.arccos((norm_vect@ray)/(np.linalg.norm(norm_vect)*np.linalg.norm(ray)))
    # Calculate angle of refraction ray
    phi = np.arccos(1-(n**2*(1-np.cos(theta)**2)/n_t**2))**2
    # Check that the angle is not over pi/2
    if(phi > np.pi/2 or phi < -np.pi/2):
        # Calculate amount of light refracted and reflected
        F_refl, F_refr = fresnel(n, n_t, theta, phi)
        # Calculate refracted ray
        ray_refr = (n*(ray+norm_vect*np.cos(theta))/n_t)-norm_vect*np.cos(phi)
        # Return eye and ray
        return True, r_point, ray_refr, F_refl, F_refr
    return False, r_point, ray, 1, 0

# Compute ambient light from above for a certain point
def ambient_light(eye, ray, norm_vect, bright, objects, eps):
    # Calculate shadow ray towards light source
    shadow_ray = np.array([0, 0, -1])
    # Compute closest object in shadow ray trajectory
    t, closest = find_closest(eye, shadow_ray, objects, eps)
    if(t<eps):
        angle = np.arccos((norm_vect@shadow_ray)/(np.linalg.norm(norm_vect)*np.linalg.norm(shadow_ray)))
        if(angle < np.pi/2 and angle > -np.pi/2):
            dimm = (np.pi/2-angle)/np.pi*2
            return dimm*bright
    return np.array([0, 0, 0])

# Compute how much light gets to a given point
def shadow(eye, ray, t, lights, obj_type, obj, objects, eps, amb_light, amb_bright):
    # Calculate collision point
    point = eye+t*ray
    bright = 0.0
    norm_vect = norm_ray(obj_type, obj, point, ray)
    for light_name in lights:
        l = lights.get(light_name)
        # Calculate shadow ray towards light source
        shadow_ray = np.array(l.get('pos'))-point
        # Calculate distance to light source
        dist = np.linalg.norm(shadow_ray)
        # Compute closest object in shadow ray trajectory
        t, closest = find_closest(point, shadow_ray, objects, eps)
        if(dist < t or t==-1):
            angle = np.arccos((norm_vect@shadow_ray)/(np.linalg.norm(norm_vect)*np.linalg.norm(shadow_ray)))
            if(angle < np.pi/2):
                dimm = (np.pi/2-angle)/np.pi*2
                dist_factor = 1/(dist**2)
                bright += dist_factor*dimm*l.get('bright')
    if(amb_light == 1):
        bright += ambient_light(point, ray, norm_vect, amb_bright, objects, eps)
    return bright    

# Computes colors recursively
def compute_color(eye, ray, n, obj, eps, refl, curr, view_dist, lights, shad_enab, amb_light, amb_bright):
    if(curr <= refl):
        # Find closest object
        t, closest_obj = find_closest(eye, ray, obj, eps)
        if(t>=0 and (t<=view_dist or view_dist == -1)):
            obj_type = closest_obj[0]
            o = obj.get(closest_obj[0]).get(closest_obj[1])
            # Compute reflection eye and ray
            eye_refl, ray_refl = reflection_ray(eye, ray, t, o, obj_type)
            # Compute color from reflection
            b_refl, color_refl, refl_obj = compute_color(eye_refl, ray_refl, n, obj, eps, refl, curr+1, view_dist, lights, shad_enab, amb_light, amb_bright)
            # Compute refraction eye and ray
            b_refr, eye_refr, ray_refr, F_refl, F_refr = refraction_ray(eye, ray, t, n, o, obj_type)
            # Compute color from refraction
            b, color_refr, refr_obj = compute_color(eye_refr, ray_refr, o.get('refr'), obj, eps, refl, curr+1, view_dist, lights, shad_enab, amb_light, amb_bright)
            # Compute shadow if enabled
            shadow_fact = 1 if (shad_enab == 0) else shadow(eye, ray, t, lights, obj_type, o, obj, eps, amb_light, amb_bright)
            # Return final color
            if(b_refr and b_refl):
                color = shadow_fact*(np.array(o.get('col'))+F_refl*o.get('refl')*color_refl+F_refr*np.array(color_refr))
                return True, color, closest_obj[1]+" ( -> "+refl_obj+") ( -> "+refr_obj+")"
            elif(b_refl):
                color = shadow_fact*(np.array(o.get('col'))+o.get('refl')*color_refl)
                return True, color, closest_obj[1]+" -> "+refl_obj
            else:
                color = shadow_fact*(np.array(o.get('col')))
                return True, color, closest_obj[1]+" -> None"
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
    
    m = ssample*m_base
    k = ssample*k_base
    print("Rendering supersample image of size", m, "x", k)
    pixels = np.ones((m, k, 3), dtype = np.float)*0.05
    px_width = 2*np.tan(np.deg2rad(fov_horizontal)/2)/m
    px_height = 2*np.tan(np.deg2rad(fov_vertical)/2)/k
    
    for i in range(m):
        if(i%int(float(m)/10) == 0):
            print("Computing ray {} of {} ({}%)".format(i*k, m*k, i/m*100))
        for j in range(k):
            ray = compute_ray(m, k, i, j, eye, display, px_width, px_height)
            b, color, obj = compute_color(eye, ray, 1.0, objects, eps, refl_num, 0, view_distance, lights, shad_enab, amb_light, amb_bright)
            if(b):
                pixels[i, j] = color
            if(debug == 1 and i%(m/10) == 0 and j%(k/10) == 0):
                print("DEBUG:", i, j, obj, j)
            if(debug == 2 and obj != "Nothing"):
                print("DEBUG:", i, j, obj, j)
                    
    print("Creating file named rt_output_big.png for large aliased image")
    create_image(m, k, pixels, "rt_output_big.png")
    print("Rendering final image of size", m_base, "x", k_base)
    real_pixels = super_sample(m, k, m_base, k_base, pixels, ssample)
    print("Creating file named rt_output.png for anti-aliased image")
    create_image(m_base, k_base, real_pixels, "rt_output.png")
    
    end = time.time()
    print("Rendered in", end-begin, "seconds")

if __name__ == "__main__":
    main()
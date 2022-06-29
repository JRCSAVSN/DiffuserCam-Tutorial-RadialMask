import torch
from math import cos, sin, radians
from random import random
import numpy as np

def det(ux, uy, vx, vy):
    return ux*vy - uy*vx

def preplot(img):
    if type(img) == torch.Tensor:
        img = img.numpy().transpose(1, 2, 0)
    elif img.shape[-1]!=3:
        img = img.transpose(1, 2, 0)
    aux = img[:, :, 0]
    img_color = np.zeros_like(img)
    img_color[:, :, 0] = img[:, :, 2]
    img_color[:, :, 1] = img[:, :, 1]
    img_color[:, :, 2] = img[:, :, 0]
    return np.flipud(img_color)

def create_mask(mask_shape, n_rays:int, angle_ray:float = None, white_pixel_ratio:bool = False, mask_rotation_angle:float = 0.0, acyclic:bool = False, grayscale:bool = False):
    """
    Generates a radial mask with n_rays, with each ray having an opening of angle_ray degrees
        OBS: Rays are equally spaced and binary

    Input:
        mask_shape: Tuple of shape (H, W, C) or (B, H, W, C)
        n_rays: Number of white rays in the mask
        angle_ray: Angle of opening for the white rays
    """
    # If white_pixel_ratio flag is True, overwrite angle_ray so as to enforce 50% white-pixel ratio in the mask
    if white_pixel_ratio:
        angle_ray = 360.0/(4*n_rays)

    # Create blank mask
    mask = torch.ones(mask_shape[:-1])

    # Define center pixel of image (This will be the center of the coordinate system for the rays)
    xc = mask_shape[0]//2
    yc = mask_shape[1]//2
    
    # Define center line of rays
    angle_center = 360.0 / n_rays
    alphas = [((i*angle_center)+mask_rotation_angle)%360 for i in range(0, n_rays)]
    if acyclic:
        alphas = [alpha for alpha in alphas if random() <= 0.8]
    
    # Define length of the triangles' sides (must exceed the matrix coordinates)
    ray_side_height = mask_shape[0] + mask_shape[1]

    # Describe rays as triangles (3 coordinate points that extend beyond the matrix elements)
    triangles = []
    for alpha in alphas:
        x1 = ray_side_height * cos(radians(alpha) + radians(angle_ray))
        y1 = ray_side_height * sin(radians(alpha) + radians(angle_ray))
        x2 = ray_side_height * cos(radians(alpha) - radians(angle_ray))
        y2 = ray_side_height * sin(radians(alpha) - radians(angle_ray))
        triangles.append([(0, 0), (x1, y1), (x2, y2)])

    # For each pixel, check if it is inside any of the triangles
    for x in range(mask_shape[0]):
        for y in range(mask_shape[1]):
            for triangle in triangles:
                a =  (det(x-xc, y-yc, triangle[2][0], triangle[2][1]) - det(triangle[0][0], triangle[0][1], triangle[2][0], triangle[2][1]))/det(triangle[1][0], triangle[1][1], triangle[2][0], triangle[2][1])
                b = -(det(x-xc, y-yc, triangle[1][0], triangle[1][1]) - det(triangle[0][0], triangle[0][1], triangle[1][0], triangle[1][1]))/det(triangle[1][0], triangle[1][1], triangle[2][0], triangle[2][1])
                if a > 0 and b > 0 and a+b < 1:
                    mask[x, y] = 0
                    break

    if grayscale:
        for x in range(mask_shape[0]):
            for y in range(mask_shape[1]):
                if mask[x, y] != 1:
                    mask[x, y] = random()*0.5
                else:
                    mask[x, y] = random()*0.5 + 0.5
    return mask
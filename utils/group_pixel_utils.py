import torch
from torch import nn
from math import cos, sin, radians
import numpy as np
import torchvision.transforms as T
from math import ceil

def det(ux, uy, vx, vy):
    return ux*vy - uy*vx

def group_pixels(mask_shape, n_rays:int, angle_ray:float = None, white_pixel_ratio:bool = True):
    """
    Creates a dictionary grouping pixels that are in the same radial ray
        OBS: Rays are equally spaced and binary

    Input:
        mask_shape: Tuple of shape (H, W, C) or (B, H, W, C)
        n_rays: Number of white rays in the mask
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
    alphas = [((i*angle_center))%360 for i in range(0, n_rays)]
    
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

    pixel_groups = {}
    for i in range(len(triangles)):
        pixel_groups[len(pixel_groups)] = []

    # For each pixel, check if it is inside any of the triangles
    for x in range(mask_shape[0]):
        for y in range(mask_shape[1]):
            for idx, triangle in enumerate(triangles):
                a =  (det(x-xc, y-yc, triangle[2][0], triangle[2][1]) - det(triangle[0][0], triangle[0][1], triangle[2][0], triangle[2][1]))/det(triangle[1][0], triangle[1][1], triangle[2][0], triangle[2][1])
                b = -(det(x-xc, y-yc, triangle[1][0], triangle[1][1]) - det(triangle[0][0], triangle[0][1], triangle[1][0], triangle[1][1]))/det(triangle[1][0], triangle[1][1], triangle[2][0], triangle[2][1])
                if a > 0 and b > 0 and a+b < 1:
                    pixel_groups[idx].append((x, y))
                    break
    return pixel_groups



class group_pixel_mask(nn.Module):
    def __init__(self, img_shape, sigma:float=1.0, img_reso=(280, 480)):
        super(group_pixel_mask, self).__init__()
        self.img_reso=img_reso
        self.input_shape = img_shape
        self.pixel_map = group_pixels(img_shape, angle_ray=2.60, n_rays=70, white_pixel_ratio=False)
        # self.pixel_val = nn.Parameter(torch.ones(len(self.pixel_map))*0.5)
        self.pixel_val = nn.Parameter(torch.Tensor(np.random.uniform(low=0.4, high=0.6, size=[len(self.pixel_map)])))
        self.pixel_val.requires_grad = True
        self.blurer = T.GaussianBlur(2*ceil(2*sigma)+1, sigma=sigma)

    def create_mask(self):
        mask = torch.zeros(self.input_shape[:-1], device=self.pixel_val.device)
        for i in range(len(self.pixel_map)):
            for j in range(len(self.pixel_map[i])):
                x, y = self.pixel_map[i][j]
                mask[x, y] = self.pixel_val[i]
        return mask

    def forward(self, batch):
        # pad = nn.ZeroPad2d((240, 240, 140, 140))
        pad = nn.ZeroPad2d((self.img_reso[1]//2, self.img_reso[1]//2, self.img_reso[0]//2, self.img_reso[0]//2))
        mask = self.create_mask()
        mask = self.blurer(mask.unsqueeze(dim=0)).squeeze()
        # Padding inputs
        b = pad(batch)
        m = pad(mask)
        # Compute 2D FFT
        b_F = torch.fft.fft2(b)
        b_F = torch.fft.fftshift(b_F)
        m_F = torch.fft.fft2(m/m.sum())
        m_F = torch.fft.fftshift(m_F)
        # Perform convolution in FFT domain
        conv_img = b_F * m_F
        # Return to real domain
        conv_img = torch.real(torch.fft.ifftshift(torch.fft.ifft2(conv_img)))
        # Undo padding
        # conv_img = conv_img[:, :, 141:141+280, 241:241+480]
        conv_img = conv_img[:, :, self.img_reso[0]//2+1:self.img_reso[0]//2+1+self.img_reso[0], self.img_reso[1]//2+1:self.img_reso[1]//2+1+self.img_reso[1]]
        return conv_img

    def valid_forward(self, batch, mask):
        # pad = nn.ZeroPad2d((240, 240, 140, 140))
        pad = nn.ZeroPad2d((self.img_reso[1]//2, self.img_reso[1]//2, self.img_reso[0]//2, self.img_reso[0]//2))
        m = self.blurer(mask.unsqueeze(dim=0)).squeeze()
        # Padding inputs
        b = pad(batch)
        m = pad(m)
        # Compute 2D FFT
        b_F = torch.fft.fft2(b)
        b_F = torch.fft.fftshift(b_F)
        m_F = torch.fft.fft2(m/m.sum())
        m_F = torch.fft.fftshift(m_F)
        # Perform convolution in FFT domain
        conv_img = b_F * m_F
        # Return to real domain
        conv_img = torch.real(torch.fft.ifftshift(torch.fft.ifft2(conv_img)))
        # Undo padding
        # conv_img = conv_img[:, :, 141:141+280, 241:241+480]
        conv_img = conv_img[:, :, self.img_reso[0]//2+1:self.img_reso[0]//2+1+self.img_reso[0], self.img_reso[1]//2+1:self.img_reso[1]//2+1+self.img_reso[1]]
        return conv_img


class fixed_mask(nn.Module):

    def __init__(self, img_shape, sigma:float=1.0, img_reso=(280, 480), mask=None):
        super(fixed_mask, self).__init__()
        # assert mask!=None, f'Error, missing mask input'
        self.mask = mask
        self.img_reso=img_reso
        self.input_shape = img_shape
        self.blurer = T.GaussianBlur(2*ceil(2*sigma)+1, sigma=sigma)

    def forward(self, batch):
        # pad = nn.ZeroPad2d((240, 240, 140, 140))
        pad = nn.ZeroPad2d((self.img_reso[1]//2, self.img_reso[1]//2, self.img_reso[0]//2, self.img_reso[0]//2))
        mask = self.blurer(self.mask.unsqueeze(dim=0)).squeeze()
        # Padding inputs
        b = pad(batch)
        m = pad(mask)
        # Compute 2D FFT
        b_F = torch.fft.fft2(b)
        b_F = torch.fft.fftshift(b_F)
        m_F = torch.fft.fft2(m/m.sum())
        m_F = torch.fft.fftshift(m_F)
        # Perform convolution in FFT domain
        conv_img = b_F * m_F
        # Return to real domain
        conv_img = torch.real(torch.fft.ifftshift(torch.fft.ifft2(conv_img)))
        # Undo padding
        # conv_img = conv_img[:, :, 141:141+280, 241:241+480]
        conv_img = conv_img[:, :, self.img_reso[0]//2+1:self.img_reso[0]//2+1+self.img_reso[0], self.img_reso[1]//2+1:self.img_reso[1]//2+1+self.img_reso[1]]
        return conv_img


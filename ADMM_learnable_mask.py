import torch
import torchvision.transforms as T
import numpy as np
import numpy.fft as fft
from PIL import Image
import matplotlib.pyplot as plt
import joblib
from math import ceil
import cv2
from utils.utils import *
from utils.group_pixel_utils import *
import argparse
import time

parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('--mu1', action='store', dest='mu1', default=1e-6, type=float, help='ADMM hyperparameter mu1')
parser.add_argument('--mu2', action='store', dest='mu2', default=1e-6, type=float, help='ADMM hyperparameter mu2')
parser.add_argument('--mu3', action='store', dest='mu3', default=1e-6, type=float, help='ADMM hyperparameter mu3')
parser.add_argument('--tau', action='store', dest='tau', default=1e-6, type=float, help='ADMM hyperparameter tau')
parser.add_argument('--iters', action='store', dest='iters', default=50, type=int, help='Number of reconstruction iterations for ADMM algorithm')
parser.add_argument('--img_reso', action='store', dest='img_reso', default=(140,140), type=tuple, help='Resolution to be used for the images in the ADMM process')
parser.add_argument('--data_path', action='store', dest='data_path', required=True, type=str, help='Path to the image to be reconstructed')
parser.add_argument('--save_path', action='store', dest='save_path', required=True, type=str, help='Path to save the final training loss and recon results')
parser.add_argument('--mask_load_path', action='store', dest='mask_load_path', required=True, type=str, help='Path to trained radial mask')
parser_opt = parser.parse_args()

def U_update(eta, image_est, tau):
    return SoftThresh(Psi(image_est) + eta/mu2, tau/mu2)

def SoftThresh(x, tau):
    # numpy automatically applies functions to each element of the array
    return np.sign(x)*np.maximum(0, np.abs(x) - tau)

def Psi(v):
    return np.stack((np.roll(v,1,axis=0) - v, np.roll(v, 1, axis=1) - v), axis=2)

def X_update(xi, image_est, H_fft, sensor_reading, X_divmat):
    return X_divmat * (xi + mu1*M(image_est, H_fft) + CT(sensor_reading))

def M(vk, H_fft):
    return np.real(fft.fftshift(fft.ifft2(fft.fft2(fft.ifftshift(vk))*H_fft)))

def C(M):
    # Image stored as matrix (row-column rather than x-y)
    top = (full_size[0] - sensor_size[0])//2
    bottom = (full_size[0] + sensor_size[0])//2
    left = (full_size[1] - sensor_size[1])//2
    right = (full_size[1] + sensor_size[1])//2
    return M[top:bottom,left:right]

def CT(b):
    v_pad = (full_size[0] - sensor_size[0])//2
    h_pad = (full_size[1] - sensor_size[1])//2
    return np.pad(b, ((v_pad, v_pad), (h_pad, h_pad)), 'constant',constant_values=(0,0))

def precompute_X_divmat(): 
    """Only call this function once! 
    Store it in a variable and only use that variable 
    during every update step"""
    return 1./(CT(np.ones(sensor_size)) + mu1)

def W_update(rho, image_est):
    return np.maximum(rho/mu3 + image_est, 0)

def r_calc(w, rho, u, eta, x, xi, H_fft):
    return (mu3*w - rho)+PsiT(mu2*u - eta) + MT(mu1*x - xi, H_fft)

def V_update(w, rho, u, eta, x, xi, H_fft, R_divmat):
    freq_space_result = R_divmat*fft.fft2( fft.ifftshift(r_calc(w, rho, u, eta, x, xi, H_fft)) )
    return np.real(fft.fftshift(fft.ifft2(freq_space_result)))

def PsiT(U):
    diff1 = np.roll(U[...,0],-1,axis=0) - U[...,0]
    diff2 = np.roll(U[...,1],-1,axis=1) - U[...,1]
    return diff1 + diff2

def MT(x, H_fft):
    x_zeroed = fft.ifftshift(x)
    return np.real(fft.fftshift(fft.ifft2(fft.fft2(x_zeroed) * np.conj(H_fft))))

def precompute_PsiTPsi():
    PsiTPsi = np.zeros(full_size)
    PsiTPsi[0,0] = 4
    PsiTPsi[0,1] = PsiTPsi[1,0] = PsiTPsi[0,-1] = PsiTPsi[-1,0] = -1
    PsiTPsi = fft.fft2(PsiTPsi)
    return PsiTPsi

def precompute_R_divmat(H_fft, PsiTPsi): 
    """Only call this function once! 
    Store it in a variable and only use that variable 
    during every update step"""
    MTM_component = mu1*(np.abs(np.conj(H_fft)*H_fft))
    PsiTPsi_component = mu2*np.abs(PsiTPsi)
    id_component = mu3
    """This matrix is a mask in frequency space. So we will only use
    it on images that have already been transformed via an fft"""
    return 1./(MTM_component + PsiTPsi_component + id_component)

def xi_update(xi, V, H_fft, X):
    return xi + mu1*(M(V,H_fft) - X)

def eta_update(eta, V, U):
    return eta + mu2*(Psi(V) - U)

def rho_update(rho, V, W):
    return rho + mu3*(V - W)

def init_Matrices(H_fft):
    X = np.zeros(full_size)
    U = np.zeros((full_size[0], full_size[1], 2))
    V = np.zeros(full_size)
    W = np.zeros(full_size)

    xi = np.zeros_like(M(V,H_fft))
    eta = np.zeros_like(Psi(V))
    rho = np.zeros_like(W)
    return X,U,V,W,xi,eta,rho

def precompute_H_fft(psf):
    return fft.fft2(fft.ifftshift(CT(psf)))

def ADMMStep(X,U,V,W,xi,eta,rho, precomputed):
    H_fft, data, X_divmat, R_divmat = precomputed
    U = U_update(eta, V, tau)
    X = X_update(xi, V, H_fft, data, X_divmat)
    V = V_update(W, rho, U, eta, X, xi, H_fft, R_divmat)
    W = W_update(rho, V)
    xi = xi_update(xi, V, H_fft, X)
    eta = eta_update(eta, V, U)
    rho = rho_update(rho, V, W)
    
    return X,U,V,W,xi,eta,rho

def runADMM(psf, data):
    H_fft = precompute_H_fft(psf)
    X,U,V,W,xi,eta,rho = init_Matrices(H_fft)
    X_divmat = precompute_X_divmat()
    PsiTPsi = precompute_PsiTPsi()
    R_divmat = precompute_R_divmat(H_fft, PsiTPsi)
    
    for i in range(iters):
        X,U,V,W,xi,eta,rho = ADMMStep(X,U,V,W,xi,eta,rho, [H_fft, data, X_divmat, R_divmat])
    image = C(V)
    image[image<0] = 0
    return image

if __name__ == '__main__':
    print(f'Experiment: {parser_opt.save_path}')
    # Load target image (for reconstruction)
    t0 = time.time()
    data = np.load(parser_opt.data_path)
    data = cv2.resize(data, parser_opt.img_reso, interpolation=cv2.INTER_AREA)
    channel_0 = data[:, :, 0] / np.linalg.norm(data[:, :, 0])
    channel_1 = data[:, :, 1] / np.linalg.norm(data[:, :, 1])
    channel_2 = data[:, :, 2] / np.linalg.norm(data[:, :, 2])
    target = np.zeros_like(data)
    target[:, :, 0] = data[:, :, 2]
    target[:, :, 1] = data[:, :, 1]
    target[:, :, 2] = data[:, :, 0]
    # data /= np.linalg.norm(data.ravel())
    sigma = 0.5
    blurer = T.GaussianBlur(2*ceil(2*sigma)+1, sigma=sigma)
    # Create Radial mask (Gaussian filter used to simulate light diffraction)
    # psf = create_mask(mask_shape=(parser_opt.img_reso[0], parser_opt.img_reso[1], 3), n_rays=18, white_pixel_ratio=True)
    RadialMaskModel = group_pixel_mask(img_shape=[parser_opt.img_reso[0], parser_opt.img_reso[1], 1], sigma=0.5, img_reso=parser_opt.img_reso)
    RadialMaskModel.load_state_dict(torch.load(parser_opt.mask_load_path + 'best_mask.pt', map_location='cpu'))
    RadialMaskModel.pixel_val.requires_grad = False
    mask = RadialMaskModel.create_mask()
    mask = mask.numpy()
    psf = blurer(torch.Tensor(mask).unsqueeze(dim=0)).squeeze().numpy()
    psf /= np.linalg.norm(psf.ravel())

    sensor_size = np.array(psf.shape)
    full_size = 2*sensor_size

    # ADMM hyperparameters
    mu1 = parser_opt.mu1
    mu2 = parser_opt.mu2
    mu3 = parser_opt.mu3
    tau = parser_opt.tau
    iters = parser_opt.iters

    # Compute sensor measurements
    c0_meas = C(M(CT(channel_0), precompute_H_fft(psf)))
    c1_meas = C(M(CT(channel_1), precompute_H_fft(psf)))
    c2_meas = C(M(CT(channel_2), precompute_H_fft(psf)))

    # Reconstruct RGB channels independently
    c0_recon = runADMM(psf, c0_meas)
    c1_recon = runADMM(psf, c1_meas)
    c2_recon = runADMM(psf, c2_meas)

    # Group reconstructed channels
    recon = np.zeros_like(data)
    recon[:, :, 0] = c2_recon / c2_recon.max()
    recon[:, :, 1] = c1_recon / c1_recon.max()
    recon[:, :, 2] = c0_recon / c0_recon.max()

    loss = torch.nn.functional.mse_loss(torch.Tensor(recon), torch.Tensor(target))
    joblib.dump({'Target': target,'Recon': recon, 'Loss': loss}, parser_opt.save_path)
    t1 = time.time()
    print(f'Running took {t1-t0} seconds')
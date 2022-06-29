import numpy as np
import numpy.fft as fft
from PIL import Image
import matplotlib.pyplot as plt
from IPython import display
import cv2
import argparse
from utils.utils import *
import joblib


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
        if i % 1 == 0:
            image = C(V)
            image[image<0] = 0
            # f = plt.figure(1)
            # plt.imshow(image, cmap='gray')
            # plt.title('Reconstruction after iteration {}'.format(i))
            # display.display(f)
            # display.clear_output(wait=True)
    return image

if __name__ == '__main__':
    # Load target image (for reconstruction)
    data = np.load('../../Desktop/datasets/LenslessLearning/ground_truth_lensed/im65.npy')
    data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    data /= np.linalg.norm(data.ravel())

    # Create Radial mask
    psf = create_mask(mask_shape=(270, 480, 3), n_rays=24, white_pixel_ratio=True)
    # psf /= psf.sum()
    psf /= np.linalg.norm(psf.ravel())

    sensor_size = np.array(psf.shape)
    full_size = 2*sensor_size

    sensor_meas = C(M(CT(data), precompute_H_fft(psf)))

    mu1 = 1e-6
    mu2 = 1e-5
    mu3 = 4e-5
    tau = 0.0001

    # Number of iterations
    iters = 50
    
    best_loss = 1000
    best_params = [0, 0, 0, 0]
    # for mu1 in [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
    #     for mu2 in [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
    #         for mu3 in [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
    #             for tau in [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
    for mu1 in [1e-10, 1e-9, 1e-7, 1e-5, 1e-3]:
        for mu2 in [1e-10, 1e-9, 1e-7, 1e-5, 1e-3]:
            for mu3 in [1e-10, 1e-9, 1e-7, 1e-5, 1e-3]:
                for tau in [1e-10, 1e-9, 1e-7, 1e-5, 1e-3]:
                    print(f'Exp {mu1} {mu2} {mu3} {tau}')
                    final_im = runADMM(psf, sensor_meas)
                    temp_loss = torch.nn.functional.mse_loss(torch.Tensor(final_im), torch.Tensor(data))
                    if temp_loss < best_loss:
                        best_loss = temp_loss
                        best_params = [mu1, mu2, mu3, tau]
                        best_img = final_im

    # Save best hyperparameters
    print(f'Best parameters:')
    print(f'Mu1: {best_params[0]}')
    print(f'Mu2: {best_params[1]}')
    print(f'Mu3: {best_params[2]}')
    print(f'Tau: {best_params[3]}')
    joblib.dump(best_params, 'best_params.pkl')
    # Plot and save best grayscale reconstruction
    plt.imshow(best_img, cmap='gray')
    plt.title(f'Best reconstruction')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.savefig('Best.png')

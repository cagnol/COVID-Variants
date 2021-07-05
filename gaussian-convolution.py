"""gaussian-convolution.py: Checks the computation of 4th-order tensor P is consistant with the multiplication of the Toeplitz matrices and verify P..I can be achieved with gaussian_blur"""

__author__    = "Christian Bongiorno and John Cagnol"
__copyright__ = "Copyright 2021 Christian Bongiorno and John Cagnol"
__license__   = "MIT"


import math
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.special import erf
from scipy.ndimage import gaussian_filter
from matplotlib import image



# Parameter for the convolution

sigma = 1.0


# Define the fourth-order tensor P

def P_tensor ():
    return np.fromfunction(lambda i,j,k,l : 1/(2*np.pi*sigma**2)*np.exp(-((i-k)**2+(j-l)**2)/(2*sigma**2)), (N,M,N,M))


# Define the Toeplitz matrix corresponding to the Gaussian convolution

def Toeplitz (order):
    return np.fromfunction(lambda i, k: 1/((2*np.pi)**(1/2)*sigma)*np.exp(-((i-k)**2)/(2*sigma**2)), (order,order))


# Load image I

start_time = time.time()
I = image.imread('lena-orig.png')
N,M = I.shape
print("Image Loaded (%dx%d) in %d seconds"%(N,M,time.time() - start_time))


# Save image with the desired formatting

plt.imshow(I,cmap='gray')
plt.colorbar()
plt.savefig("image.png")
plt.clf()


# Computing tensor P

start_time = time.time()
P=P_tensor()
print("Tensor P computed in %d second for method 1"%(time.time() - start_time))

# Veryfing P can also be expressed with Toeplitz matrices

start_time = time.time()
P_tensor_from_Toeplitz = np.transpose(np.tensordot(Toeplitz(N),Toeplitz(M),0),(0,2,1,3))
print("Tensor P computed in %d second for method 2"%(time.time() - start_time))

print("Maximum difference in computing P by the two methods: = %e"%abs(P-P_tensor_from_Toeplitz).max())


# Create a save the image P..I

start_time = time.time()
image_gaussian_1 = np.tensordot(P,I,2)
print("Gaussian convolution computed in %d second for P..I"%(time.time() - start_time))
plt.imshow(image_gaussian_1,cmap='gray',vmin=0,vmax=1)
plt.colorbar()
plt.savefig("image_gaussian_1.png")
plt.clf()


# Create a save the Gaussian filter of I (Python function)

start_time = time.time()
image_gaussian_2 = gaussian_filter(I,sigma,mode='constant',cval=0)
print("Gaussian convolution computed in %d second with the Python routine"%(time.time() - start_time))
plt.imshow(image_gaussian_2,cmap='gray',vmin=0,vmax=1)
plt.colorbar()
plt.savefig("image_gaussian_2.png")
plt.clf()


# Check the difference between the two methods

image_gaussian_delta = abs(image_gaussian_1 - image_gaussian_2)
plt.imshow(image_gaussian_delta,cmap='jet')
plt.colorbar()
plt.savefig("image_gaussian_delta.png")
plt.clf()

print("Maximum difference in computing the Gaussian convolution by the two methods: %e"%image_gaussian_delta.max())

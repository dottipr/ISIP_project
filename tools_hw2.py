"""
Contains functions that are mainly used in all exercises of HW2
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import time
from scipy.misc import imresize
from scipy.signal import convolve2d

def gauss1d(sigma, filter_length=10):
    # INPUTS
    # @ sigma         : standard deviation of gaussian distribution
    # @ filter_length : integer denoting the filter length, default is 10
    # OUTPUTS
    # @ gauss_filter  : 1D gaussian filter

    # if filter_length is even add one
    filter_length += ~filter_length % 2
    x = np.linspace(np.int(-filter_length/2),np.int(filter_length/2), filter_length)

    gauss_filter = np.exp(- (x ** 2) / (2 * (sigma ** 2)))

    gauss_filter = gauss_filter / np.sum(gauss_filter)

    return gauss_filter

def gauss2d(sigma, filter_size=10):
    # INPUTS
    # @ sigma           : standard deviation of gaussian distribution
    # @ filter_size     : integer denoting the filter size, default is 10
    # OUTPUTS
    # @ gauss2d_filter  : 2D gaussian filter

    # create a 1D gaussian filter
    gauss1d_filter = gauss1d(sigma, filter_size)[np.newaxis, :]
    # convolve it with its transpose
    gauss2d_filter = convolve2d(gauss1d_filter, np.transpose(gauss1d_filter))

    return gauss2d_filter

def myconv(signal, filt):
    # This function performs a 1D convolution between signal and filt. This
    # function should return the result of a 1D convolution of the signal and
    # the filter.
    # INPUTS
    # @ signal          : 1D image, as numpy array, of length m
    # @ filt            : 1D or 2D filter of length k
    # OUTPUTS
    # signal_filtered   : 1D filtered signal, of size (m+k-1)

    m = len(signal)
    k = len(filt)

    padding_lenght = k-1
    signal = np.pad(signal, padding_lenght, 'constant')

    filt = np.flip(filt,0)
    conv_signal = np.zeros_like(signal)

    for i in range(k//2,padding_lenght+m+k//2):
        box = signal[i-k//2:i+(k+1)//2]
        conv_signal[i] = sum(box*filt)

    return conv_signal[k//2:padding_lenght+m+k//2]

def myconv2(img, filt):
    # This function performs a 2D convolution between image and filt, image being a 2D image. This
    # function should return the result of a 2D convolution of these two
    # images. NOTE: If you can not implement 2D convolution, you can use
    # scipy.signal.convolve2d() in order to be able to continue with the other exercises.
    # INPUTS
    # @ img           : 2D image, as numpy array, size mxn
    # @ filt          : 1D or 2D filter of size kxl
    # OUTPUTS
    # img_filtered    : 2D filtered image, of size (m+k-1)x(n+l-1)

    m, n = len(img), len(img[0])
    k, l = len(filt), len(filt[0])

    padding0 = k-1
    padding1 = l-1
    img = np.pad(img, ((padding0,padding0),(padding1,padding1)), 'constant')

    filt = np.flip(filt,0)
    filt = np.flip(filt,1)

    conv_img = np.zeros_like(img)

    for i in range(k//2,padding0+m+k//2):
        for j in range(l//2,padding1+n+l//2):
            box = img[i-k//2:i+(k+1)//2, j-l//2:j+(l+1)//2]
            conv_img[i,j] = np.sum(filt*box)

    return conv_img[k//2:padding0+m+k//2,l//2:padding1+n+l//2]

def gconv(img, sigma, filter_size):
    # Function that filters an image with a Gaussian filter
    # INPUTS
    # @ img           : 2D image
    # @ sigma         : the standard deviation of gaussian distribution
    # @ size          : the size of the filter
    # OUTPUTS
    # @ img_filtered  : filtered image with gaussian filter

    filter = gauss2d(sigma, filter_size)

    return convolve2d(img, filter, mode='valid')

def DoG(img, sigma_1, sigma_2, filter_size):
    # Function that creates Difference of Gaussians (DoG) for given standard
    # deviations and filter size
    # INPUTS
    # @ img
    # @ img           : 2D image (MxN)
    # @ sigma_1       : standard deviation of of the first Gaussian distribution
    # @ sigma_2       : standard deviation of the second Gaussian distribution
    # @ filter_size   : the size of the filters
    # OUTPUTS
    # @ dog           : Difference of Gaussians of size
    #                   (M+filter_size-1)x(N_filter_size-1)
    if sigma_1 > sigma_2:
        t = sigma_1
        sigma_1 = sigma_2
        sigma_2 = t

    filtered1 = gconv(img, sigma_1, filter_size)
    filtered2 = gconv(img, sigma_2, filter_size)

    return filtered2-filtered1

def blur_and_downsample(img, sigma, filter_size, scale):
    # INPUTS
    # @ img                 : 2D image (MxN)
    # @ sigma               : standard deviation of the Gaussian filter to be used at
    #                         all levels
    # @ filter_size         : the size of the filters
    # @ scale               : Downscaling factor (of type float, between 0-1)
    # OUTPUTS
    # @ img_br_ds           : The blurred and downscaled image

    img = gconv(img, sigma, filter_size)
    img = imresize(img, scale)

    return img

def generate_gaussian_pyramid(img, sigma, filter_size, scale, num_levels):
    # Function that creates Gaussian Pyramid as described in the homework
    # It blurs and downsacle the iimage subsequently. Please keep in mind that
    # the first element of the pyramid is the oirignal image, which is
    # considered as the level-0. The number of levels that is given as argument
    # INCLUDES the level-0 as well. It means there will be num_levels-1 times
    # blurring and down_scaling.
    # INPUTS
    # @ img                 : 2D image (MxN)
    # @ sigma               : standard deviation of the Gaussian filter to be used at
    #                         all levels
    # @ filter_size         : the size of the filters
    # @ scale               : Downscaling factor (of type float, between 0-1)
    # OUTPUTS
    # @ gaussian_pyramid    : A list connatining images of pyramid. The first
    #                         element SHOULD be the image at the original scale
    #                         without any blurring.

    pyramid = [img]
    level = img

    for i in range(num_levels-1):
        level = blur_and_downsample(level, sigma, filter_size, scale)
        pyramid.append(level)

    return pyramid

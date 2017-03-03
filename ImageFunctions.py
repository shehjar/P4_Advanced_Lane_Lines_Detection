# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:04:46 2017

@author: admin
"""

import cv2
import numpy as np

def Color_thresholding(img, sthresh=(0,255), lthresh = (0,255)):
    HLS_image = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    S_channel = HLS_image[:,:,2]
    s_binary_image = np.zeros_like(S_channel)
    s_binary_image[(S_channel >= sthresh[0]) & (S_channel <= sthresh[1])] = 1
    L_channel = HLS_image[:,:,1]
    l_binary_image = np.zeros_like(L_channel)
    l_binary_image[(L_channel >= lthresh[0]) & (L_channel <= lthresh[1])] = 1
    binary_image = np.zeros_like(l_binary_image)
    binary_image[(l_binary_image == 1) & (s_binary_image == 1)] = 1
    return binary_image

def Sobel_mag_thresholding(img,kernel = 3, thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = kernel)
    abs_sobel = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_image = np.zeros_like(scaled_sobel)
    binary_image[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_image

def Sobel_abs_thresholding(img, orient = 'x', kernel = 3, thresh = (0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_image = np.zeros_like(scaled_sobel)
    binary_image[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_image

def Sobel_dir_thresholding(img, kernel = 3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = kernel)
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    dir_sobel = np.arctan2(abs_sobel_y,abs_sobel_x)
    binary_image = np.zeros_like(dir_sobel)
    binary_image[(dir_sobel >= thresh[0]) & (dir_sobel <= thresh[1])] = 1
    return binary_image
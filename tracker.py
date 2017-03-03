# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 18:15:11 2017

@author: admin
"""
import pickle
import cv2
import numpy as np
class tracker():
    
    def __init__(self, Mywindow_width, Mywindow_height, Mymargin, My_ym = 1, My_xm = 1, Mysmooth_factor = 15):
        # list of all the past centers| used for smoothing the output
        self.recent_centers = []
        # the window pixel width of the center values| used to count pixels inside center windows to determine curve values
        self.window_width = Mywindow_width
        # the window pixel height of the center values| used to count pixels inside center windows to determine curve values
        # breaks the image into vertical levels
        self.window_height = Mywindow_height
        
        # The pixel distance in both directions to slide (left_window + right_window) template for searching
        self.margin = Mymargin
        
        self.ym_per_pix = My_ym
        self.xm_per_pix = My_xm
        self.smooth_factor = Mysmooth_factor
        
    def find_window_centroids(self,warped):
        
        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin
        
        window_centroids = []                # Store center data per level
        window = np.ones(window_width)      # template for Convolution
        
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 
    
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
        l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
        r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
        
        # Add what we found for the first layer
        window_centroids.append((l_center,r_center))
        
        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(warped.shape[0]/window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
            if np.count_nonzero(conv_signal[l_min_index:l_max_index]) == 0:
                l_center = window_centroids[-1][0]
            else:
                l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center+offset-margin,0))
            r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
            if np.count_nonzero(conv_signal[r_min_index:r_max_index]) == 0:
                r_center = window_centroids[-1][1]
            else:
                r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            # Add what we found for that layer
            window_centroids.append((l_center,r_center))
        self.recent_centers.append(window_centroids)
        return np.average(self.recent_centers[-self.smooth_factor:], axis = 0)
    
class pTransform():
    def __init__(self, src_pts, dst_pts):
        self.src = src_pts
        self.dst = dst_pts
        self.M = cv2.getPerspectiveTransform(np.float32(self.src),np.float32(self.dst))
        self.Minv = cv2.getPerspectiveTransform(np.float32(self.dst),np.float32(self.src))
        
    def warped(self,img):
        M = self.M
        shape = (img.shape[1],img.shape[0])
        warped_img = cv2.warpPerspective(img,M,shape,flags=cv2.INTER_LINEAR)
        return warped_img
        
    def invWarped(self, img):
        Minv = self.Minv
        shape = (img.shape[1],img.shape[0])
        inv_warped_img = cv2.warpPerspective(img,Minv,shape,flags=cv2.INTER_LINEAR)
        return inv_warped_img
    
class Calibration():
    def __init__(self,pfile):
        dist_dir = pickle.load(open(pfile,'rb'))
        dist = dist_dir['dist']
        mtx = dist_dir['mtx']
        self.dist = dist
        self.mtx = mtx
        
    def undistort(self, img):
        undistort_img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undistort_img

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

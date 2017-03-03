# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
curfile = os.getcwd()
srcfile = os.path.join(curfile,'camera_cal')
nx = 9
ny = 6

# Creating object points data
objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

imagePaths = glob.glob(os.path.join(srcfile,'calibration*.jpg'))
# Step through the list and search for chessboard corners
for idx, imagePath in enumerate(imagePaths):
    img = cv2.imread(imagePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        write_name = 'corners_found'+str(idx)+'.jpg'
        savefile = os.path.join(srcfile,write_name)
        cv2.imwrite(savefile, img)
        #cv2.imshow('img', img)
        #cv2.waitKey(500)

# Do camera calibration given object points and image points
img_size = (img.shape[1],img.shape[0])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open(os.path.join(srcfile,"image_distortion_matrices.p"), "wb"))

# Visualize Distortion
rand_index = np.random.randint(len(imagePaths))
vis_img = cv2.imread(imagePaths[rand_index])
undistort_vis_img = cv2.undistort(vis_img, mtx, dist, None, mtx)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
vis_img = cv2.cvtColor(vis_img,cv2.COLOR_BGR2RGB)
ax1.imshow(vis_img)
ax1.set_title('Original Image', fontsize=30)
undistort_vis_img = cv2.cvtColor(undistort_vis_img,cv2.COLOR_BGR2RGB)
ax2.imshow(undistort_vis_img)
ax2.set_title('Undistorted Image', fontsize=30)
plt.savefig(os.path.join(srcfile,'Undistortion_Example.jpg'))
